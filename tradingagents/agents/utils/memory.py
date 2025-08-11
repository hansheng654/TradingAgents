import numpy as np
import hashlib
import threading
from collections import OrderedDict
from typing import List, Tuple
import time

try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except Exception:
    _TIKTOKEN_AVAILABLE = False

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# Safe embedding chunking for models with ~8k token limits
_EMBED_MAX_TOKENS = 8000  # keep a little headroom under 8192
_CACHE_MAX = 512  # max cached embeddings per process
_NORM_VER = "v1"  # bump when normalization logic changes


def _get_tokenizer():
    if _TIKTOKEN_AVAILABLE:
        # text-embedding-3-* use cl100k_base
        return tiktoken.get_encoding("cl100k_base")
    # Fallback: whitespace tokenization approximation
    return None


def _count_tokens(text: str) -> int:
    tok = _get_tokenizer()
    if tok is None:
        return len(text.split())
    return len(tok.encode(text))


def _chunk_by_tokens(text: str, max_tokens: int = _EMBED_MAX_TOKENS):
    """
    DEPRECATED: kept for backward-compat. Prefer `_chunk_for_embedding`, which yields (text, token_count).
    This wrapper now delegates to `_chunk_for_embedding` and yields only the chunk text.
    """
    for chunk_text, _tc in _chunk_for_embedding(text, max_tokens):
        yield chunk_text


def _weighted_mean_pool(chunks: List[Tuple[List[float], int]]) -> List[float]:
    """
    Token-weighted mean pooling.
    chunks: list of (vector, token_count)
    """
    if not chunks:
        return []
    vecs = []
    weights = []
    for vec, tok in chunks:
        if vec is None:
            continue
        vecs.append(np.array(vec, dtype=float))
        weights.append(max(1, int(tok)))
    if not vecs:
        return []
    weights_arr = np.array(weights, dtype=float)
    stacked = np.vstack(vecs)
    weighted = (stacked.T @ weights_arr) / weights_arr.sum()
    return weighted.tolist()


def _normalize_text(text: str) -> str:
    # Deterministic, minimal normalization. Adjust as needed but keep stable.
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return "\n".join(lines)


def _embed_cache_key(provider: str, model: str, norm_ver: str, text: str) -> str:
    payload = f"{provider}|{model}|{norm_ver}|{_normalize_text(text)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Provider-aware chunking yielding (chunk_text, token_count)
def _chunk_for_embedding(text: str, max_tokens: int = _EMBED_MAX_TOKENS):
    """
    Yield (chunk_text, token_count) pairs. Falls back gracefully when tokenizer is unavailable.
    """
    tok = _get_tokenizer()
    if tok is None:
        # crude fallback by words ~ treat words as tokens
        words = text.split()
        for i in range(0, len(words), max_tokens):
            chunk_words = words[i : i + max_tokens]
            yield " ".join(chunk_words), len(chunk_words)
        return
    ids = tok.encode(text)
    for i in range(0, len(ids), max_tokens):
        chunk_ids = ids[i : i + max_tokens]
        yield tok.decode(chunk_ids), len(chunk_ids)


class FinancialSituationMemory:
    def __init__(self, name, config):
        self.config = config
        self.backend_url = config["backend_url"]

        # Determine which API to use based on backend_url
        if (
            "openai" in self.backend_url.lower()
            or "localhost:11434" in self.backend_url
        ):
            self.api_type = "openai"
            if config["backend_url"] == "http://localhost:11434/v1":
                self.embedding = "nomic-embed-text"
            else:
                self.embedding = "text-embedding-3-small"
            self.client = OpenAI(base_url=config["backend_url"])
        elif (
            "google" in self.backend_url.lower()
            or "generativeai" in self.backend_url.lower()
        ):
            self.api_type = "google"
            self.embedding = (
                "models/text-embedding-004"  # Google's latest embedding model
            )
            # Uses GOOGLE_API_KEY environment variable automatically
            self.google_embeddings = GoogleGenerativeAIEmbeddings(
                model=self.embedding, transport="rest", request_options={"timeout": 120}
            )
            self.client = None  # Google doesn't use the same client pattern
        else:
            raise ValueError(f"Unsupported backend URL: {self.backend_url}")

        # --- Embedding cache & metrics ---
        self._embed_cache = OrderedDict()  # key -> vector
        self._cache_lock = threading.Lock()
        self.metrics = {"embed_calls": 0, "cache_hits": 0}

        # --- Vector store ---
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        # use get_or_create to avoid accidental resets across runs
        try:
            self.situation_collection = self.chroma_client.get_or_create_collection(
                name=name
            )
        except Exception:
            # fallback for older chromadb versions
            self.situation_collection = self.chroma_client.create_collection(name=name)

    def get_embedding(self, text):
        """Get embedding for text using the configured API with caching and safe chunking."""
        start_time = time.time()
        norm_text = _normalize_text(text or "")
        # Determine a conservative per-provider chunk size (tokens)
        # Keep 8k default for OpenAI; keep 8k-ish approximation for Google as we count with tiktoken fallback.
        max_tokens = _EMBED_MAX_TOKENS

        # Cache lookup
        cache_key = _embed_cache_key(
            self.api_type, self.embedding, _NORM_VER, norm_text
        )
        with self._cache_lock:
            if cache_key in self._embed_cache:
                # move to end (LRU)
                self._embed_cache.move_to_end(cache_key)
                self.metrics["cache_hits"] += 1
                print(f"[TIMING] Embedding cache hit: {time.time() - start_time:.2f}s")
                return self._embed_cache[cache_key]

        # Helper to insert into cache with LRU eviction
        def _cache_put(key, value):
            with self._cache_lock:
                self._embed_cache[key] = value
                self._embed_cache.move_to_end(key)
                if len(self._embed_cache) > _CACHE_MAX:
                    self._embed_cache.popitem(last=False)  # evict LRU

        # Compute embedding
        try:
            total_tokens = _count_tokens(norm_text)
            if total_tokens <= max_tokens:
                if self.api_type == "openai":
                    api_start = time.time()
                    resp = self.client.embeddings.create(
                        model=self.embedding, input=norm_text
                    )
                    print(f"[TIMING] Embedding API call: {time.time() - api_start:.2f}s")
                    vec = resp.data[0].embedding
                elif self.api_type == "google":
                    api_start = time.time()
                    vecs = self.google_embeddings.embed_documents([norm_text])
                    print(f"[TIMING] Embedding API call: {time.time() - api_start:.2f}s")
                    vec = vecs[0]
                else:
                    raise ValueError(f"Unsupported api_type: {self.api_type}")
                self.metrics["embed_calls"] += 1
                _cache_put(cache_key, vec)
                print(f"[TIMING] Total embedding time: {time.time() - start_time:.2f}s")
                return vec

            # Oversized: chunk + token-weighted pool
            chunk_vecs: List[Tuple[List[float], int]] = []
            for chunk_text, tok_count in _chunk_for_embedding(norm_text, max_tokens):
                if self.api_type == "openai":
                    api_start = time.time()
                    resp = self.client.embeddings.create(
                        model=self.embedding, input=chunk_text
                    )
                    print(f"[TIMING] Embedding API call: {time.time() - api_start:.2f}s")
                    vec = resp.data[0].embedding
                elif self.api_type == "google":
                    api_start = time.time()
                    vecs = self.google_embeddings.embed_documents([chunk_text])
                    print(f"[TIMING] Embedding API call: {time.time() - api_start:.2f}s")
                    vec = vecs[0]
                else:
                    raise ValueError(f"Unsupported api_type: {self.api_type}")
                self.metrics["embed_calls"] += 1
                chunk_vecs.append((vec, tok_count))

            pooled = _weighted_mean_pool(chunk_vecs)
            _cache_put(cache_key, pooled)
            print(f"[TIMING] Total embedding time: {time.time() - start_time:.2f}s")
            return pooled
        except Exception as e:
            print(f"[TIMING] Embedding failed after {time.time() - start_time:.2f}s")
            # Do not cache failures; surface the error
            print(f"Embedding error ({self.api_type}/{self.embedding}): {e}")
            raise

    def clear_embedding_cache(self):
        """Clear in-process embedding cache and metrics (call on ticker change if desired)."""
        with self._cache_lock:
            self._embed_cache.clear()
        self.metrics = {"embed_calls": 0, "cache_hits": 0}

    def add_situations(self, situations_and_advice):
        """Add financial situations and their corresponding advice. Parameter is a list of tuples (situation, rec)"""
        situations = []
        advice = []
        ids = []
        embeddings = []

        offset = 0
        try:
            offset = self.situation_collection.count()
        except Exception:
            # best-effort; continue without offset if count fails
            offset = 0

        for i, (situation, recommendation) in enumerate(situations_and_advice):
            situations.append(situation)
            advice.append(recommendation)
            ids.append(str(offset + i))
            embeddings.append(self.get_embedding(situation))

        try:
            self.situation_collection.add(
                documents=situations,
                metadatas=[{"recommendation": rec} for rec in advice],
                embeddings=embeddings,
                ids=ids,
            )
        except Exception as e:
            print(f"Chroma add error: {e}")
            raise

    def get_memories(self, current_situation, n_matches=1):
        """Find matching recommendations using configured embeddings API"""
        query_embedding = self.get_embedding(current_situation)
        try:
            results = self.situation_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_matches,
                include=["metadatas", "documents", "distances"],
            )
        except Exception as e:
            print(f"Chroma query error: {e}")
            raise

        matched_results = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for i in range(len(docs)):
            meta = metas[i] if i < len(metas) else {}
            dist = dists[i] if i < len(dists) else 1.0
            matched_results.append(
                {
                    "matched_situation": docs[i],
                    "recommendation": meta.get("recommendation"),
                    "similarity_score": 1 - dist,
                }
            )

        return matched_results


if __name__ == "__main__":
    """
    Lightweight self-tests for the FinancialSituationMemory when this file is executed directly.
    - Runs against OpenAI if OPENAI_API_KEY is set; otherwise tries Google if GOOGLE_API_KEY is set.
    - Verifies cache hits vs. API calls.
    - Forces chunking with a tiny token limit, and checks multiple API calls occur.
    - Adds/queries a few situations to make sure vector store path is alive.
    """
    import os
    import time

    use_openai = bool(os.getenv("OPENAI_API_KEY"))
    use_google = bool(os.getenv("GOOGLE_API_KEY"))

    if not (use_openai or use_google):
        print(
            "No API key detected. Set OPENAI_API_KEY or GOOGLE_API_KEY to run live self-tests."
        )
        # Still proceed to run the vector store path with a fake backend URL (will not embed)
        # Exit early to avoid network calls.
        exit(0)

    backend_url = (
        "https://api.openai.com/v1"
        if use_openai
        else "https://generativeai.googleapis.com/"
    )
    provider = "OpenAI" if use_openai else "Google"
    print(f"== Running self-tests using {provider} backend ==")

    cfg = {"backend_url": backend_url}
    mem = FinancialSituationMemory("selftest_memory", cfg)
    mem.clear_embedding_cache()

    # --- Test 1: Cache hit for short text ---
    text_short = "Hello world, this is a short text for caching test."
    print("\n[TEST 1] cache hits vs. API calls")
    v1 = mem.get_embedding(text_short)
    v2 = mem.get_embedding(text_short)  # should be a cache hit
    print("  Metrics:", mem.metrics)
    assert isinstance(v1, list) and len(v1) > 0, "Embedding vector should be non-empty"
    assert v1 == v2, "Second call should return cached vector"
    assert mem.metrics["cache_hits"] == 1, "Expected one cache hit"
    assert mem.metrics["embed_calls"] == 1, "Expected only one API call for same text"

    # --- Test 2: Force chunking + weighted pooling ---
    print("\n[TEST 2] forced chunking & weighted pooling")
    long_text = " ".join(
        [f"word{i}" for i in range(1, 61)]
    )  # 60 space-delimited tokens
    # Save originals
    _orig_token_limit = _EMBED_MAX_TOKENS
    _orig_get_tok = _get_tokenizer
    try:
        # Force small chunking window
        globals()["_EMBED_MAX_TOKENS"] = 5

        # Replace tokenizer with a deterministic whitespace tokenizer
        def _fake_tok():
            class _Tok:
                def encode(self, s):
                    return s.split()

                def decode(self, ids):
                    return " ".join(ids)

            return _Tok()

        globals()["_get_tokenizer"] = _fake_tok

        before_calls = mem.metrics["embed_calls"]
        vec = mem.get_embedding(long_text)
        after_calls = mem.metrics["embed_calls"]
        print("  embed calls during chunking:", after_calls - before_calls)
        assert (
            after_calls - before_calls >= 2
        ), "Expected multiple API calls due to chunking"
        assert (
            isinstance(vec, list) and len(vec) > 0
        ), "Pooled vector should be non-empty"
    finally:
        # Restore globals
        globals()["_EMBED_MAX_TOKENS"] = _orig_token_limit
        globals()["_get_tokenizer"] = _orig_get_tok

    # --- Test 3: Add situations and query memories ---
    print("\n[TEST 3] vector store add/query")
    example_data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates.",
        ),
    ]
    mem.add_situations(example_data)

    current_situation = (
        "Market showing increased volatility in tech sector, with institutional investors "
        "reducing positions and rising interest rates affecting growth stock valuations"
    )

    recs = mem.get_memories(current_situation, n_matches=2)
    for i, rec in enumerate(recs, 1):
        print(f"\n  Match {i}:")
        print(f"    Similarity Score: {rec['similarity_score']:.3f}")
        print(f"    Recommendation:  {rec['recommendation']}")

    print("\n== Self-tests complete ==")
    print("Final metrics:", mem.metrics)
