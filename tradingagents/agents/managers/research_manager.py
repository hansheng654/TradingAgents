import time
import json


def create_research_manager(llm, memory):
    def research_manager_node(state) -> dict:
        investment_debate_state = state.get("investment_debate_state", {})
        history = investment_debate_state.get("history", "")
        market_research_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")

        # Use Memory Broker outputs instead of per-node retrieval
        past_preview = (
            state.get("past_memories_preview")
            or state.get("past_memories_str")
        )
        if not past_preview:
            # Deterministic fallback if preview not present but structured memories exist
            items = []
            for i, rec in enumerate(state.get("past_memories") or [], 1):
                rec_txt = (rec.get("recommendation") or "").strip()
                score = rec.get("similarity_score")
                if isinstance(score, (int, float)):
                    score_str = f"{score:.3f}"
                else:
                    try:
                        score_str = f"{float(score):.3f}"
                    except (TypeError, ValueError):
                        score_str = "NA"
                items.append(f"- ID={i} | score={score_str} | {rec_txt}")
            past_preview = "\n".join(items) if items else "No past memories found."

        prompt = f"""As the portfolio manager and debate facilitator, your role is to critically evaluate this round of debate and make a definitive decision: align with the bear analyst, the bull analyst, or choose Hold only if it is strongly justified based on the arguments presented.

Summarize the key points from both sides concisely, focusing on the most compelling evidence or reasoning. Your recommendation—Buy, Sell, or Hold—must be clear and actionable. Avoid defaulting to Hold simply because both sides have valid points; commit to a stance grounded in the debate's strongest arguments.

Additionally, develop a detailed investment plan for the trader. This should include:

Your Recommendation: A decisive stance supported by the most convincing arguments.
Rationale: An explanation of why these arguments lead to your conclusion.
Strategic Actions: Concrete steps for implementing the recommendation.

Use lessons from prior, similar situations to avoid repeating mistakes.

Past reflections (preview):
{past_preview}

Reference analyst inputs:
- Market: {market_research_report}
- Sentiment: {sentiment_report}
- News: {news_report}
- Fundamentals: {fundamentals_report}

Debate History:
{history}
"""
        response = llm.invoke(prompt)

        new_investment_debate_state = {
            "judge_decision": response.content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": response.content,
            "count": investment_debate_state.get("count", 0),
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": response.content,
        }

    return research_manager_node


if __name__ == "__main__":
    """
    Lightweight self-tests for Research Manager.
    Verifies:
      1) Uses past_memories_preview when present
      2) Falls back to state["past_memories"] when preview is absent
      3) Gracefully handles no memories
      4) Preserves/increments debate state fields robustly
    Run:
        python tradingagents/agents/managers/research_manager.py
    """
    from types import SimpleNamespace

    class DummyLLM:
        def __init__(self, tag):
            self.tag = tag
            self.calls = 0
            self.last_prompt = None
        def invoke(self, prompt):
            self.calls += 1
            self.last_prompt = prompt
            return SimpleNamespace(content=f"{self.tag}: DECISION=BUY")

    def _mk_state(preview=None, memories=None, history="", count=0):
        st = {
            "investment_debate_state": {
                "history": history,
                "bear_history": "",
                "bull_history": "",
                "current_response": "",
                "count": count,
            },
            "market_report": "Market steady.",
            "sentiment_report": "Neutral.",
            "news_report": "Inline.",
            "fundamentals_report": "Solid margins.",
        }
        if preview is not None:
            st["past_memories_preview"] = preview
        if memories is not None:
            st["past_memories"] = memories
        return st

    print("\n[Test 1] Uses preview when present")
    llm1 = DummyLLM("RM1")
    node1 = create_research_manager(llm1, memory=None)
    prev1 = "- ID=1 | score=0.88 | Prior miss due to ignoring liquidity risk; include stop-loss."
    out1 = node1(_mk_state(preview=prev1, count=0))
    assert out1["investment_plan"].startswith("RM1"), "LLM output should flow through"
    assert prev1 in llm1.last_prompt, "Prompt should include preview"
    print("  ✓ Passed")

    print("\n[Test 2] Fallback to past_memories when no preview")
    llm2 = DummyLLM("RM2")
    node2 = create_research_manager(llm2, memory=None)
    mems = [
        {"recommendation": "Avoid leverage; maintain cash buffer.", "similarity_score": 0.42},
        {"recommendation": "Rotate to defensives on rising yields.", "similarity_score": None},
    ]
    out2 = node2(_mk_state(preview=None, memories=mems, count=3))
    assert "- ID=1" in llm2.last_prompt and "- ID=2" in llm2.last_prompt, "Fallback should render ID list"
    assert out2["investment_debate_state"].get("count", 0) == 3
    print("  ✓ Passed")

    print("\n[Test 3] No memories -> friendly message")
    llm3 = DummyLLM("RM3")
    node3 = create_research_manager(llm3, memory=None)
    out3 = node3(_mk_state(preview=None, memories=None, count=5))
    assert "No past memories found" in llm3.last_prompt
    print("  ✓ Passed")

    print("\nResearch Manager self-tests passed.\n")
