from langchain_core.messages import AIMessage
import time
import json


def create_bear_researcher(llm, memory):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = investment_debate_state.get("current_response", "")
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

        prompt = f"""You are a Bear Analyst making the case against investing in the stock. Your goal is to present a well-reasoned argument emphasizing risks, challenges, and negative indicators. Leverage the provided research and data to highlight potential downsides and counter bullish arguments effectively.

Key points to focus on:

- Risks and Challenges: Highlight factors like market saturation, financial instability, or macroeconomic threats that could hinder the stock's performance.
- Competitive Weaknesses: Emphasize vulnerabilities such as weaker market positioning, declining innovation, or threats from competitors.
- Negative Indicators: Use evidence from financial data, market trends, or recent adverse news to support your position.
- Bull Counterpoints: Critically analyze the bull argument with specific data and sound reasoning, exposing weaknesses or over-optimistic assumptions.
- Engagement: Present your argument in a conversational style, directly engaging with the bull analyst's points and debating effectively rather than simply listing facts.

Resources available:

Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
Company fundamentals report: {fundamentals_report}
Conversation history of the debate: {history}
Last bull argument: {current_response}
Reflections from similar situations and lessons learned (preview): {past_preview}
Use this information to deliver a compelling bear argument, refute the bull's claims, and engage in a dynamic debate that demonstrates the risks and weaknesses of investing in the stock. You must also address reflections and learn from lessons and mistakes you made in the past.
"""

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state.get("count", 0) + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node


if __name__ == "__main__":
    """
    Lightweight self-tests for Bear Researcher.
    Verifies:
      1) Uses past_memories_preview when present
      2) Falls back to state["past_memories"] when preview is absent
      3) Gracefully handles no memories
      4) Increments count and appends to history; robust when reports missing
    Run:
        python tradingagents/agents/researchers/bear_researcher.py
    """
    from types import SimpleNamespace

    class DummyLLM:
        def __init__(self):
            self.calls = 0
            self.last_prompt = None
        def invoke(self, prompt):
            self.calls += 1
            self.last_prompt = prompt
            return SimpleNamespace(content="BEAR: OK")

    def _mk_state(preview=None, memories=None, history="", count=0):
        state = {
            "investment_debate_state": {
                "history": history,
                "bear_history": "",
                "bull_history": "",
                "current_response": "",
                "count": count,
            },
            # Reports may be empty; node should not crash
            "market_report": "Market risk rising, revenue slowing.",
            "sentiment_report": "Mixed.",
            "news_report": "Regulatory headwinds.",
            "fundamentals_report": "Cash burn elevated.",
        }
        if preview is not None:
            state["past_memories_preview"] = preview
        if memories is not None:
            state["past_memories"] = memories
        return state

    print("\n[Test 1] Uses past_memories_preview when present")
    llm = DummyLLM()
    node = create_bear_researcher(llm, memory=None)
    preview = "- ID=1 | score=0.420 | Cut exposure; rising yields pressure valuations."
    out = node(_mk_state(preview=preview))
    st = out["investment_debate_state"]
    assert st.get("count", 0) == 1, "Count should increment from 0 to 1"
    assert "Bear Analyst:" in st.get("history", ""), "History should include bear prefix"
    assert preview in llm.last_prompt, "Prompt should include provided preview"
    print("  ✓ Passed")

    print("\n[Test 2] Fallback preview from memories when preview is absent")
    llm2 = DummyLLM()
    node2 = create_bear_researcher(llm2, memory=None)
    memories = [
        {"recommendation": "Avoid high beta; hedge with puts.", "similarity_score": 0.333},
        {"recommendation": "Shift to quality balance sheets.", "similarity_score": None},
    ]
    out2 = node2(_mk_state(preview=None, memories=memories, count=1))
    st2 = out2["investment_debate_state"]
    assert st2.get("count", 0) == 2, "Count should increment from 1 to 2"
    assert "- ID=1" in llm2.last_prompt and "- ID=2" in llm2.last_prompt, "Fallback preview should list IDs"
    print("  ✓ Passed")

    print("\n[Test 3] No preview and no memories -> friendly message in prompt")
    llm3 = DummyLLM()
    node3 = create_bear_researcher(llm3, memory=None)
    out3 = node3(_mk_state(preview=None, memories=None, count=2))
    st3 = out3["investment_debate_state"]
    assert st3.get("count", 0) == 3, "Count should increment from 2 to 3"
    assert "No past memories found" in llm3.last_prompt, "Prompt should include friendly no-memories message"
    print("  ✓ Passed")

    print("Bear Researcher self-tests passed.")
