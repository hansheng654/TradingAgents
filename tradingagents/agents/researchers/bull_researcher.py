from langchain_core.messages import AIMessage
import time
import json


def create_bull_researcher(llm, memory):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

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

        prompt = f"""You are a Bull Analyst advocating for investing in the stock. Your task is to build a strong, evidence-based case emphasizing growth potential, competitive advantages, and positive market indicators. Leverage the provided research and data to address concerns and counter bearish arguments effectively.

Key points to focus on:
- Growth Potential: Highlight the company's market opportunities, revenue projections, and scalability.
- Competitive Advantages: Emphasize factors like unique products, strong branding, or dominant market positioning.
- Positive Indicators: Use financial health, industry trends, and recent positive news as evidence.
- Bear Counterpoints: Critically analyze the bear argument with specific data and sound reasoning, addressing concerns thoroughly and showing why the bull perspective holds stronger merit.
- Engagement: Present your argument in a conversational style, engaging directly with the bear analyst's points and debating effectively rather than just listing data.

Resources available:
Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
Company fundamentals report: {fundamentals_report}
Conversation history of the debate: {history}
Last bear argument: {current_response}
Reflections from similar situations and lessons learned (preview): {past_preview}
Use this information to deliver a compelling bull argument, refute the bear's concerns, and engage in a dynamic debate that demonstrates the strengths of the bull position. You must also address reflections and learn from lessons and mistakes you made in the past.
"""

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state.get("count", 0) + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node


if __name__ == "__main__":
    """
    Lightweight self-tests for Bull Researcher.
    Verifies:
      1) Uses past_memories_preview when present (no fallback needed)
      2) Falls back to state["past_memories"] when preview is absent
      3) Gracefully handles no memories (prints friendly message)
      4) Increments debate count and appends to history without KeyErrors when reports missing
    Run:
        python tradingagents/agents/researchers/bull_researcher.py
    """
    from types import SimpleNamespace

    class DummyLLM:
        def __init__(self):
            self.calls = 0
            self.last_prompt = None
        def invoke(self, prompt):
            self.calls += 1
            self.last_prompt = prompt
            return SimpleNamespace(content="BULL: OK")

    def _mk_state(preview=None, memories=None, history="", count=0):
        state = {
            "investment_debate_state": {
                "history": history,
                "bull_history": "",
                "bear_history": "",
                "current_response": "",
                "count": count,
            },
            # Reports may be empty; node should not crash
            "market_report": "Market uptrend, revenue +12% YoY.",
            "sentiment_report": "Improving.",
            "news_report": "Beat EPS.",
            "fundamentals_report": "Margins stable.",
        }
        if preview is not None:
            state["past_memories_preview"] = preview
        if memories is not None:
            state["past_memories"] = memories
        return state

    print("\n[Test 1] Uses past_memories_preview when present (no fallback needed)")
    llm = DummyLLM()
    node = create_bull_researcher(llm, memory=None)
    preview = "- ID=1 | score=0.900 | Raise allocation to leaders; watch rates."
    out = node(_mk_state(preview=preview))
    assert "investment_debate_state" in out
    st = out["investment_debate_state"]
    assert st.get("count", 0) == 1, "Count should increment from 0 to 1"
    assert "Bull Analyst:" in st.get("history", ""), "History should include bull prefix"
    assert preview in llm.last_prompt, "Prompt should include provided preview"
    print("  ✓ Passed")

    print("\n[Test 2] Falls back to state['past_memories'] when preview is absent")
    llm2 = DummyLLM()
    node2 = create_bull_researcher(llm2, memory=None)
    memories = [
        {"recommendation": "Trim duration; prefer cash-flow names.", "similarity_score": None},
        {"recommendation": "Rotate to defensives if yields rise.", "similarity_score": 0.512},
    ]
    out2 = node2(_mk_state(preview=None, memories=memories, count=1))
    st2 = out2["investment_debate_state"]
    assert st2.get("count", 0) == 2, "Count should increment from 1 to 2"
    # Fallback preview should render ID lines with scores ('NA' allowed for None)
    assert "- ID=1" in llm2.last_prompt and "- ID=2" in llm2.last_prompt, "Fallback preview should list IDs"
    print("  ✓ Passed")

    print("\n[Test 3] Gracefully handles no memories (prints friendly message)")
    llm3 = DummyLLM()
    node3 = create_bull_researcher(llm3, memory=None)
    out3 = node3(_mk_state(preview=None, memories=None, count=2))
    st3 = out3["investment_debate_state"]
    assert st3.get("count", 0) == 3, "Count should increment from 2 to 3"
    assert "No past memories found" in llm3.last_prompt, "Prompt should include friendly no-memories message"
    print("  ✓ Passed")

    print("\n[Test 4] Increments debate count and appends to history without KeyErrors when reports missing")
    llm4 = DummyLLM()
    node4 = create_bull_researcher(llm4, memory=None)
    state4 = {
        "investment_debate_state": {
            "history": "Initial history.",
            "bull_history": "",
            "bear_history": "",
            "current_response": "",
            "count": 5,
        },
        # Missing reports intentionally
    }
    out4 = node4(state4)
    st4 = out4["investment_debate_state"]
    assert st4.get("count", 0) == 6, "Count should increment from 5 to 6"
    assert "Bull Analyst:" in st4.get("history", ""), "History should include bull prefix"
    print("  ✓ Passed")

    print("Bull Researcher self-tests passed.")
