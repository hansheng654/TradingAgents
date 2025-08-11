import functools
import time
import json


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state.get("company_of_interest", "Unknown Company")
        investment_plan = state.get("investment_plan", "")
        market_research_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")

        # Use Memory Broker outputs
        past_preview = (
            state.get("past_memories_preview")
            or state.get("past_memories_str")
        )
        if not past_preview:
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

        context = {
            "role": "user",
            "content": f"Based on a comprehensive analysis by a team of analysts, here is an investment plan tailored for {company_name}. This plan incorporates insights from current technical market trends, macroeconomic indicators, and social media sentiment. Use this plan as a foundation for evaluating your next trading decision.\n\nProposed Investment Plan: {investment_plan}\n\nLeverage these insights to make an informed and strategic decision.",
        }

        messages = [
            {
                "role": "system",
                "content": f"""You are a trading agent analyzing market data to make investment decisions. Provide a specific recommendation to buy, sell, or hold. End with a firm decision and always conclude your response with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**'. Incorporate lessons from prior similar situations to avoid repeat mistakes.

Past reflections (preview):
{past_preview}

Reference analyst inputs:
- Market: {market_research_report}
- Sentiment: {sentiment_report}
- News: {news_report}
- Fundamentals: {fundamentals_report}
""",
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")


if __name__ == "__main__":
    """
    Lightweight self-tests for Trader.
    Verifies:
      1) Uses past_memories_preview when present
      2) Falls back to state["past_memories"] when preview is absent
      3) Gracefully handles no memories
      4) Produces a 'messages' list and 'trader_investment_plan'
    Run:
        python tradingagents/agents/trader/trader.py
    """
    from types import SimpleNamespace

    class DummyLLM:
        def __init__(self):
            self.calls = 0
            self.last_messages = None
        def invoke(self, messages):
            self.calls += 1
            self.last_messages = messages
            return SimpleNamespace(content="TRADER: FINAL TRANSACTION PROPOSAL: **BUY**")

    def _mk_state(preview=None, memories=None):
        st = {
            "company_of_interest": "ACME",
            "investment_plan": "Buy on pullbacks; scale in.",
            "market_report": "Trend up.",
            "sentiment_report": "Positive.",
            "news_report": "Beats.",
            "fundamentals_report": "Cash flow strong.",
        }
        if preview is not None:
            st["past_memories_preview"] = preview
        if memories is not None:
            st["past_memories"] = memories
        return st

    print("\n[Test 1] Uses preview when present")
    llm1 = DummyLLM()
    node1 = create_trader(llm1, memory=None)
    prev1 = "- ID=1 | score=0.88 | Prior stop-loss too loose; tighten."
    out1 = node1(_mk_state(preview=prev1))
    assert isinstance(out1.get("messages"), list) and out1["messages"], "Should return messages list"
    assert out1.get("trader_investment_plan", "").startswith("TRADER:"), "Should forward LLM content"
    # inspect last prompt
    system_msg = llm1.last_messages[0]["content"]
    assert prev1 in system_msg
    print("  ✓ Passed")

    print("\n[Test 2] Fallback to past_memories when no preview")
    llm2 = DummyLLM()
    node2 = create_trader(llm2, memory=None)
    mems = [
        {"recommendation": "Hedge with puts.", "similarity_score": 0.333},
        {"recommendation": "Shift to quality.", "similarity_score": None},
    ]
    out2 = node2(_mk_state(preview=None, memories=mems))
    system_msg2 = llm2.last_messages[0]["content"]
    assert "- ID=1" in system_msg2 and "- ID=2" in system_msg2
    print("  ✓ Passed")

    print("\n[Test 3] No memories -> friendly message")
    llm3 = DummyLLM()
    node3 = create_trader(llm3, memory=None)
    out3 = node3(_mk_state(preview=None, memories=None))
    system_msg3 = llm3.last_messages[0]["content"]
    assert "No past memories found" in system_msg3
    print("  ✓ Passed")

    print("\nTrader self-tests passed.\n")
