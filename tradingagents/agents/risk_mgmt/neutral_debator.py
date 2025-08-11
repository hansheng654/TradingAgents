import time
import json


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state.get("risk_debate_state", {})
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_safe_response = risk_debate_state.get("current_safe_response", "")

        market_research_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")

        trader_decision = state.get("trader_investment_plan", "")

        past_preview = state.get("past_memories_preview") or state.get("past_memories_str")
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

        prompt = f"""As the Neutral Risk Analyst, your role is to provide a balanced perspective, weighing both the potential benefits and risks of the trader's decision or plan. You prioritize a well-rounded approach, evaluating the upsides and downsides while factoring in broader market trends, potential economic shifts, and diversification strategies. Here is the trader's decision:

{trader_decision}

Past reflections (preview):
{past_preview}

Use insights from the following data sources to support a moderate, sustainable strategy to adjust the trader's decision:
- Market: {market_research_report}
- Sentiment: {sentiment_report}
- News: {news_report}
- Fundamentals: {fundamentals_report}

Conversation history: {history}
Last from Risky: {current_risky_response}
Last from Safe: {current_safe_response}
(If there are no responses from the other viewpoints, do not hallucinate; just present your point.)

Engage actively by analyzing both sides critically, addressing weaknesses in the risky and conservative arguments to advocate for a more balanced approach. Challenge each of their points to illustrate why a moderate risk strategy might offer the best of both worlds, providing growth potential while safeguarding against extreme volatility. Output conversationally without special formatting."""

        response = llm.invoke(prompt)

        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risk_debate_state.get("risky_history", ""),
            "safe_history": risk_debate_state.get("safe_history", ""),
            "neutral_history": neutral_history + "\n" + argument,
            "latest_speaker": "Neutral",
            "current_risky_response": risk_debate_state.get(
                "current_risky_response", ""
            ),
            "current_safe_response": risk_debate_state.get("current_safe_response", ""),
            "current_neutral_response": argument,
            "count": risk_debate_state.get("count", 0) + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node


if __name__ == "__main__":
    """
    Self-tests for Neutral Debator.
    Verifies preview usage, fallback, no-memory path, and robust count/history.
    Run: python tradingagents/agents/risk_mgmt/neutral_debator.py
    """
    from types import SimpleNamespace

    class DummyLLM:
        def __init__(self):
            self.calls = 0
            self.last_prompt = None
        def invoke(self, prompt):
            self.calls += 1
            self.last_prompt = prompt
            return SimpleNamespace(content="NEUTRAL: ok")

    def _mk_state(preview=None, memories=None, count=0):
        st = {
            "risk_debate_state": {
                "history": "",
                "risky_history": "",
                "safe_history": "",
                "neutral_history": "",
                "current_risky_response": "",
                "current_safe_response": "",
                "current_neutral_response": "",
                "count": count,
            },
            "trader_investment_plan": "Blend momentum with risk controls.",
            "market_report": "Range-bound.",
            "sentiment_report": "Mixed.",
            "news_report": "In-line.",
            "fundamentals_report": "Stable.",
        }
        if preview is not None:
            st["past_memories_preview"] = preview
        if memories is not None:
            st["past_memories"] = memories
            
        return st

    print("\n[Test 1] Uses preview when present")
    llm1 = DummyLLM()
    node1 = create_neutral_debator(llm1)
    prev1 = "- ID=1 | score=0.60 | Blended strategy worked when volatility was moderate."
    out1 = node1(_mk_state(preview=prev1))
    assert "NEUTRAL:" in out1["risk_debate_state"]["current_neutral_response"]
    assert prev1 in llm1.last_prompt
    assert out1["risk_debate_state"].get("count", 0) == 1
    print("  ✓ Passed")

    print("\n[Test 2] Fallback to past_memories list when no preview")
    llm2 = DummyLLM()
    node2 = create_neutral_debator(llm2)
    mems = [
        {"recommendation": "Hedge tail risk; keep dry powder.", "similarity_score": 0.45},
        {"recommendation": "Diversify across factors.", "similarity_score": None},
    ]
    out2 = node2(_mk_state(preview=None, memories=mems, count=5))
    assert "- ID=1" in llm2.last_prompt and "- ID=2" in llm2.last_prompt
    assert out2["risk_debate_state"].get("count", 0) == 6
    print("  ✓ Passed")

    print("\n[Test 3] No memories -> friendly message")
    llm3 = DummyLLM()
    node3 = create_neutral_debator(llm3)
    out3 = node3(_mk_state(preview=None, memories=None, count=2))
    assert "No past memories found" in llm3.last_prompt
    assert out3["risk_debate_state"].get("count", 0) == 3
    print("  ✓ Passed\n")
