from langchain_core.messages import AIMessage
import time
import json


def create_safe_debator(llm):
    def safe_node(state) -> dict:
        risk_debate_state = state.get("risk_debate_state", {})
        history = risk_debate_state.get("history", "")
        safe_history = risk_debate_state.get("safe_history", "")

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

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

        prompt = f"""As the Safe/Conservative Risk Analyst, your primary objective is to protect assets, minimize volatility, and ensure steady, reliable growth. You prioritize stability, security, and risk mitigation, carefully assessing potential losses, economic downturns, and market volatility. When evaluating the trader's decision or plan, critically examine high-risk elements, pointing out where the decision may expose the firm to undue risk and where more cautious alternatives could secure long-term gains. Here is the trader's decision:

{trader_decision}

Past reflections (preview):
{past_preview}

Your task is to actively counter the arguments of the Risky and Neutral Analysts, highlighting where their views may overlook potential threats or fail to prioritize sustainability. Respond directly to their points, drawing from the following data sources to build a convincing case for a low-risk approach adjustment to the trader's decision:
- Market: {market_research_report}
- Sentiment: {sentiment_report}
- News: {news_report}
- Fundamentals: {fundamentals_report}

Conversation history: {history}
Last from Risky: {current_risky_response}
Last from Neutral: {current_neutral_response}
(If there are no responses from the other viewpoints, do not hallucinate; just present your point.)

Engage by questioning their optimism and emphasizing the potential downsides they may have overlooked. Address each of their counterpoints to showcase why a conservative stance is ultimately the safest path for the firm's assets. Focus on debating and critiquing their arguments to demonstrate the strength of a low-risk strategy over their approaches. Output conversationally without special formatting."""

        response = llm.invoke(prompt)

        argument = f"Safe Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risk_debate_state.get("risky_history", ""),
            "safe_history": safe_history + "\n" + argument,
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Safe",
            "current_risky_response": risk_debate_state.get(
                "current_risky_response", ""
            ),
            "current_safe_response": argument,
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state.get("count", 0) + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return safe_node


if __name__ == "__main__":
    """
    Self-tests for Safe/Conservative Debator.
    Verifies preview usage, fallback, no-memory path, and robust count/history.
    Run: python tradingagents/agents/risk_mgmt/conservative_debator.py
    """
    from types import SimpleNamespace

    class DummyLLM:
        def __init__(self):
            self.calls = 0
            self.last_prompt = None
        def invoke(self, prompt):
            self.calls += 1
            self.last_prompt = prompt
            return SimpleNamespace(content="SAFE: ok")

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
            "trader_investment_plan": "Reduce risk; tighten stops.",
            "market_report": "Choppy.",
            "sentiment_report": "Cautious.",
            "news_report": "Mixed data.",
            "fundamentals_report": "Decent balance sheet.",
        }
        if preview is not None:
            st["past_memories_preview"] = preview
        if memories is not None:
            st["past_memories"] = memories
        return st

    print("\n[Test 1] Uses preview when present")
    llm1 = DummyLLM()
    node1 = create_safe_debator(llm1)
    prev1 = "- ID=1 | score=0.72 | Prior drawdowns avoided by cutting risk early during volatility spikes."
    out1 = node1(_mk_state(preview=prev1))
    assert "SAFE:" in out1["risk_debate_state"]["current_safe_response"], "Should include model content"
    assert prev1 in llm1.last_prompt, "Preview must be in prompt"
    assert out1["risk_debate_state"].get("count", 0) == 1
    print("  ✓ Passed")

    print("\n[Test 2] Fallback to past_memories list when no preview")
    llm2 = DummyLLM()
    node2 = create_safe_debator(llm2)
    mems = [
        {"recommendation": "Cut position size; raise stops.", "similarity_score": 0.41},
        {"recommendation": "Favor defensives; avoid leverage.", "similarity_score": None},
    ]
    out2 = node2(_mk_state(preview=None, memories=mems, count=3))
    assert "- ID=1" in llm2.last_prompt and "- ID=2" in llm2.last_prompt
    assert out2["risk_debate_state"].get("count", 0) == 4
    print("  ✓ Passed")

    print("\n[Test 3] No memories -> friendly message")
    llm3 = DummyLLM()
    node3 = create_safe_debator(llm3)
    out3 = node3(_mk_state(preview=None, memories=None, count=2))
    assert "No past memories found" in llm3.last_prompt
    assert out3["risk_debate_state"].get("count", 0) == 3
    print("  ✓ Passed\n")
