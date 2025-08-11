import time
import json


def create_risky_debator(llm):
    def risky_node(state) -> dict:
        risk_debate_state = state.get("risk_debate_state", {})
        history = risk_debate_state.get("history", "")
        risky_history = risk_debate_state.get("risky_history", "")

        current_safe_response = risk_debate_state.get("current_safe_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")

        trader_decision = state.get("trader_investment_plan", "")

        # Memory Broker preview (or deterministic fallback)
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

        prompt = f"""As the Risky Risk Analyst, your role is to actively champion high-reward, high-risk opportunities, emphasizing bold strategies and competitive advantages. When evaluating the trader's decision or plan, focus intently on the potential upside, growth potential, and innovative benefits—even when these come with elevated risk. Use the provided market data and sentiment analysis to strengthen your arguments and challenge the opposing views. Specifically, respond directly to each point made by the conservative and neutral analysts, countering with data-driven rebuttals and persuasive reasoning. Highlight where their caution might miss critical opportunities or where their assumptions may be overly conservative. Here is the trader's decision:

{trader_decision}

Past reflections (preview):
{past_preview}

Incorporate insights from the following sources into your arguments:
- Market: {market_research_report}
- Sentiment: {sentiment_report}
- News: {news_report}
- Fundamentals: {fundamentals_report}

Conversation history: {history}
Last from Safe: {current_safe_response}
Last from Neutral: {current_neutral_response}
(If there are no responses from the other viewpoints, do not hallucinate; just present your point.)

Engage actively by addressing any specific concerns raised, refuting the weaknesses in their logic, and asserting the benefits of risk-taking to outpace market norms. Maintain a focus on debating and persuading, not just presenting data. Challenge each counterpoint to underscore why a high-risk approach is optimal. Output conversationally without special formatting."""

        response = llm.invoke(prompt)

        argument = f"Risky Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risky_history + "\n" + argument,
            "safe_history": risk_debate_state.get("safe_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Risky",
            "current_risky_response": argument,
            "current_safe_response": risk_debate_state.get("current_safe_response", ""),
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state.get("count", 0) + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return risky_node


if __name__ == "__main__":
    """
    Self-tests for Risky Debator.
    Verifies preview usage, fallback, no-memory path, and robust count/history.
    Run: python tradingagents/agents/risk_mgmt/aggresive_debator.py
    """
    from types import SimpleNamespace

    class DummyLLM:
        def __init__(self):
            self.calls = 0
            self.last_prompt = None
        def invoke(self, prompt):
            self.calls += 1
            self.last_prompt = prompt
            return SimpleNamespace(content="RISKY: ok")

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
            "trader_investment_plan": "Enter aggressively on momentum.",
            "market_report": "Uptrend.",
            "sentiment_report": "Bullish.",
            "news_report": "Strong guide.",
            "fundamentals_report": "Growing margins.",
        }
        if preview is not None:
            st["past_memories_preview"] = preview
        if memories is not None:
            st["past_memories"] = memories
        return st

    print("\n[Test 1] Uses preview when present")
    llm1 = DummyLLM()
    node1 = create_risky_debator(llm1)
    prev1 = "- ID=1 | score=0.80 | Prior success with momentum entries when liquidity strong."
    out1 = node1(_mk_state(preview=prev1))
    assert "RISKY:" in out1["risk_debate_state"]["current_risky_response"], "Should include model content"
    assert prev1 in llm1.last_prompt, "Preview must be in prompt"
    assert out1["risk_debate_state"].get("count", 0) == 1
    print("  ✓ Passed")

    print("\n[Test 2] Fallback to past_memories list when no preview")
    llm2 = DummyLLM()
    node2 = create_risky_debator(llm2)
    mems = [
        {"recommendation": "Press winners; keep tight trailing stops.", "similarity_score": 0.51},
        {"recommendation": "Scale in on breakouts.", "similarity_score": None},
    ]
    out2 = node2(_mk_state(preview=None, memories=mems, count=1))
    assert "- ID=1" in llm2.last_prompt and "- ID=2" in llm2.last_prompt
    assert out2["risk_debate_state"].get("count", 0) == 2
    print("  ✓ Passed")

    print("\n[Test 3] No memories -> friendly message")
    llm3 = DummyLLM()
    node3 = create_risky_debator(llm3)
    out3 = node3(_mk_state(preview=None, memories=None, count=2))
    assert "No past memories found" in llm3.last_prompt
    assert out3["risk_debate_state"].get("count", 0) == 3
    print("  ✓ Passed\n")
