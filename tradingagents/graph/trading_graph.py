# TradingAgents/graph/trading_graph.py

import os
from pathlib import Path
import json
from datetime import date
from typing import Dict, Any, Tuple, List, Optional

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.dataflows.interface import set_config

# Support running as a package module or as a standalone script
try:
    from .conditional_logic import ConditionalLogic  # package-relative
    from .setup import GraphSetup
    from .propagation import Propagator
    from .reflection import Reflector
    from .signal_processing import SignalProcessor
except Exception:
    try:
        # absolute imports when executed directly
        from tradingagents.graph.conditional_logic import ConditionalLogic
        from tradingagents.graph.setup import GraphSetup
        from tradingagents.graph.propagation import Propagator
        from tradingagents.graph.reflection import Reflector
        from tradingagents.graph.signal_processing import SignalProcessor
    except Exception as e:
        raise ImportError(
            "Could not import graph components. Try running as a module: \n"
            "    python -m tradingagents.graph.trading_graph\n"
            "or ensure project root is on PYTHONPATH."
        ) from e


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        # Initialize LLMs
        if self.config["llm_provider"].lower() == "openai" or self.config["llm_provider"] == "ollama" or self.config["llm_provider"] == "openrouter":
            self.deep_thinking_llm = ChatOpenAI(model=self.config["deep_think_llm"], base_url=self.config["backend_url"])
            self.quick_thinking_llm = ChatOpenAI(model=self.config["quick_think_llm"], base_url=self.config["backend_url"])
        elif self.config["llm_provider"].lower() == "anthropic":
            self.deep_thinking_llm = ChatAnthropic(model=self.config["deep_think_llm"], base_url=self.config["backend_url"])
            self.quick_thinking_llm = ChatAnthropic(model=self.config["quick_think_llm"], base_url=self.config["backend_url"])
        elif self.config["llm_provider"].lower() == "google":
            self.deep_thinking_llm = ChatGoogleGenerativeAI(model=self.config["deep_think_llm"])
            self.quick_thinking_llm = ChatGoogleGenerativeAI(model=self.config["quick_think_llm"])
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config['llm_provider']}")
        
        self.toolkit = Toolkit(config=self.config)

        # Initialize a single shared memory for retrieval & reflections
        self.shared_memory = FinancialSituationMemory("situations", self.config)

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic()
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.toolkit,
            self.tool_nodes,
            self.shared_memory,
            self.conditional_logic,
        )

        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph
        self.graph = self.graph_setup.setup_graph(selected_analysts)

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create tool nodes for different data sources."""
        return {
            "market": ToolNode(
                [
                    # online tools (batch first to encourage single-call usage)
                    self.toolkit.get_stockstats_multi_indicators_report_online,
                    self.toolkit.get_YFin_data_online,
                    self.toolkit.get_stockstats_indicators_report_online,
                    # offline tools
                    self.toolkit.get_YFin_data,
                    self.toolkit.get_stockstats_indicators_report,
                ]
            ),
            "social": ToolNode(
                [
                    # online tools
                    self.toolkit.get_stock_news_openai,
                    # offline tools
                    self.toolkit.get_reddit_stock_info,
                ]
            ),
            "news": ToolNode(
                [
                    # online tools
                    self.toolkit.get_global_news_openai,
                    self.toolkit.get_google_news,
                    # offline tools
                    self.toolkit.get_finnhub_news,
                    self.toolkit.get_reddit_news,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # online tools
                    self.toolkit.get_fundamentals_openai,
                    # offline tools
                    self.toolkit.get_finnhub_company_insider_sentiment,
                    self.toolkit.get_finnhub_company_insider_transactions,
                    self.toolkit.get_simfin_balance_sheet,
                    self.toolkit.get_simfin_cashflow,
                    self.toolkit.get_simfin_income_stmt,
                ]
            ),
        }

    def propagate(self, company_name, trade_date):
        """Run the trading agents graph for a company on a specific date."""

        self.ticker = company_name

        # Initialize state
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date
        )
        args = self.propagator.get_graph_args()

        if self.debug:
            # Debug mode with tracing
            trace = []
            for chunk in self.graph.stream(init_agent_state, **args):
                if len(chunk["messages"]) == 0:
                    pass
                else:
                    chunk["messages"][-1].pretty_print()
                    trace.append(chunk)

            final_state = trace[-1]
        else:
            # Standard mode without tracing
            final_state = self.graph.invoke(init_agent_state, **args)

        # Store current state for reflection
        self.curr_state = final_state

        # Log state
        self._log_state(trade_date, final_state)

        # Return decision and processed signal
        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        inv = final_state.get("investment_debate_state", {})
        risk = final_state.get("risk_debate_state", {})

        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state.get("company_of_interest", self.ticker or ""),
            "trade_date": final_state.get("trade_date", str(trade_date)),
            "market_report": final_state.get("market_report", ""),
            "sentiment_report": final_state.get("sentiment_report", ""),
            "news_report": final_state.get("news_report", ""),
            "fundamentals_report": final_state.get("fundamentals_report", ""),
            "investment_debate_state": {
                "bull_history": inv.get("bull_history", ""),
                "bear_history": inv.get("bear_history", ""),
                "history": inv.get("history", ""),
                "current_response": inv.get("current_response", ""),
                "judge_decision": inv.get("judge_decision", ""),
            },
            "trader_investment_decision": final_state.get("trader_investment_plan", ""),
            "risk_debate_state": {
                "risky_history": risk.get("risky_history", ""),
                "safe_history": risk.get("safe_history", ""),
                "neutral_history": risk.get("neutral_history", ""),
                "history": risk.get("history", ""),
                "judge_decision": risk.get("judge_decision", ""),
            },
            "investment_plan": final_state.get("investment_plan", ""),
            "final_trade_decision": final_state.get("final_trade_decision", ""),
        }

        # Save to file
        directory = Path(f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/")
        directory.mkdir(parents=True, exist_ok=True)

        with open(
            f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/full_states_log_{trade_date}.json",
            "w",
        ) as f:
            json.dump(self.log_states_dict, f, indent=4)

    def reflect_and_remember(self, returns_losses):
        """Reflect on decisions and update memory based on returns.
        Uses the shared memory store for all roles to keep retrieval unified.
        """
        self.reflector.reflect_bull_researcher(
            self.curr_state, returns_losses, self.shared_memory
        )
        self.reflector.reflect_bear_researcher(
            self.curr_state, returns_losses, self.shared_memory
        )
        self.reflector.reflect_trader(
            self.curr_state, returns_losses, self.shared_memory
        )
        self.reflector.reflect_invest_judge(
            self.curr_state, returns_losses, self.shared_memory
        )
        self.reflector.reflect_risk_manager(
            self.curr_state, returns_losses, self.shared_memory
        )

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)


# === Self-test harness for TradingAgentsGraph wiring ===
if __name__ == "__main__":
    """
    Lightweight end-to-end self-test for TradingAgentsGraph wiring with the Memory Broker.
    - Avoids any network/LLM/tool calls by monkeypatching dependencies to dummies.
    - Verifies that analysts run, Memory Broker populates preview, and the flow reaches a final decision.

    Run:
        python tradingagents/graph/trading_graph.py
    """
    from types import SimpleNamespace
    from datetime import datetime, timezone
    import tradingagents.graph.setup as setup_mod

    print("\n== TradingAgentsGraph wiring self-test ==\n")

    # ---- Dummy LLM that matches Chat* constructors and captures prompts ----
    class DummyLLM:
        def __init__(self, model=None, base_url=None, **kwargs):
            self.model = model
            self.base_url = base_url
            self.calls = 0
            self.last_input = None
        def invoke(self, x):
            self.calls += 1
            self.last_input = x
            return SimpleNamespace(content="DUMMY: ok")

    # Monkeypatch the LLM classes used in this module (so __init__ succeeds without network)
    ChatOpenAI = DummyLLM          # type: ignore
    ChatAnthropic = DummyLLM       # type: ignore
    ChatGoogleGenerativeAI = DummyLLM  # type: ignore

    # ---- Dummy Shared Memory so Memory Broker does not embed/call APIs ----
    class DummySharedMemory:
        def __init__(self, *args, **kwargs):
            self.calls = 0
        def get_memories(self, current_situation, n_matches=2):
            self.calls += 1
            # Return two simple memories
            return [
                {
                    "matched_situation": "Tech rotation with rising yields",
                    "recommendation": "Reduce duration; prefer cash-flow positive leaders.",
                    "similarity_score": 0.71,
                },
                {
                    "matched_situation": "Earnings season defensives bid",
                    "recommendation": "Increase staples/utilities; hedge beta.",
                    "similarity_score": 0.53,
                },
            ][:n_matches]

    # Patch the FinancialSituationMemory symbol in this module to our dummy
    FinancialSituationMemory = DummySharedMemory  # type: ignore

    # ---- Patch tool nodes to pure no-ops (no external calls) ----
    def _noop_tool(state):
        return {}
    def _no_tool_nodes(self):
        return {
            "market": _noop_tool,
            "social": _noop_tool,
            "news": _noop_tool,
            "fundamentals": _noop_tool,
        }
    TradingAgentsGraph._create_tool_nodes = _no_tool_nodes  # type: ignore

    # ---- Control flow logic: always advance deterministically ----
    class DummyConditionalLogic:
        # Analysts: always choose 'clear' branch (advance)
        def __getattr__(self, name):
            if name.startswith("should_continue_"):
                return lambda state: "clear"
            raise AttributeError(name)
        # Debate: jump to Research Manager immediately
        def should_continue_debate(self, state):
            return "Research Manager"
        # Risk: jump to Risk Judge for final decision
        def should_continue_risk_analysis(self, state):
            return "Risk Judge"

    # Patch ConditionalLogic used by this module
    ConditionalLogic = DummyConditionalLogic  # type: ignore

    # ---- Monkeypatch node factories inside setup module under test ----
    def _mk_long_report(name, key):
        def node(state):
            filler = (
                f" {name} context: RSI 65, MACD bullish, CPI 3.2% YoY, EPS +10%, "
                f"tickers AAPL NVDA MSFT; liquidity ample; breadth improving."
            ) * 60  # ensure Memory Broker triggers with production thresholds
            return {key: f"{name} report OK at {datetime.now(timezone.utc).isoformat()}." + filler}
        return node

    def _mk_clear():
        def node(state):
            return {}
        return node

    def _mk_set(key, value):
        def node(state):
            return {key: value}
        return node

    # Analysts
    setup_mod.create_market_analyst = lambda llm, tk: _mk_long_report("Market", "market_report")  # type: ignore
    setup_mod.create_social_media_analyst = lambda llm, tk: _mk_long_report("Social", "sentiment_report")  # type: ignore
    setup_mod.create_news_analyst = lambda llm, tk: _mk_long_report("News", "news_report")  # type: ignore
    setup_mod.create_fundamentals_analyst = lambda llm, tk: _mk_long_report("Fundamentals", "fundamentals_report")  # type: ignore
    setup_mod.create_msg_delete = lambda: _mk_clear()  # type: ignore

    # Researchers & manager (minimal outputs)
    setup_mod.create_bull_researcher = lambda llm, mem: _mk_set("investment_debate_state", {"bull_history": "BULL ok"})  # type: ignore
    setup_mod.create_bear_researcher = lambda llm, mem: _mk_set("investment_debate_state", {"bear_history": "BEAR ok"})  # type: ignore
    setup_mod.create_research_manager = lambda llm, mem: _mk_set("investment_plan", "RM: BUY with plan")  # type: ignore

    # Trader
    setup_mod.create_trader = lambda llm, mem: _mk_set("trader_investment_plan", "TRADER: BUY on signal")  # type: ignore

    # Risk debators & judge; judge emits final decision
    setup_mod.create_risky_debator = lambda llm: _mk_set("risk_debate_state", {"risky_history": "RISKY ok"})  # type: ignore
    setup_mod.create_neutral_debator = lambda llm: _mk_set("risk_debate_state", {"neutral_history": "NEUTRAL ok"})  # type: ignore
    setup_mod.create_safe_debator = lambda llm: _mk_set("risk_debate_state", {"safe_history": "SAFE ok"})  # type: ignore
    setup_mod.create_risk_manager = lambda llm, mem: (  # type: ignore
        _mk_set("final_trade_decision", "FINAL: BUY")
    )

    # ---- Build config (no real endpoints used) ----
    cfg = DEFAULT_CONFIG.copy()
    cfg["llm_provider"] = "openai"
    cfg["backend_url"] = "http://localhost"
    cfg["deep_think_llm"] = "dummy-deep"
    cfg["quick_think_llm"] = "dummy-quick"
    cfg["online_tools"] = False

    # ---- Instantiate and run once ----
    graph = TradingAgentsGraph(selected_analysts=["market", "news"], debug=False, config=cfg)
    final_state, decision = graph.propagate("TEST", "2025-01-01")

    # ---- Diagnostics ----
    produced_keys = [k for k in final_state.keys() if k in (
        "market_report","news_report","past_memories","past_memories_preview","past_memories_str",
        "investment_plan","trader_investment_plan","risk_debate_state","final_trade_decision"
    )]
    print("Produced keys:", produced_keys)
    print("past_memories count:", len(final_state.get("past_memories", [])))
    print("preview len:", len((final_state.get("past_memories_preview") or "")))
    print("final decision:", final_state.get("final_trade_decision"))

    # ---- Assertions ----
    assert "market_report" in final_state and "news_report" in final_state, "Analyst reports missing"
    assert isinstance(final_state.get("past_memories"), list), "Memory Broker did not populate past_memories"
    assert isinstance(final_state.get("past_memories_preview"), str), "No preview string from Memory Broker"
    assert final_state.get("final_trade_decision") in ("FINAL: BUY", "BUY", "SELL", "HOLD"), "No final decision produced"

    print("\nTradingAgentsGraph wiring self-test passed.\n")
