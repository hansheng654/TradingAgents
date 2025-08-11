# TradingAgents/graph/setup.py

from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState, create_memory_broker
from tradingagents.agents.utils.agent_utils import Toolkit

# from .conditional_logic import ConditionalLogic
from tradingagents.graph.conditional_logic import ConditionalLogic


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: ChatOpenAI,
        deep_thinking_llm: ChatOpenAI,
        toolkit: Toolkit,
        tool_nodes: Dict[str, ToolNode],
        shared_memory,
        conditional_logic: ConditionalLogic,
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.toolkit = toolkit
        self.tool_nodes = tool_nodes
        self.shared_memory = shared_memory
        self.conditional_logic = conditional_logic

    def setup_graph(
        self, selected_analysts=["market", "social", "news", "fundamentals"]
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Social media analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        if len(selected_analysts) == 0:
            raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")

        # Create analyst nodes
        analyst_nodes = {}
        delete_nodes = {}
        tool_nodes = {}

        if "market" in selected_analysts:
            analyst_nodes["market"] = create_market_analyst(
                self.quick_thinking_llm, self.toolkit
            )
            delete_nodes["market"] = create_msg_delete()
            tool_nodes["market"] = self.tool_nodes["market"]

        if "social" in selected_analysts:
            analyst_nodes["social"] = create_social_media_analyst(
                self.quick_thinking_llm, self.toolkit
            )
            delete_nodes["social"] = create_msg_delete()
            tool_nodes["social"] = self.tool_nodes["social"]

        if "news" in selected_analysts:
            analyst_nodes["news"] = create_news_analyst(
                self.quick_thinking_llm, self.toolkit
            )
            delete_nodes["news"] = create_msg_delete()
            tool_nodes["news"] = self.tool_nodes["news"]

        if "fundamentals" in selected_analysts:
            analyst_nodes["fundamentals"] = create_fundamentals_analyst(
                self.quick_thinking_llm, self.toolkit
            )
            delete_nodes["fundamentals"] = create_msg_delete()
            tool_nodes["fundamentals"] = self.tool_nodes["fundamentals"]

        # Create researcher and manager nodes
        bull_researcher_node = create_bull_researcher(
            self.quick_thinking_llm, self.shared_memory
        )
        bear_researcher_node = create_bear_researcher(
            self.quick_thinking_llm, self.shared_memory
        )
        research_manager_node = create_research_manager(
            self.deep_thinking_llm, self.shared_memory
        )
        trader_node = create_trader(self.quick_thinking_llm, self.shared_memory)

        # Create risk analysis nodes
        risky_analyst = create_risky_debator(self.quick_thinking_llm)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
        safe_analyst = create_safe_debator(self.quick_thinking_llm)
        risk_manager_node = create_risk_manager(
            self.deep_thinking_llm, self.shared_memory
        )

        # Create workflow
        workflow = StateGraph(AgentState)

        # Add analyst nodes to the graph
        for analyst_type, node in analyst_nodes.items():
            workflow.add_node(f"{analyst_type.capitalize()} Analyst", node)
            workflow.add_node(
                f"Msg Clear {analyst_type.capitalize()}", delete_nodes[analyst_type]
            )
            workflow.add_node(f"tools_{analyst_type}", tool_nodes[analyst_type])

        # Add other nodes
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Risky Analyst", risky_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Safe Analyst", safe_analyst)
        workflow.add_node("Risk Judge", risk_manager_node)

        # Memory Broker node: compute memories once for all downstream agents
        memory_broker_node = create_memory_broker(
            self.shared_memory,
            n_matches=2,
            min_chars=800,
            summary_max_tokens=1200,
            llm_only_if_needed=True,
            preview_trigger_chars=2048,
        )
        workflow.add_node("Memory Broker", memory_broker_node)

        # Define edges
        # Start with the first analyst
        first_analyst = selected_analysts[0]
        workflow.add_edge(START, f"{first_analyst.capitalize()} Analyst")

        # Connect analysts in sequence
        for i, analyst_type in enumerate(selected_analysts):
            current_analyst = f"{analyst_type.capitalize()} Analyst"
            current_tools = f"tools_{analyst_type}"
            current_clear = f"Msg Clear {analyst_type.capitalize()}"

            # Add conditional edges for current analyst, mapping keys to match ConditionalLogic returns
            workflow.add_conditional_edges(
                current_analyst,
                getattr(self.conditional_logic, f"should_continue_{analyst_type}"),
                {
                    f"tools_{analyst_type}": current_tools,
                    f"Msg Clear {analyst_type.capitalize()}": current_clear,
                },
            )
            workflow.add_edge(current_tools, current_analyst)

            # Connect to next analyst or to Memory Broker if this is the last analyst
            if i < len(selected_analysts) - 1:
                next_analyst = f"{selected_analysts[i+1].capitalize()} Analyst"
                workflow.add_edge(current_clear, next_analyst)
            else:
                workflow.add_edge(current_clear, "Memory Broker")

        # Add remaining edges
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "Research Manager": "Research Manager",
            },
        )

        # After computing memories, proceed to the debate
        workflow.add_edge("Memory Broker", "Bull Researcher")

        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Risky Analyst")
        workflow.add_conditional_edges(
            "Risky Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Safe Analyst": "Safe Analyst",
                "Neutral Analyst": "Neutral Analyst",
                "Risk Judge": "Risk Judge",
            },
        )
        workflow.add_conditional_edges(
            "Safe Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Neutral Analyst": "Neutral Analyst",
                "Risky Analyst": "Risky Analyst",
                "Risk Judge": "Risk Judge",
            },
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Risky Analyst": "Risky Analyst",
                "Safe Analyst": "Safe Analyst",
                "Risk Judge": "Risk Judge",
            },
        )

        workflow.add_edge("Risk Judge", END)


        # Compile and return
        return workflow.compile()


# --- Lightweight self-test harness ---
if __name__ == "__main__":
    """
    Lightweight wiring test for GraphSetup and the Memory Broker.
    This avoids real LLM/tool calls by monkeypatching the agent node factories
    with simple no-op nodes, and using a dummy shared memory object.

    Run:
        python tradingagents/graph/setup.py
    Expected:
        - Graph compiles
        - Single run produces analyst reports
        - Memory Broker populates past_memories & preview string
        - No infinite loops; research moves to manager, then trader, then risk judge.
    """
    from types import SimpleNamespace
    from datetime import datetime

    # ---- Dummy shared memory (no network) ----
    class DummySharedMemory:
        def __init__(self):
            self.calls = 0
        def get_memories(self, current_situation, n_matches=2):
            self.calls += 1
            return [
                {
                    "matched_situation": "Prior tech selloff with rising yields",
                    "recommendation": "Reduce high-duration growth. Tilt to cash-flow positive names.",
                    "similarity_score": 0.71,
                },
                {
                    "matched_situation": "Rotation into defensives during earnings",
                    "recommendation": "Increase exposure to staples/utilities; hedge beta.",
                    "similarity_score": 0.54,
                },
            ][:n_matches]

    # ---- Dummy conditional logic ----
    class DummyConditionalLogic:
        # For analysts: always choose the 'clear' branch (advance), avoid looping on tools
        def __getattr__(self, name):
            if name.startswith("should_continue_"):
                return lambda state: "clear"  # branch label must match keys in mapping
            raise AttributeError(name)
        # For debate: pick Research Manager to end debate
        def should_continue_debate(self, state):
            return "Research Manager"
        # For risk: jump straight to Risk Judge
        def should_continue_risk_analysis(self, state):
            return "Risk Judge"

    # ---- Monkeypatch agent node creators with trivial nodes ----
    # Each returns a callable(state)->dict that sets expected fields.
    from datetime import datetime, timezone
    def _mk_report(name, key):
        def node(state):
            # Make the analyst report long enough to exceed broker min_chars when combined
            filler = (" " + name + " context with numbers 123 and tickers AAPL NVDA MSFT, macro CPI 3.2% YoY, EPS +10%.") * 60
            return {key: f"{name} report OK at {datetime.now(timezone.utc).isoformat()}." + filler}
        return node
    def _mk_clear():
        def node(state):
            return {}
        return node
    def _mk_pass(name, key):
        def node(state):
            # Append to the messages log if present, else just set a marker
            return {key: f"{name} OK"}
        return node

    # Patch imported factory functions in this module's globals
    g = globals()
    g["create_market_analyst"] = lambda llm, tk: _mk_report("Market", "market_report")
    g["create_social_media_analyst"] = lambda llm, tk: _mk_report("Social", "sentiment_report")
    g["create_news_analyst"] = lambda llm, tk: _mk_report("News", "news_report")
    g["create_fundamentals_analyst"] = lambda llm, tk: _mk_report("Fundamentals", "fundamentals_report")
    g["create_msg_delete"] = lambda: _mk_clear()

    g["create_bull_researcher"] = lambda llm, mem: _mk_pass("Bull", "investment_debate_state")
    g["create_bear_researcher"] = lambda llm, mem: _mk_pass("Bear", "investment_debate_state")
    g["create_research_manager"] = lambda llm, mem: _mk_pass("Research Manager", "investment_debate_state")

    g["create_trader"] = lambda llm, mem: _mk_pass("Trader", "trader_investment_plan")

    g["create_risky_debator"] = lambda llm: _mk_pass("Risky", "risk_debate_state")
    g["create_neutral_debator"] = lambda llm: _mk_pass("Neutral", "risk_debate_state")
    g["create_safe_debator"] = lambda llm: _mk_pass("Safe", "risk_debate_state")
    g["create_risk_manager"] = lambda llm, mem: _mk_pass("Risk Judge", "risk_debate_state")

    # Tool nodes can be simple no-op functions
    tools = {k: (lambda state: {}) for k in ["market", "social", "news", "fundamentals"]}


    # Build the graph
    setup = GraphSetup(
        quick_thinking_llm=None,
        deep_thinking_llm=None,
        toolkit=SimpleNamespace(),
        tool_nodes=tools,
        shared_memory=DummySharedMemory(),
        conditional_logic=DummyConditionalLogic(),
    )

    app = setup.setup_graph(selected_analysts=["market", "news"])  # keep it small

    # Run once with a minimal initial state
    init_state = {
        "ticker": "TEST",
        "analysis_date": "2025-01-01",
        # fields the graph may expect
        "messages": [],
    }

    print("\n== Running wiring self-test ==\n")
    out = app.invoke(init_state)

    # Basic assertions / prints
    keys = [k for k in out.keys() if k in (
        "market_report", "news_report", "past_memories", "past_memories_preview", "past_memories_str",
        "investment_debate_state", "trader_investment_plan", "risk_debate_state"
    )]

    print("Produced keys:", keys)
    print("past_memories count:", len(out.get("past_memories", [])))
    print("preview len:", len((out.get("past_memories_preview") or "")))

    assert "market_report" in out and "news_report" in out, "Analyst reports missing"
    assert isinstance(out.get("past_memories"), list), "Memory Broker did not populate past_memories"
    assert isinstance(out.get("past_memories_preview"), str), "No preview string from Memory Broker"

    print("\nGraph wiring self-test passed.\n")
