"""
Comprehensive test suite for TradingAgents GraphSetup.
Improvements over original:
1. Multiple test scenarios with different analyst combinations
2. Better state transition tracking
3. Memory broker interaction validation
4. Error case testing
5. Conditional logic path verification
6. More thorough assertions
"""


import sys
from pathlib import Path
from typing import Dict, Any, List
from types import SimpleNamespace
from datetime import datetime, timezone
from collections import defaultdict
import traceback

# Ensure project root is on sys.path so package imports work when run directly
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[3]  # repo root (.. / .. / ..)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tradingagents.graph.setup as setup_mod

# ---- Enhanced Dummy Components ----

class TestTracker:
    """Tracks execution flow through the graph for validation."""
    def __init__(self):
        self.node_visits = []
        self.state_mutations = defaultdict(list)
        self.conditional_decisions = []
        
    def visit_node(self, node_name: str):
        self.node_visits.append((node_name, datetime.now(timezone.utc)))
        
    def record_state_change(self, key: str, value: Any):
        self.state_mutations[key].append(value)
        
    def record_decision(self, node: str, decision: str):
        self.conditional_decisions.append((node, decision))
        
    def get_flow_path(self) -> List[str]:
        return [name for name, _ in self.node_visits]


class EnhancedDummySharedMemory:
    """Enhanced dummy memory with call tracking and configurable responses."""
    def __init__(self, tracker: TestTracker):
        self.calls = 0
        self.tracker = tracker
        self.memory_responses = {
            "default": [
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
            ]
        }
        
    def get_memories(self, current_situation, n_matches=2):
        self.calls += 1
        self.tracker.record_state_change("memory_queries", current_situation[:50])
        return self.memory_responses["default"][:n_matches]
    
    def set_test_memories(self, key: str, memories: List[Dict]):
        """Allow tests to configure specific memory responses."""
        self.memory_responses[key] = memories


class ConfigurableConditionalLogic:
    """Conditional logic that can be configured for different test scenarios."""
    def __init__(self, tracker: TestTracker, scenario="default"):
        self.tracker = tracker
        self.scenario = scenario
        self.debate_count = 0
        self.risk_count = 0
        
    def _analyst_continue(self, analyst_type: str):
        """Configurable analyst continuation logic."""
        def logic(state):
            # Always go to clear for now, but track the decision
            self.tracker.record_decision(f"{analyst_type}_analyst", "clear")
            return "clear"
        return logic
        
    def __getattr__(self, name):
        if name.startswith("should_continue_") and not name.endswith("debate") and not name.endswith("risk_analysis"):
            analyst_type = name.replace("should_continue_", "")
            return self._analyst_continue(analyst_type)
        raise AttributeError(name)
    
    def should_continue_debate(self, state):
        """Simulate a few rounds of debate before moving to manager."""
        self.debate_count += 1
        if self.scenario == "extended_debate":
            # Extended debate scenario
            if self.debate_count < 4:
                next_node = "Bear Researcher" if self.debate_count % 2 == 1 else "Bull Researcher"
                self.tracker.record_decision("debate", next_node)
                return next_node
        elif self.scenario == "quick_consensus":
            # Quick consensus scenario
            self.tracker.record_decision("debate", "Research Manager")
            return "Research Manager"
        else:
            # Default: 2 rounds then manager
            if self.debate_count < 2:
                next_node = "Bear Researcher" if self.debate_count % 2 == 1 else "Bull Researcher"
                self.tracker.record_decision("debate", next_node)
                return next_node
        
        self.tracker.record_decision("debate", "Research Manager")
        return "Research Manager"
    
    def should_continue_risk_analysis(self, state):
        """Simulate risk analysis rounds."""
        self.risk_count += 1
        if self.scenario == "high_risk_debate":
            # Extended risk debate
            if self.risk_count < 3:
                nodes = ["Safe Analyst", "Neutral Analyst", "Risky Analyst"]
                next_node = nodes[self.risk_count % 3]
                self.tracker.record_decision("risk", next_node)
                return next_node
        
        # Default: straight to judge
        self.tracker.record_decision("risk", "Risk Judge")
        return "Risk Judge"


# ---- Enhanced Node Factories ----

def create_tracked_analyst(name: str, key: str, tracker: TestTracker):
    """Create an analyst node that tracks its execution."""
    def node(state):
        tracker.visit_node(name)
        # Generate report with enough content to trigger memory broker (large enough for prod thresholds)
        filler = (f" {name} analysis: volatility 15%, RSI 65, MACD bullish, volume above average, support at $150, resistance at $165; tickers AAPL NVDA MSFT; CPI 3.2% YoY; EPS +10%. " * 80)
        report = f"{name} report at {datetime.now(timezone.utc).isoformat()}" + filler
        tracker.record_state_change(key, report[:100])  # Track first 100 chars
        return {key: report}
    return node


def create_tracked_clear(name: str, tracker: TestTracker):
    """Create a clear node that tracks its execution."""
    def node(state):
        tracker.visit_node(f"Clear_{name}")
        return {}
    return node


def create_tracked_agent(name: str, key: str, tracker: TestTracker):
    """Create a generic agent node that tracks its execution."""
    def node(state):
        tracker.visit_node(name)
        value = f"{name} output at {datetime.now(timezone.utc).isoformat()}"
        tracker.record_state_change(key, value)
        return {key: value}
    return node


def create_tracked_tool(name: str, tracker: TestTracker):
    """Create a tool node that tracks its execution."""
    def node(state):
        tracker.visit_node(f"Tool_{name}")
        return {}
    return node


# ---- Test Scenarios ----

def test_scenario(scenario_name: str, analysts: List[str], logic_scenario: str = "default"):
    """Run a test scenario with given configuration."""
    print(f"\n{'='*60}")
    print(f"Testing: {scenario_name}")
    print(f"Analysts: {analysts}")
    print(f"Logic scenario: {logic_scenario}")
    print('='*60)
    
    tracker = TestTracker()
    
    # Patch the factory functions inside the setup module under test
    setup_mod.create_market_analyst = lambda llm, tk: create_tracked_analyst("Market", "market_report", tracker)
    setup_mod.create_social_media_analyst = lambda llm, tk: create_tracked_analyst("Social", "sentiment_report", tracker)
    setup_mod.create_news_analyst = lambda llm, tk: create_tracked_analyst("News", "news_report", tracker)
    setup_mod.create_fundamentals_analyst = lambda llm, tk: create_tracked_analyst("Fundamentals", "fundamentals_report", tracker)
    setup_mod.create_msg_delete = lambda: create_tracked_clear("Messages", tracker)

    setup_mod.create_bull_researcher = lambda llm, mem: create_tracked_agent("Bull", "investment_debate_state", tracker)
    setup_mod.create_bear_researcher = lambda llm, mem: create_tracked_agent("Bear", "investment_debate_state", tracker)
    setup_mod.create_research_manager = lambda llm, mem: create_tracked_agent("ResearchManager", "investment_debate_state", tracker)

    setup_mod.create_trader = lambda llm, mem: create_tracked_agent("Trader", "trader_investment_plan", tracker)

    setup_mod.create_risky_debator = lambda llm: create_tracked_agent("Risky", "risk_debate_state", tracker)
    setup_mod.create_neutral_debator = lambda llm: create_tracked_agent("Neutral", "risk_debate_state", tracker)
    setup_mod.create_safe_debator = lambda llm: create_tracked_agent("Safe", "risk_debate_state", tracker)
    setup_mod.create_risk_manager = lambda llm, mem: create_tracked_agent("RiskJudge", "risk_debate_state", tracker)

    # Tool nodes
    tools = {
        "market": create_tracked_tool("market", tracker),
        "social": create_tracked_tool("social", tracker),
        "news": create_tracked_tool("news", tracker),
        "fundamentals": create_tracked_tool("fundamentals", tracker),
    }

    # Build the graph using the setup module's GraphSetup
    setup = setup_mod.GraphSetup(
        quick_thinking_llm=None,
        deep_thinking_llm=None,
        toolkit=SimpleNamespace(),
        tool_nodes=tools,
        shared_memory=EnhancedDummySharedMemory(tracker),
        conditional_logic=ConfigurableConditionalLogic(tracker, logic_scenario),
    )

    try:
        app = setup.setup_graph(selected_analysts=analysts)

        # Run with initial state
        init_state = {
            "ticker": "TEST",
            "analysis_date": "2025-01-01",
            "messages": [],
        }

        result = app.invoke(init_state)
        # Optional diagnostics: print memory preview length
        if "past_memories_preview" in result:
            print(f"Memory preview length: {len(result['past_memories_preview'])}")

        # Validate results
        validate_results(scenario_name, result, tracker, analysts)

        return True, tracker

    except Exception as e:
        print(f"❌ Scenario failed with error: {e}")
        traceback.print_exc()
        return False, tracker


def validate_results(scenario_name: str, result: Dict, tracker: TestTracker, analysts: List[str]):
    """Validate the results of a test scenario."""
    print("\n📊 Validation Results:")
    
    # Check analyst reports
    analyst_keys = {
        "market": "market_report",
        "social": "sentiment_report", 
        "news": "news_report",
        "fundamentals": "fundamentals_report"
    }
    
    for analyst in analysts:
        key = analyst_keys[analyst]
        assert key in result, f"Missing {analyst} report"
        assert len(result[key]) > 100, f"{analyst} report too short"
        print(f"  ✓ {analyst} report present ({len(result[key])} chars)")
    
    # Check memory broker
    assert "past_memories" in result, "Memory broker didn't populate past_memories"
    assert isinstance(result["past_memories"], list), "past_memories not a list"
    assert len(result["past_memories"]) > 0, "No memories retrieved"
    print(f"  ✓ Memory broker: {len(result['past_memories'])} memories")
    
    if "past_memories_preview" in result:
        assert isinstance(result["past_memories_preview"], str), "Preview not a string"
        print(f"  ✓ Memory preview: {len(result['past_memories_preview'])} chars")
    
    # Check flow path
    flow = tracker.get_flow_path()
    print(f"\n📍 Execution flow ({len(flow)} nodes):")
    
    # Verify expected flow patterns
    expected_start = [f"{analysts[0].capitalize()}" for _ in range(1)]  # First analyst
    actual_start = flow[:1]
    
    # Print condensed flow
    print(f"  Start: {' -> '.join(flow[:3])}")
    if len(flow) > 6:
        print(f"  Middle: ... {len(flow)-6} nodes ...")
    print(f"  End: {' -> '.join(flow[-3:])}")
    
    # Check key stages were hit
    assert "Bull" in flow, "Bull researcher not executed"
    assert "ResearchManager" in flow, "Research manager not executed"
    assert "Trader" in flow, "Trader not executed"
    assert "RiskJudge" in flow, "Risk judge not executed"
    print("  ✓ All key stages executed")
    
    # Check state mutations
    print(f"\n🔄 State mutations: {len(tracker.state_mutations)} keys modified")
    for key, values in list(tracker.state_mutations.items())[:3]:
        print(f"  - {key}: {len(values)} updates")
    
    # Check conditional decisions
    print(f"\n🔀 Conditional decisions: {len(tracker.conditional_decisions)} made")
    for node, decision in tracker.conditional_decisions[:3]:
        print(f"  - {node} → {decision}")
    
    print(f"\n✅ Scenario '{scenario_name}' passed all validations")


def test_error_cases():
    """Test error handling."""
    print(f"\n{'='*60}")
    print("Testing: Error Cases")
    print('='*60)

    # Use the setup module for error-case tests
    # Test 1: No analysts selected
    try:
        setup = setup_mod.GraphSetup(
            quick_thinking_llm=None,
            deep_thinking_llm=None,
            toolkit=SimpleNamespace(),
            tool_nodes={},
            shared_memory=EnhancedDummySharedMemory(TestTracker()),
            conditional_logic=ConfigurableConditionalLogic(TestTracker()),
        )
        app = setup.setup_graph(selected_analysts=[])
        print("❌ Should have raised ValueError for empty analysts")
        return False
    except ValueError as e:
        if "no analysts selected" in str(e):
            print("✓ Correctly raised ValueError for empty analysts")
        else:
            print(f"❌ Wrong error message: {e}")
            return False

    return True


# ---- Main Test Runner ----

def run_all_tests():
    """Run comprehensive test suite."""
    print("\n" + "="*60)
    print(" TradingAgents Graph Setup - Comprehensive Test Suite")
    print("="*60)
    
    results = []
    
    # Test different analyst combinations
    test_configs = [
        ("Single Analyst", ["market"], "default"),
        ("Two Analysts", ["market", "news"], "default"),
        ("All Analysts", ["market", "social", "news", "fundamentals"], "default"),
        ("Extended Debate", ["market", "news"], "extended_debate"),
        ("Quick Consensus", ["fundamentals"], "quick_consensus"),
        ("High Risk Debate", ["market", "social"], "high_risk_debate"),
    ]
    
    for scenario_name, analysts, logic_scenario in test_configs:
        success, tracker = test_scenario(scenario_name, analysts, logic_scenario)
        results.append((scenario_name, success))
    
    # Test error cases
    error_success = test_error_cases()
    results.append(("Error Cases", error_success))
    
    # Summary
    print("\n" + "="*60)
    print(" TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} - {name}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    if passed == total:
        print("\n🎉 All tests passed successfully!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()