# mock_trading_agent.py
import random
from datetime import date
from typing import Dict, Any, Tuple

class MockTradingAgentsGraph:
    """Mock version of TradingAgentsGraph for testing the backtester."""
    
    def __init__(self, debug: bool = False, config: Dict[str, Any] = None):
        self.debug = debug
        self.config = config or {}
        # You can control randomness with a seed for reproducible tests
        if "random_seed" in self.config:
            random.seed(self.config["random_seed"])
        
        # Optional: Add bias to make testing more interesting
        self.decision_weights = self.config.get("decision_weights", [0.33, 0.33, 0.34])
        
    def propagate(self, ticker: str, trade_date: Any) -> Tuple[Dict, str]:
        """
        Mock propagate function that returns random decisions.
        
        Returns:
            Tuple of (final_state, decision)
            - final_state: Mock state dictionary
            - decision: One of "BUY", "SELL", "HOLD"
        """
        # Generate random decision with optional weights
        decisions = ["BUY", "SELL", "HOLD"]
        decision = random.choices(decisions, weights=self.decision_weights)[0]
        
        # Create mock state (you can make this more elaborate if needed)
        final_state = {
            "ticker": ticker,
            "date": str(trade_date),
            "decision": decision,
            "mock_data": {
                "confidence": random.random(),
                "signals": {
                    "technical": random.choice(["bullish", "bearish", "neutral"]),
                    "fundamental": random.choice(["strong", "weak", "mixed"]),
                    "sentiment": random.choice(["positive", "negative", "neutral"])
                }
            }
        }
        
        if self.debug:
            print(f"Mock Agent: {ticker} on {trade_date} → {decision}")
        
        return final_state, decision


# Alternative: Simple strategy-based mock for more realistic testing
class StrategyMockTradingAgentsGraph:
    """Mock with simple strategies instead of pure randomness."""
    
    def __init__(self, debug: bool = False, config: Dict[str, Any] = None):
        self.debug = debug
        self.config = config or {}
        self.strategy = self.config.get("strategy", "momentum")  # momentum, contrarian, or random
        self.last_decisions = {}  # Track decisions per ticker
        
    def propagate(self, ticker: str, trade_date: Any) -> Tuple[Dict, str]:
        """Mock propagate with simple strategies."""
        
        if self.strategy == "random":
            decision = random.choice(["BUY", "SELL", "HOLD"])
        
        elif self.strategy == "momentum":
            # Simple momentum: tend to hold trends
            if ticker not in self.last_decisions:
                decision = random.choice(["BUY", "HOLD"])
                self.last_decisions[ticker] = decision
            else:
                last = self.last_decisions[ticker]
                if last == "BUY":
                    # After buying, likely to hold or sell
                    decision = random.choices(["HOLD", "SELL"], weights=[0.7, 0.3])[0]
                elif last == "SELL":
                    # After selling, likely to wait or buy back
                    decision = random.choices(["HOLD", "BUY"], weights=[0.6, 0.4])[0]
                else:  # HOLD
                    # When holding, might break either way
                    decision = random.choices(["BUY", "SELL", "HOLD"], weights=[0.3, 0.3, 0.4])[0]
                self.last_decisions[ticker] = decision
        
        elif self.strategy == "contrarian":
            # Contrarian: buy low, sell high pattern
            # Use date to create a cycle
            day_num = hash(str(trade_date)) % 10
            if day_num <= 3:
                decision = "BUY"
            elif day_num >= 7:
                decision = "SELL"
            else:
                decision = "HOLD"
        
        else:
            decision = "HOLD"  # Default fallback
        
        final_state = {
            "ticker": ticker,
            "date": str(trade_date),
            "decision": decision,
            "strategy": self.strategy
        }
        
        if self.debug:
            print(f"Mock Agent ({self.strategy}): {ticker} on {trade_date} → {decision}")
        
        return final_state, decision


# Test file showing how to use the mock
if __name__ == "__main__":
    from datetime import date, timedelta
    
    print("Testing Random Mock:")
    mock1 = MockTradingAgentsGraph(debug=True, config={"random_seed": 42})
    for i in range(5):
        test_date = date.today() - timedelta(days=i*14)
        _, decision = mock1.propagate("NVDA", test_date)
    
    print("\nTesting Weighted Random Mock (buy-heavy):")
    mock2 = MockTradingAgentsGraph(
        debug=True, 
        config={"decision_weights": [0.5, 0.2, 0.3]}  # 50% BUY, 20% SELL, 30% HOLD
    )
    for i in range(5):
        test_date = date.today() - timedelta(days=i*14)
        _, decision = mock2.propagate("NVDA", test_date)
    
    print("\nTesting Momentum Strategy Mock:")
    mock3 = StrategyMockTradingAgentsGraph(
        debug=True,
        config={"strategy": "momentum"}
    )
    for i in range(10):
        test_date = date.today() - timedelta(days=i*14)
        _, decision = mock3.propagate("AAPL", test_date)