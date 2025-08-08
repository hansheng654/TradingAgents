# complete_backtest_solution.py
"""
Complete backtesting solution with integrated mock for testing.
Fixes:
1. Date issues (no future dates)
2. yfinance column handling
3. Integrated mock (no separate file needed)
"""
import traceback
import os
import json
import csv
import random
from pathlib import Path
from datetime import date, timedelta, datetime
from typing import List, Dict, Any, Tuple

import yfinance as yf


###############################################################################
# MOCK AGENT (Integrated - no separate file needed)
###############################################################################
class MockTradingAgentsGraph:
    """Mock version of TradingAgentsGraph for testing."""

    def __init__(self, debug: bool = False, config: Dict[str, Any] = None):
        self.debug = debug
        self.config = config or {}
        if "random_seed" in self.config:
            random.seed(self.config["random_seed"])
        self.decision_weights = self.config.get("decision_weights", [0.33, 0.33, 0.34])

    def propagate(self, ticker: str, trade_date: Any) -> Tuple[Dict, str]:
        decisions = ["BUY", "SELL", "HOLD"]
        decision = random.choices(decisions, weights=self.decision_weights)[0]

        final_state = {
            "ticker": ticker,
            "date": str(trade_date),
            "decision": decision,
            "mock": True,
        }

        if self.debug:
            print(f"Mock: {ticker} on {trade_date} → {decision}")

        return final_state, decision


###############################################################################
# CONFIGURATION
###############################################################################
USE_MOCK = False  # CHANGE THIS TO False TO USE REAL AGENT

if not USE_MOCK:
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.default_config import DEFAULT_CONFIG
else:
    TradingAgentsGraph = MockTradingAgentsGraph
    DEFAULT_CONFIG = {}


###############################################################################
# UTILITIES
###############################################################################
def generate_trade_dates(end: date, months_back: int = 6) -> List[date]:
    """Generate trading dates going BACKWARDS from end date."""
    # Make sure we don't go into the future
    if end > date.today():
        end = date.today()

    start = end - timedelta(days=30 * months_back)

    # Find first Monday on/after start
    days_ahead = 0 - start.weekday()  # Monday is 0
    if days_ahead <= 0:
        days_ahead += 7
    first_mon = start + timedelta(days=days_ahead)

    schedule = []
    d = first_mon
    while d <= end:
        schedule.append(d)
        d += timedelta(weeks=2)  # Every 2 weeks

    return schedule


def get_close_price(ticker: str, trade_date: date) -> float:
    """Fetch the last available adjusted close price up to the given date."""
    if trade_date > date.today():
        raise RuntimeError(f"Cannot get price for future date {trade_date}")

    # Create Ticker object
    ticker_obj = yf.Ticker(ticker)

    # Fetch a small date range around the target date
    data = ticker_obj.history(
        start=trade_date - timedelta(days=5),
        end=trade_date + timedelta(days=1),
        auto_adjust=True
    )

    if data.empty:
        raise RuntimeError(f"No price data for {ticker} around {trade_date}")

    # Ensure we have the right price column
    price_col = "Close" if "Close" in data.columns else "Adj Close"
    if price_col not in data.columns:
        raise RuntimeError(f"No price column found. Columns: {data.columns.tolist()}")

    # Filter to dates up to trade_date
    available_data = data[data.index.date <= trade_date]

    if not available_data.empty:
        # Last available price before or on trade_date
        return float(available_data[price_col].iloc[-1])
    else:
        # If no earlier data, return the earliest available after
        return float(data[price_col].iloc[0])
    
def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


###############################################################################
# BACKTESTER
###############################################################################
class Backtester:
    def __init__(
        self,
        ticker: str,
        start_cash: float = 1_000.0,
        fee: float = 1.0,
        config_overrides: Dict[str, Any] = None,
        debug: bool = False,
    ):
        self.ticker = ticker.upper()
        self.initial_cash = start_cash
        self.cash = start_cash
        self.shares = 0
        self.fee = fee
        self.txn_log: List[Dict[str, Any]] = []

        # Config & agent graph
        cfg = DEFAULT_CONFIG.copy() if not USE_MOCK else {}
        cfg.update(config_overrides or {})
        self.graph = TradingAgentsGraph(debug=debug, config=cfg)

        # Paths - add timestamp to avoid overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(
            f"backtesting/{self.ticker}{'_MOCK' if USE_MOCK else ''}_{timestamp}"
        )
        safe_mkdir(self.base_dir)
        safe_mkdir(self.base_dir / "states")

        print(f"📁 Output directory: {self.base_dir}")

    def _make_serializable(self, obj):
        """Convert non-serializable objects to serializable format."""
        if obj is None:
            return None
        # Handle objects with get_text method (BeautifulSoup elements)
        if hasattr(obj, "get_text"):
            try:
                return obj.get_text()
            except:
                return str(obj)
        if hasattr(obj, "content"):  # Langchain message objects
            return {"type": obj.__class__.__name__, "content": obj.content}
        elif hasattr(obj, "dict"):  # Pydantic models
            return obj.dict()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # For any other non-serializable object, convert to string
            return str(obj)

    def trade(self, trade_date: date):
        """Run agent, execute trade, persist artifacts."""
        # Skip future dates
        if trade_date > date.today():
            print(f"⏭️  {trade_date}: Skipping future date")
            return

        try:
            final_state, decision = self.graph.propagate(self.ticker, trade_date)
        except Exception as e:
            print(f"❌ {trade_date}: propagate error → {e}")
            traceback.print_exc()
            return

        try:
            price = get_close_price(self.ticker, trade_date)
        except Exception as e:
            print(f"❌ {trade_date}: price error → {e}")
            return

        # Decision logic
        action = "HOLD"
        qty = 0

        if decision == "BUY" and self.cash > price + self.fee:
            qty = max(int((self.cash - self.fee) // price), 0)
            if qty == 0:
                action = "HOLD"
            if qty > 0:
                self.cash -= qty * price + self.fee
                self.shares += qty
                action = "BUY"

        elif decision == "SELL" and self.shares > 0:
            qty = self.shares
            proceeds = qty * price - self.fee
            self.cash += proceeds
            self.shares = 0
            action = "SELL"

        # Calculate NAV for display
        nav = self.cash + self.shares * price
        return_pct = ((nav - self.initial_cash) / self.initial_cash) * 100

        # Log transaction
        txn = {
            "date": trade_date.isoformat(),
            "decision": decision,
            "executed_action": action,
            "qty": qty,
            "price": round(price, 2),
            "fee": self.fee if action != "HOLD" else 0,
            "cash_after": round(self.cash, 2),
            "shares_after": self.shares,
            "nav": round(nav, 2),
            "return_pct": round(return_pct, 2),
        }
        self.txn_log.append(txn)

        # Emoji for better visibility
        emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}[action]

        print(
            f"{emoji} {trade_date} → {decision:<4} | exec: {action:<4} "
            f"qty={qty:>3} @ ${price:>7.2f} | "
            f"NAV: ${nav:>8.2f} ({return_pct:>+6.2f}%)"
        )

        # Persist
        csv_path = self.base_dir / "transactions.csv"
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=txn.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(txn)

        state_path = self.base_dir / "states" / f"{trade_date}.json"
        serializable = self._make_serializable(final_state)
        with open(state_path, "w") as f:
            json.dump(serializable, f, indent=2)

    def finalize(self, end_date: date):
        """Compute ending NAV and summary stats."""
        if not self.txn_log:
            print("No transactions to summarize")
            return {}

        # Use last transaction date if end_date is in future
        if end_date > date.today():
            end_date = date.today()

        try:
            latest_price = get_close_price(self.ticker, end_date)
        except:
            # Fallback to last transaction price
            latest_price = self.txn_log[-1]["price"]

        nav = self.cash + self.shares * latest_price
        start_nav = self.initial_cash

        # Time calculations
        start_date = datetime.strptime(self.txn_log[0]["date"], "%Y-%m-%d").date()
        years = (end_date - start_date).days / 365.25

        # CAGR
        if years > 0:
            cagr = ((nav / start_nav) ** (1 / years)) - 1
        else:
            cagr = 0

        # Trade statistics
        buys = sum(1 for t in self.txn_log if t["executed_action"] == "BUY")
        sells = sum(1 for t in self.txn_log if t["executed_action"] == "SELL")
        holds = sum(1 for t in self.txn_log if t["executed_action"] == "HOLD")

        summary = {
            "ticker": self.ticker,
            "is_mock": USE_MOCK,
            "period": f"{start_date} to {end_date}",
            "days": (end_date - start_date).days,
            "years": round(years, 2),
            "initial_capital": start_nav,
            "final_cash": round(self.cash, 2),
            "final_shares": self.shares,
            "latest_price": round(latest_price, 2),
            "final_nav": round(nav, 2),
            "total_return": round(nav - start_nav, 2),
            "total_return_pct": round(((nav - start_nav) / start_nav) * 100, 2),
            "cagr_pct": round(cagr * 100, 2),
            "total_trades": len(self.txn_log),
            "buys": buys,
            "sells": sells,
            "holds": holds,
        }

        # Save summary
        with open(self.base_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        # Save run config
        with open(self.base_dir / "run_config.json", "w") as f:
            json.dump(self.graph.config, f, indent=2)

        # Print nice summary
        print("\n" + "=" * 60)
        print(" BACKTEST SUMMARY ".center(60, "="))
        print("=" * 60)
        print(f"{'Mode:':<20} {'🧪 MOCK' if USE_MOCK else '🚀 REAL'}")
        print(f"{'Ticker:':<20} {summary['ticker']}")
        print(f"{'Period:':<20} {summary['period']}")
        print(f"{'Days:':<20} {summary['days']}")
        print("-" * 60)
        print(f"{'Initial Capital:':<20} ${summary['initial_capital']:,.2f}")
        print(f"{'Final NAV:':<20} ${summary['final_nav']:,.2f}")
        print(f"{'Total Return:':<20} ${summary['total_return']:,.2f}")
        print(f"{'Return %:':<20} {summary['total_return_pct']:.2f}%")
        print(f"{'CAGR:':<20} {summary['cagr_pct']:.2f}%")
        print("-" * 60)
        print(f"{'Total Trades:':<20} {summary['total_trades']}")
        print(f"{'  - Buys:':<20} {summary['buys']}")
        print(f"{'  - Sells:':<20} {summary['sells']}")
        print(f"{'  - Holds:':<20} {summary['holds']}")
        print("=" * 60)
        print(f"\n📊 Results saved to: {self.base_dir}/")

        return summary


###############################################################################
# MAIN EXECUTION
###############################################################################
if __name__ == "__main__":
    # Configuration
    TICKER = "HPE"
    END_DATE = date.today()  # This will be today 
    MONTHS_BACK = 6

    print(f"{'='*60}")
    print(f" BACKTESTING: {TICKER} ".center(60, "="))
    print(f"{'='*60}")
    print(f"Mode: {'🧪 MOCK AGENT' if USE_MOCK else '🚀 REAL AGENT'}")
    print(f"End Date: {END_DATE}")
    print(f"Lookback: {MONTHS_BACK} months")

    # Generate dates
    TRADE_DATES = generate_trade_dates(END_DATE, months_back=MONTHS_BACK)
    print(f"Trade Dates: {len(TRADE_DATES)} sessions")
    if TRADE_DATES:
        print(f"First: {TRADE_DATES[0]}, Last: {TRADE_DATES[-1]}")
    print(f"{'='*60}\n")

    if USE_MOCK:
        # Mock configuration
        custom_cfg = {
            "random_seed": 42,  # For reproducibility
            "decision_weights": [0.4, 0.3, 0.3],  # 40% BUY, 30% SELL, 30% HOLD
        }
    else:
        # Real agent configuration - YOUR CONFIG
        custom_cfg = {
            # "llm_provider": "google",
            # "backend_url": "https://generativelanguage.googleapis.com/v1",
            # "deep_think_llm": "gemini-2.5-flash-lite",
            # "quick_think_llm": "gemini-2.0-flash",
            "deep_think_llm": "gpt-5",
            "quick_think_llm": "gpt-5-mini",
            "max_debate_rounds": 1,
            "online_tools": True,  # Set to False for faster testing
        }

    # Create and run backtester
    bt = Backtester(
        ticker=TICKER,
        start_cash=1_000.0,
        fee=1.0,
        config_overrides=custom_cfg,
        debug=False,
    )

    # Run trades
    for d in TRADE_DATES:
        bt.trade(d)

    # Final report
    bt.finalize(END_DATE)
