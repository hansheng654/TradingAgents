 # lightweight per-process cache for batch indicator calls (avoids duplicate work in a run)
_batch_indicators_cache: dict[tuple[str, str, int, tuple[str, ...]], str] = {}
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from typing import List
from typing import Annotated
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import RemoveMessage
from langchain_core.tools import tool
from datetime import date, timedelta, datetime
import functools
import pandas as pd
import os
from dateutil.relativedelta import relativedelta
from langchain_openai import ChatOpenAI
import tradingagents.dataflows.interface as interface
from tradingagents.default_config import DEFAULT_CONFIG
from langchain_core.messages import HumanMessage
import traceback
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handle_tool_error(func_name: str, error: Exception, **kwargs) -> str:
    """
    Centralized error handling for toolkit functions
    Prints to CLI and returns formatted error message
    """
    error_msg = f"ERROR in {func_name}: {str(error)}"
    full_error = f"""
================================================================================
TOOLKIT ERROR in {func_name}
================================================================================
Parameters: {kwargs}
Error Type: {type(error).__name__}
Error Message: {str(error)}
Traceback:
{traceback.format_exc()}
================================================================================
"""
    
    # Print to CLI for immediate visibility
    print(full_error)
    
    # Also log it
    logger.error(full_error)
    
    # Return a structured error message for the calling system
    return f"TOOL_ERROR: {func_name} failed - {str(error)}. Check CLI output for full details."


def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]
        
        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]
        
        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")
        
        return {"messages": removal_operations + [placeholder]}
    
    return delete_messages


class Toolkit:
    _config = DEFAULT_CONFIG.copy()

    @classmethod
    def update_config(cls, config):
        """Update the class-level configuration."""
        cls._config.update(config)

    @property
    def config(self):
        """Access the configuration."""
        return self._config

    def __init__(self, config=None):
        if config:
            self.update_config(config)

    @staticmethod
    @tool
    def get_reddit_news(
        curr_date: Annotated[str, "Date you want to get news for in yyyy-mm-dd format"],
    ) -> str:
        """
        Retrieve global news from Reddit within a specified time frame.
        Args:
            curr_date (str): Date you want to get news for in yyyy-mm-dd format
        Returns:
            str: A formatted dataframe containing the latest global news from Reddit in the specified time frame.
        """
        try:
            print(f"[TOOLKIT] Fetching Reddit news for {curr_date}")
            global_news_result = interface.get_reddit_global_news(curr_date, 7, 5)
            print(f"[TOOLKIT] Successfully retrieved Reddit news")
            return global_news_result
        except Exception as e:
            return handle_tool_error("get_reddit_news", e, curr_date=curr_date)

    @staticmethod
    @tool
    def get_finnhub_news(
        ticker: Annotated[
            str,
            "Search query of a company, e.g. 'AAPL, TSM, etc.",
        ],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    ):
        """
        Retrieve the latest news about a given stock from Finnhub within a date range
        Args:
            ticker (str): Ticker of a company. e.g. AAPL, TSM
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
        Returns:
            str: A formatted dataframe containing news about the company within the date range from start_date to end_date
        """
        try:
            print(f"[TOOLKIT] Fetching Finnhub news for {ticker} from {start_date} to {end_date}")
            
            end_date_str = end_date
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
            look_back_days = (end_date - start_date).days

            finnhub_news_result = interface.get_finnhub_news(
                ticker, end_date_str, look_back_days
            )
            print(f"[TOOLKIT] Successfully retrieved Finnhub news for {ticker}")
            return finnhub_news_result
        except Exception as e:
            return handle_tool_error("get_finnhub_news", e, ticker=ticker, start_date=start_date, end_date=end_date)

    @staticmethod
    @tool
    def get_reddit_stock_info(
        ticker: Annotated[
            str,
            "Ticker of a company. e.g. AAPL, TSM",
        ],
        curr_date: Annotated[str, "Current date you want to get news for"],
    ) -> str:
        """
        Retrieve the latest news about a given stock from Reddit, given the current date.
        Args:
            ticker (str): Ticker of a company. e.g. AAPL, TSM
            curr_date (str): current date in yyyy-mm-dd format to get news for
        Returns:
            str: A formatted dataframe containing the latest news about the company on the given date
        """
        try:
            print(f"[TOOLKIT] Fetching Reddit stock info for {ticker} on {curr_date}")
            stock_news_results = interface.get_reddit_company_news(ticker, curr_date, 7, 5)
            print(f"[TOOLKIT] Successfully retrieved Reddit stock info for {ticker}")
            return stock_news_results
        except Exception as e:
            return handle_tool_error("get_reddit_stock_info", e, ticker=ticker, curr_date=curr_date)

    @staticmethod
    @tool
    def get_YFin_data(
        symbol: Annotated[str, "ticker symbol of the company"],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    ) -> str:
        """
        Retrieve the stock price data for a given ticker symbol from Yahoo Finance.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
        Returns:
            str: A formatted dataframe containing the stock price data for the specified ticker symbol in the specified date range.
        """
        try:
            print(f"[TOOLKIT] Fetching Yahoo Finance data for {symbol} from {start_date} to {end_date}")
            result_data = interface.get_YFin_data(symbol, start_date, end_date)
            print(f"[TOOLKIT] Successfully retrieved Yahoo Finance data for {symbol}")
            return result_data
        except Exception as e:
            return handle_tool_error("get_YFin_data", e, symbol=symbol, start_date=start_date, end_date=end_date)

    @staticmethod
    @tool
    def get_YFin_data_online(
        symbol: Annotated[str, "ticker symbol of the company"],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    ) -> str:
        """
        Retrieve the stock price data for a given ticker symbol from Yahoo Finance.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
        Returns:
            str: A formatted dataframe containing the stock price data for the specified ticker symbol in the specified date range.
        """
        try:
            print(f"[TOOLKIT] Fetching Yahoo Finance data ONLINE for {symbol} from {start_date} to {end_date}")
            result_data = interface.get_YFin_data_online(symbol, start_date, end_date)
            print(f"[TOOLKIT] Successfully retrieved Yahoo Finance data ONLINE for {symbol}")
            return result_data
        except Exception as e:
            return handle_tool_error("get_YFin_data_online", e, symbol=symbol, start_date=start_date, end_date=end_date)

    @staticmethod
    @tool
    def get_stockstats_indicators_report(
        symbol: Annotated[str, "ticker symbol of the company"],
        indicator: Annotated[
            str, "technical indicator to get the analysis and report of"
        ],
        curr_date: Annotated[
            str, "The current trading date you are trading on, YYYY-mm-dd"
        ],
        look_back_days: Annotated[int, "how many days to look back"] = 30,
    ) -> str:
        """
        Retrieve stock stats indicators for a given ticker symbol and indicator.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            indicator (str): Technical indicator to get the analysis and report of
            curr_date (str): The current trading date you are trading on, YYYY-mm-dd
            look_back_days (int): How many days to look back, default is 30
        Returns:
            str: A formatted dataframe containing the stock stats indicators for the specified ticker symbol and indicator.
        """
        try:
            print(f"[TOOLKIT] Fetching stock stats indicators for {symbol}, indicator: {indicator}, date: {curr_date}")
            result_stockstats = interface.get_stock_stats_indicators_window(
                symbol, indicator, curr_date, look_back_days, False
            )
            print(f"[TOOLKIT] Successfully retrieved stock stats indicators for {symbol}")
            return result_stockstats
        except Exception as e:
            return handle_tool_error("get_stockstats_indicators_report", e, 
                                   symbol=symbol, indicator=indicator, curr_date=curr_date, look_back_days=look_back_days)

    @staticmethod
    @tool
    def get_stockstats_indicators_report_online(
        symbol: Annotated[str, "ticker symbol of the company"],
        indicator: Annotated[
            str, "technical indicator to get the analysis and report of"
        ],
        curr_date: Annotated[
            str, "The current trading date you are trading on, YYYY-mm-dd"
        ],
        look_back_days: Annotated[int, "how many days to look back"] = 30,
    ) -> str:
        """
        Retrieve stock stats indicators for a given ticker symbol and indicator.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            indicator (str): Technical indicator to get the analysis and report of
            curr_date (str): The current trading date you are trading on, YYYY-mm-dd
            look_back_days (int): How many days to look back, default is 30
        Returns:
            str: A formatted dataframe containing the stock stats indicators for the specified ticker symbol and indicator.
        """
        try:
            print(f"[TOOLKIT] Fetching stock stats indicators ONLINE for {symbol}, indicator: {indicator}, date: {curr_date}")
            result_stockstats = interface.get_stock_stats_indicators_window(
                symbol, indicator, curr_date, look_back_days, True
            )
            print(f"[TOOLKIT] Successfully retrieved stock stats indicators ONLINE for {symbol}")
            return result_stockstats
        except Exception as e:
            return handle_tool_error("get_stockstats_indicators_report_online", e, 
                                   symbol=symbol, indicator=indicator, curr_date=curr_date, look_back_days=look_back_days)

    @staticmethod
    @tool
    def get_stockstats_multi_indicators_report_online(
        symbol: Annotated[str, "ticker symbol of the company"],
        indicators: Annotated[list[str], "technical indicators to compute (e.g., ['rsi','macd','close_50_sma'])"],
        curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
        look_back_days: Annotated[int, "how many days to look back"] = 30,
    ) -> str:
        """
        Retrieve **multiple** stock stats indicators for a given ticker in **one** tool call.
        Internally loops over the existing single-indicator interface to avoid new dependencies.
        Returns a combined, human-readable report.
        """
        try:
            # normalize & cache key
            norm_inds = tuple(sorted([i.strip() for i in indicators if i and isinstance(i, str)]))
            cache_key = (symbol.upper().strip(), curr_date, int(look_back_days), norm_inds)
            if cache_key in _batch_indicators_cache:
                print(f"[TOOLKIT][CACHE HIT] Stock stats (BATCH) ONLINE for {symbol}, {len(norm_inds)} indicators, date: {curr_date}")
                return _batch_indicators_cache[cache_key]

            print(f"[TOOLKIT] Fetching stock stats (BATCH) ONLINE for {symbol}, {len(norm_inds)} indicators, date: {curr_date}")
            reports: list[str] = []
            for ind in norm_inds:
                try:
                    r = interface.get_stock_stats_indicators_window(
                        symbol, ind, curr_date, look_back_days, True
                    )
                except Exception as ie:
                    r = handle_tool_error(
                        "get_stockstats_multi_indicators_report_online/subcall",
                        ie, symbol=symbol, indicator=ind, curr_date=curr_date, look_back_days=look_back_days,
                    )
                reports.append(str(r))

            combined = "\n\n".join(reports)
            print(f"[TOOLKIT] Successfully retrieved stock stats (BATCH) ONLINE for {symbol}")
            _batch_indicators_cache[cache_key] = combined
            return combined
        except Exception as e:
            return handle_tool_error(
                "get_stockstats_multi_indicators_report_online", e,
                symbol=symbol, indicators=indicators, curr_date=curr_date, look_back_days=look_back_days,
            )

    @staticmethod
    @tool
    def get_finnhub_company_insider_sentiment(
        ticker: Annotated[str, "ticker symbol for the company"],
        curr_date: Annotated[
            str,
            "current date of you are trading at, yyyy-mm-dd",
        ],
    ):
        """
        Retrieve insider sentiment information about a company (retrieved from public SEC information) for the past 30 days
        Args:
            ticker (str): ticker symbol of the company
            curr_date (str): current date you are trading at, yyyy-mm-dd
        Returns:
            str: a report of the sentiment in the past 30 days starting at curr_date
        """
        try:
            print(f"[TOOLKIT] Fetching Finnhub insider sentiment for {ticker} on {curr_date}")
            data_sentiment = interface.get_finnhub_company_insider_sentiment(
                ticker, curr_date, 30
            )
            print(f"[TOOLKIT] Successfully retrieved Finnhub insider sentiment for {ticker}")
            return data_sentiment
        except Exception as e:
            return handle_tool_error("get_finnhub_company_insider_sentiment", e, ticker=ticker, curr_date=curr_date)

    @staticmethod
    @tool
    def get_finnhub_company_insider_transactions(
        ticker: Annotated[str, "ticker symbol"],
        curr_date: Annotated[
            str,
            "current date you are trading at, yyyy-mm-dd",
        ],
    ):
        """
        Retrieve insider transaction information about a company (retrieved from public SEC information) for the past 30 days
        Args:
            ticker (str): ticker symbol of the company
            curr_date (str): current date you are trading at, yyyy-mm-dd
        Returns:
            str: a report of the company's insider transactions/trading information in the past 30 days
        """
        try:
            print(f"[TOOLKIT] Fetching Finnhub insider transactions for {ticker} on {curr_date}")
            data_trans = interface.get_finnhub_company_insider_transactions(
                ticker, curr_date, 30
            )
            print(f"[TOOLKIT] Successfully retrieved Finnhub insider transactions for {ticker}")
            return data_trans
        except Exception as e:
            return handle_tool_error("get_finnhub_company_insider_transactions", e, ticker=ticker, curr_date=curr_date)

    @staticmethod
    @tool
    def get_simfin_balance_sheet(
        ticker: Annotated[str, "ticker symbol"],
        freq: Annotated[
            str,
            "reporting frequency of the company's financial history: annual/quarterly",
        ],
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    ):
        """
        Retrieve the most recent balance sheet of a company
        Args:
            ticker (str): ticker symbol of the company
            freq (str): reporting frequency of the company's financial history: annual / quarterly
            curr_date (str): current date you are trading at, yyyy-mm-dd
        Returns:
            str: a report of the company's most recent balance sheet
        """
        try:
            print(f"[TOOLKIT] Fetching SimFin balance sheet for {ticker}, frequency: {freq}, date: {curr_date}")
            data_balance_sheet = interface.get_simfin_balance_sheet(ticker, freq, curr_date)
            print(f"[TOOLKIT] Successfully retrieved SimFin balance sheet for {ticker}")
            return data_balance_sheet
        except Exception as e:
            return handle_tool_error("get_simfin_balance_sheet", e, ticker=ticker, freq=freq, curr_date=curr_date)

    @staticmethod
    @tool
    def get_simfin_cashflow(
        ticker: Annotated[str, "ticker symbol"],
        freq: Annotated[
            str,
            "reporting frequency of the company's financial history: annual/quarterly",
        ],
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    ):
        """
        Retrieve the most recent cash flow statement of a company
        Args:
            ticker (str): ticker symbol of the company
            freq (str): reporting frequency of the company's financial history: annual / quarterly
            curr_date (str): current date you are trading at, yyyy-mm-dd
        Returns:
                str: a report of the company's most recent cash flow statement
        """
        try:
            print(f"[TOOLKIT] Fetching SimFin cash flow for {ticker}, frequency: {freq}, date: {curr_date}")
            data_cashflow = interface.get_simfin_cashflow(ticker, freq, curr_date)
            print(f"[TOOLKIT] Successfully retrieved SimFin cash flow for {ticker}")
            return data_cashflow
        except Exception as e:
            return handle_tool_error("get_simfin_cashflow", e, ticker=ticker, freq=freq, curr_date=curr_date)

    @staticmethod
    @tool
    def get_simfin_income_stmt(
        ticker: Annotated[str, "ticker symbol"],
        freq: Annotated[
            str,
            "reporting frequency of the company's financial history: annual/quarterly",
        ],
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    ):
        """
        Retrieve the most recent income statement of a company
        Args:
            ticker (str): ticker symbol of the company
            freq (str): reporting frequency of the company's financial history: annual / quarterly
            curr_date (str): current date you are trading at, yyyy-mm-dd
        Returns:
                str: a report of the company's most recent income statement
        """
        try:
            print(f"[TOOLKIT] Fetching SimFin income statement for {ticker}, frequency: {freq}, date: {curr_date}")
            data_income_stmt = interface.get_simfin_income_statements(
                ticker, freq, curr_date
            )
            print(f"[TOOLKIT] Successfully retrieved SimFin income statement for {ticker}")
            return data_income_stmt
        except Exception as e:
            return handle_tool_error("get_simfin_income_stmt", e, ticker=ticker, freq=freq, curr_date=curr_date)

    @staticmethod
    @tool
    def get_google_news(
        query: Annotated[str, "Query to search with"],
        curr_date: Annotated[str, "Curr date in yyyy-mm-dd format"],
    ):
        """
        Retrieve the latest news from Google News based on a query and date range.
        Args:
            query (str): Query to search with
            curr_date (str): Current date in yyyy-mm-dd format
            look_back_days (int): How many days to look back
        Returns:
            str: A formatted string containing the latest news from Google News based on the query and date range.
        """
        try:
            print(f"[TOOLKIT] Fetching Google News for query: '{query}' on {curr_date}")
            google_news_results = interface.get_google_news(query, curr_date, 7)
            print(f"[TOOLKIT] Successfully retrieved Google News for query: '{query}'")
            return google_news_results
        except Exception as e:
            return handle_tool_error("get_google_news", e, query=query, curr_date=curr_date)

    @staticmethod
    @tool
    def get_stock_news_openai(
        ticker: Annotated[str, "the company's ticker"],
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    ):
        """
        Retrieve the latest news about a given stock by using OpenAI's news API.
        Args:
            ticker (str): Ticker of a company. e.g. AAPL, TSM
            curr_date (str): Current date in yyyy-mm-dd format
        Returns:
            str: A formatted string containing the latest news about the company on the given date.
        """
        try:
            print(f"[TOOLKIT] Fetching OpenAI stock news for {ticker} on {curr_date}")
            openai_news_results = interface.get_stock_news_openai(ticker, curr_date)
            print(f"[TOOLKIT] Successfully retrieved OpenAI stock news for {ticker}")
            return openai_news_results
        except Exception as e:
            return handle_tool_error("get_stock_news_openai", e, ticker=ticker, curr_date=curr_date)

    @staticmethod
    @tool
    def get_global_news_openai(
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    ):
        """
        Retrieve the latest macroeconomics news on a given date using OpenAI's macroeconomics news API.
        Args:
            curr_date (str): Current date in yyyy-mm-dd format
        Returns:
            str: A formatted string containing the latest macroeconomic news on the given date.
        """
        try:
            print(f"[TOOLKIT] Fetching OpenAI global news for {curr_date}")
            openai_news_results = interface.get_global_news_openai(curr_date)
            print(f"[TOOLKIT] Successfully retrieved OpenAI global news")
            return openai_news_results
        except Exception as e:
            return handle_tool_error("get_global_news_openai", e, curr_date=curr_date)

    @staticmethod
    @tool
    def get_fundamentals_openai(
        ticker: Annotated[str, "the company's ticker"],
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    ):
        """
        Retrieve the latest fundamental information about a given stock on a given date by using OpenAI's news API.
        Args:
            ticker (str): Ticker of a company. e.g. AAPL, TSM
            curr_date (str): Current date in yyyy-mm-dd format
        Returns:
            str: A formatted string containing the latest fundamental information about the company on the given date.
        """
        try:
            print(f"[TOOLKIT] Fetching OpenAI fundamentals for {ticker} on {curr_date}")
            openai_fundamentals_results = interface.get_fundamentals_openai(
                ticker, curr_date
            )
            print(f"[TOOLKIT] Successfully retrieved OpenAI fundamentals for {ticker}")
            return openai_fundamentals_results
        except Exception as e:
            return handle_tool_error("get_fundamentals_openai", e, ticker=ticker, curr_date=curr_date)