"""
API Agent - Handles queries about real-time stock prices.

This agent uses the RapidAPI client to fetch current stock data.
Handles Problem 3: real-time stock market information.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.api_client import StockAPIClient
from utils.groq_helper import GroqHelper
from typing import Dict, List


class APIAgent:
    """
    Agent for answering questions using real-time stock API data.
    
    This fetches current stock prices and market data from RapidAPI.
    """
    
    def __init__(self):
        """Initialize the API agent."""
        self.stock_client = StockAPIClient()
        self.groq = GroqHelper()
    
    def answer(self, query: str) -> Dict:
        """
        Answer a query about real-time stock data.
        
        Args:
            query: The question (e.g., "What is Amazon's current price?")
            
        Returns:
            Dictionary with answer and data
            
        Example:
            agent = APIAgent()
            result = agent.answer("What is the current price of AMZN?")
            print(result['answer'])
        """
        print(f"📈 Fetching real-time stock data for query: {query}")
        
        # Extract stock symbols from query using Groq
        symbols = self._extract_symbols(query)
        
        if not symbols:
            return {
                'answer': "I couldn't identify which stock(s) you're asking about. Please mention AMZN (Amazon) or UBER.",
                'data': None,
                'query': query
            }
        
        print(f"   Identified symbols: {symbols}")
        
        # Determine if this is a historical or current price query
        if "last" in query.lower() or "days" in query.lower() or "week" in query.lower():
            return self._answer_historical(query, symbols)
        else:
            return self._answer_current(query, symbols)
    
    def _extract_symbols(self, query: str) -> List[str]:
        """
        Extract stock symbols from query.
        
        Args:
            query: The user's question
            
        Returns:
            List of stock symbols (AMZN, UBER)
        """
        symbols = []
        
        # Check for common mentions
        query_lower = query.lower()
        
        if "amazon" in query_lower or "amzn" in query_lower:
            symbols.append("AMZN")
        if "uber" in query_lower:
            symbols.append("UBER")
        
        return symbols
    
    def _answer_current(self, query: str, symbols: List[str]) -> Dict:
        """
        Answer query about current stock prices.
        
        Args:
            query: The question
            symbols: List of stock symbols
            
        Returns:
            Response dictionary
        """
        # Fetch data for all symbols
        all_data = {}
        
        for symbol in symbols:
            data = self.stock_client.get_current_price(symbol)
            if data:
                all_data[symbol] = data
        
        if not all_data:
            return {
                'answer': "Sorry, I couldn't fetch the stock data. Please check your RapidAPI key configuration.",
                'data': None,
                'query': query
            }
        
        # Build context from stock data
        context_parts = []
        for symbol, data in all_data.items():
            revenue_str = f"${data['revenue']:,.0f}" if data['revenue'] else 'N/A'
            revenue_growth = f"{data['revenueGrowth']*100:.2f}%" if data['revenueGrowth'] else 'N/A'
            profit_margin = f"{data['profitMargins']*100:.2f}%" if data['profitMargins'] else 'N/A'
            
            context_parts.append(f"""
{symbol} Stock Data:
- Current Price: ${data['currentPrice']}
- Currency: {data['currency']}
- Revenue: {revenue_str}
- Revenue Growth: {revenue_growth}
- Profit Margin: {profit_margin}
""")
        
        context = "\n".join(context_parts)
        
        # Generate answer
        system_prompt = """You are a financial assistant providing real-time stock information.

Instructions:
1. Answer based on the current stock data provided
2. Format prices clearly with $ and proper formatting
3. Be concise and direct
4. If asked about multiple stocks, compare them
"""
        
        answer = self.groq.ask(
            question=query,
            context=context,
            system_prompt=system_prompt
        )
        
        return {
            'answer': answer,
            'data': all_data,
            'query': query,
            'symbols': symbols
        }
    
    def _answer_historical(self, query: str, symbols: List[str]) -> Dict:
        """
        Answer query about historical prices (last N days).
        
        Args:
            query: The question
            symbols: List of stock symbols
            
        Returns:
            Response dictionary
        """
        # Extract time range from query
        time_range = "7d"  # Default
        if "7" in query or "week" in query.lower():
            time_range = "7d"
        elif "30" in query or "month" in query.lower():
            time_range = "1mo"
        
        # Fetch chart data
        all_data = {}
        
        for symbol in symbols:
            data = self.stock_client.get_chart_data(symbol, range=time_range, interval="1d")
            if data:
                all_data[symbol] = data
        
        if not all_data:
            return {
                'answer': "Sorry, I couldn't fetch the historical stock data.",
                'data': None,
                'query': query
            }
        
        # Build context from chart data
        context_parts = []
        for symbol, data in all_data.items():
            closes = data.get('close', [])
            dates = len(closes)
            
            if closes:
                recent_prices = [f"${p:.2f}" for p in closes[-7:] if p]  # Last 7 data points
                context_parts.append(f"""
{symbol} Historical Prices ({time_range}):
- Number of data points: {dates}
- Recent prices: {', '.join(recent_prices)}
- Current Price: ${data['currentPrice']}
- Previous Close: ${data['previousClose']}
- Day High: ${data['dayHigh']}
- Day Low: ${data['dayLow']}
""")
        
        context = "\n".join(context_parts)
        
        # Generate answer
        system_prompt = """You are a financial assistant providing historical stock price information.

Instructions:
1. Summarize the price trend clearly
2. Mention high, low, and recent values
3. Format prices with $
4. Be specific about the time period
"""
        
        answer = self.groq.ask(
            question=query,
            context=context,
            system_prompt=system_prompt
        )
        
        return {
            'answer': answer,
            'data': all_data,
            'query': query,
            'symbols': symbols,
            'time_range': time_range
        }


if __name__ == "__main__":
    # Test the API agent
    agent = APIAgent()
    
    print("🧪 Testing API Agent...\n")
    
    # Test 1: Current price
    print("="*60)
    print("Test 1: Current stock price")
    print("="*60)
    result = agent.answer("What is the current stock price of Amazon?")
    print(f"\nAnswer: {result['answer']}")
    if result['data']:
        for symbol, data in result['data'].items():
            print(f"\n{symbol}: ${data['currentPrice']}")
    
    # Test 2: Multiple stocks
    print("\n" + "="*60)
    print("Test 2: Multiple stocks")
    print("="*60)
    result = agent.answer("What are the current prices of Amazon and Uber?")
    print(f"\nAnswer: {result['answer']}")
