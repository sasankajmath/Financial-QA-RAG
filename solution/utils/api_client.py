"""
API Client - Wrapper for RapidAPI stock data endpoints.

This handles Problem 3: fetching real-time stock prices using the YFinance API
from RapidAPI.
"""

import requests
import sys
import os
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RAPIDAPI_KEY, RAPIDAPI_HOST, STOCK_REGION


class StockAPIClient:
    """
    Simple client for YFinance API on RapidAPI.
    
    Provides easy methods to get stock prices and financial data.
    """
    
    def __init__(self):
        """Initialize the API client with credentials."""
        self.api_key = RAPIDAPI_KEY
        self.host = RAPIDAPI_HOST
        self.base_url = f"https://{self.host}/api/stock"
        
        # Check if API key is configured
        if not self.api_key or self.api_key == "your_rapidapi_key_here":
            print("⚠️ WARNING: RapidAPI key not configured!")
            self.configured = False
        else:
            self.configured = True
    
    def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """
        Make a request to the RapidAPI endpoint.
        
        Args:
            endpoint: API endpoint (e.g., "/get-financial-data")
            params: Query parameters
            
        Returns:
            JSON response as dictionary, or None if error
        """
        if not self.configured:
            print("❌ Error: RapidAPI key not configured")
            return None
        
        # Build full URL
        url = f"{self.base_url}{endpoint}"
        
        # Set headers
        headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": self.host
        }
        
        try:
            # Make the request
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            # Check for errors
            if response.status_code != 200:
                print(f"❌ API Error: Status {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                return None
            
            # Return JSON data
            return response.json()
        
        except requests.exceptions.Timeout:
            print("❌ Error: Request timed out")
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Error: {str(e)}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """
        Get current stock price and financial data.
        
        Args:
            symbol: Stock ticker (e.g., "AMZN", "UBER")
            
        Returns:
            Dictionary with current price and financial metrics
            
        Example:
            client = StockAPIClient()
            data = client.get_current_price("AMZN")
            print(f"Current price: ${data['currentPrice']}")
        """
        params = {
            "symbol": symbol,
            "region": STOCK_REGION
        }
        
        response = self._make_request("/get-financial-data", params)
        
        if not response:
            return None
        
        # Extract financial data
        try:
            financial_data = response['quoteSummary']['result'][0]['financialData']
            
            # Extract key metrics
            result = {
                'symbol': symbol,
                'currentPrice': financial_data.get('currentPrice', {}).get('raw'),
                'targetHighPrice': financial_data.get('targetHighPrice', {}).get('raw'),
                'targetLowPrice': financial_data.get('targetLowPrice', {}).get('raw'),
                'targetMeanPrice': financial_data.get('targetMeanPrice', {}).get('raw'),
                'revenue': financial_data.get('totalRevenue', {}).get('raw'),
                'revenueGrowth': financial_data.get('revenueGrowth', {}).get('raw'),
                'profitMargins': financial_data.get('profitMargins', {}).get('raw'),
                'currency': financial_data.get('financialCurrency')
            }
            
            return result
        
        except (KeyError, TypeError, IndexError) as e:
            print(f"❌ Error parsing response: {e}")
            return None
    
    def get_chart_data(
        self,
        symbol: str,
        range: str = "1d",
        interval: str = "1d"
    ) -> Optional[Dict]:
        """
        Get historical price chart data.
        
        Args:
            symbol: Stock ticker (e.g., "AMZN", "UBER")
            range: Time range - "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"
            interval: Data interval - "1m", "5m", "1d", etc.
            
        Returns:
            Dictionary with chart data including timestamps and prices
            
        Example:
            # Get last 7 days of daily prices
            data = client.get_chart_data("UBER", range="7d", interval="1d")
        """
        params = {
            "symbol": symbol,
            "region": STOCK_REGION,
            "range": range,
            "interval": interval
        }
        
        response = self._make_request("/get-chart", params)
        
        if not response:
            return None
        
        try:
            chart = response['chart']['result'][0]
            meta = chart['meta']
            
            # Extract price data if available
            timestamps = chart.get('timestamp', [])
            indicators = chart.get('indicators', {}).get('quote', [{}])[0]
            
            result = {
                'symbol': symbol,
                'currency': meta.get('currency'),
                'currentPrice': meta.get('regularMarketPrice'),
                'previousClose': meta.get('previousClose'),
                'dayHigh': meta.get('regularMarketDayHigh'),
                'dayLow': meta.get('regularMarketDayLow'),
                'timestamps': timestamps,
                'open': indicators.get('open', []),
                'high': indicators.get('high', []),
                'low': indicators.get('low', []),
                'close': indicators.get('close', []),
                'volume': indicators.get('volume', [])
            }
            
            return result
        
        except (KeyError, TypeError, IndexError) as e:
            print(f"❌ Error parsing chart data: {e}")
            return None
    
    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, Optional[Dict]]:
        """
        Get current prices for multiple stocks.
        
        Args:
            symbols: List of stock tickers
            
        Returns:
            Dictionary mapping symbol to price data
            
        Example:
            prices = client.get_multiple_prices(["AMZN", "UBER"])
        """
        results = {}
        
        for symbol in symbols:
            print(f"Fetching data for {symbol}...")
            results[symbol] = self.get_current_price(symbol)
        
        return results


if __name__ == "__main__":
    # Test the API client
    client = StockAPIClient()
    
    print("🧪 Testing Stock API Client...\n")
    
    if not client.configured:
        print("⚠️ API key not configured. Set your RAPIDAPI_KEY in .env file to test.")
    else:
        # Test 1: Get current price
        print("Test 1: Get current price for AAPL")
        data = client.get_current_price("AAPL")
        if data:
            print(f"✅ Current price: ${data['currentPrice']}")
            print(f"   Revenue: ${data['revenue']:,.0f}" if data['revenue'] else "")
        
        # Test 2: Get chart data
        print("\nTest 2: Get 1-day chart for AAPL")
        chart = client.get_chart_data("AAPL", range="1d", interval="5m")
        if chart:
            print(f"✅ Day High: ${chart['dayHigh']}")
            print(f"   Day Low: ${chart['dayLow']}")
            print(f"   Data points: {len(chart['timestamps'])}")
