# Financial Q&A System - Comprehensive Test Questions

This document contains robust test questions for each of the three problems in the Financial Q&A system. These questions are designed to thoroughly test the system's capabilities.

---

## Problem 1: RAG System with Company/Year Isolation

**Purpose:** Test the RAG system's ability to retrieve accurate information from specific company 10-K filings for specific years.

### Question 1.1: Specific Financial Metric Extraction
**Query:** "What were the total liabilities reported by Uber in their 2020 10-K filing?"

**Expected Behavior:**
- Should retrieve only from Uber 2020 documents
- Should not mix data from Amazon or other years
- Should provide specific dollar amounts with proper citations
- Answer should include source page references

### Question 1.2: Operational Data Retrieval
**Query:** "How many full-time employees did Amazon report in their 2019 10-K?"

**Expected Behavior:**
- Should isolate Amazon 2019 filing
- Should extract the specific employee count
- Should distinguish between full-time, part-time, and seasonal workers if mentioned
- Should cite the relevant section of the 10-K

### Question 1.3: Revenue Breakdown Query
**Query:** "What were Amazon's net product sales versus net service sales in 2021?"

**Expected Behavior:**
- Should retrieve only from Amazon 2021 filing
- Should provide breakdown of product vs. service revenue
- Should handle multi-part questions within same document
- Should provide dollar amounts for both categories

### Question 1.4: Risk Factors Identification
**Query:** "What were the primary risk factors mentioned in Uber's 2021 10-K related to autonomous vehicles?"

**Expected Behavior:**
- Should search within Uber 2021 risk factors section
- Should identify and summarize relevant risks
- Should not confuse with Amazon's risks or other years
- Should provide context from the document

---

## Problem 2: Multi-Company/Multi-Year Query Decomposition

**Purpose:** Test the system's ability to decompose complex queries that span multiple companies and/or years, then synthesize results.

### Question 2.1: Year-over-Year Comparison (Single Company)
**Query:** "Compare Amazon's net sales growth between 2019 and 2021. What was the percentage increase?"

**Expected Behavior:**
- Should decompose into two sub-queries (Amazon 2019 and Amazon 2021)
- Should extract net sales from both years
- Should calculate percentage growth
- Should synthesize a coherent comparative answer
- Should cite both source documents

### Question 2.2: Cross-Company Comparison (Same Year)
**Query:** "How did the total liabilities of Amazon and Uber compare in 2020? Which company had higher liabilities?"

**Expected Behavior:**
- Should decompose into two sub-queries (Amazon 2020 and Uber 2020)
- Should extract total liabilities from both companies
- Should provide direct comparison with dollar amounts
- Should clearly state which company had higher liabilities
- Should cite both sources

### Question 2.3: Multi-Dimensional Comparison
**Query:** "Summarize and compare the major risk factors mentioned by both Amazon in 2020 and Uber in 2021. What are the common themes?"

**Expected Behavior:**
- Should decompose into two separate retrieval operations
- Should identify risk factors from both documents
- Should compare and contrast the risks
- Should identify common themes (e.g., regulatory, competition, technology)
- Should provide balanced summary from both sources

### Question 2.4: Trend Analysis Across Years
**Query:** "How did Uber's research and development expenses change from 2019 to 2021? Show the trend."

**Expected Behavior:**
- Should query Uber 2019, 2020, and 2021 filings
- Should extract R&D expenses from all three years
- Should present the trend (increasing/decreasing)
- Should calculate year-over-year changes
- Should provide proper citations for all three years

---

## Problem 3: Real-Time Stock Price Retrieval via RapidAPI

**Purpose:** Test the system's ability to fetch current stock market data using the RapidAPI integration.

### Question 3.1: Current Stock Price Query
**Query:** "What is the current stock price of Amazon (AMZN)?"

**Expected Behavior:**
- Should route to API agent (not RAG)
- Should use RapidAPI YFinance endpoint
- Should return current price in USD
- Should include additional context (day's high, low, volume if available)
- Should show timestamp of data

### Question 3.2: Multi-Stock Current Price
**Query:** "What are the current stock prices for both Amazon and Uber?"

**Expected Behavior:**
- Should handle multiple ticker symbols (AMZN and UBER)
- Should fetch both prices from API
- Should present both clearly labeled
- Should allow for comparison if requested
- Response should be well-formatted

### Question 3.3: Historical Price Range Query
**Query:** "What were Uber's stock prices for the last 7 days?"

**Expected Behavior:**
- Should use historical price API endpoint
- Should retrieve 7 days of closing prices
- Should format as a clear time series (date and price)
- Should handle weekends/holidays appropriately
- Should present in chronological order

### Question 3.4: Stock Performance Metrics
**Query:** "What is Amazon's current stock price, market cap, and P/E ratio?"

**Expected Behavior:**
- Should fetch multiple financial metrics from API
- Should handle multi-metric queries
- Should present all requested data points clearly
- Should explain or define terms like P/E ratio if ambiguous
- Should handle cases where some metrics might be unavailable

---

## Bonus: Edge Cases and Robustness Tests

### Edge Case 1: Ambiguous Time Reference
**Query:** "What was Amazon's revenue last year?"

**Expected Behavior:**
- Should recognize ambiguity (which year?)
- Should either ask for clarification or assume most recent year in dataset (2021)
- Should explain assumption if made

### Edge Case 2: Mixed Query Type
**Query:** "What was Amazon's revenue in 2020, and what is their current stock price?"

**Expected Behavior:**
- Should recognize hybrid query (RAG + API)
- Should route to both agents
- Should combine results coherently
- Should clearly distinguish historical (2020) vs current data

### Edge Case 3: No Data Available
**Query:** "What were Amazon's revenues in 2022?"

**Expected Behavior:**
- Should recognize data not available in system
- Should inform user that only 2019-2021 data is available
- Should not make up or hallucinate data
- Should suggest available years

### Edge Case 4: Ticker Symbol Confusion
**Query:** "What is the stock price of Amazon?"

**Expected Behavior:**
- Should infer ticker symbol (AMZN) from company name
- Should handle both common names and ticker symbols
- Should work for "Uber" → "UBER" as well

### Edge Case 5: Company Not in Dataset - Historical Query
**Query:** "What was Google's revenue in 2020?" or "What did Microsoft report in their 2021 10-K about cloud services?"

**Expected Behavior (RAG Query):**
- Should recognize that the query is asking for historical/10-K data
- Should inform user that only Amazon and Uber 10-K filings (2019-2021) are available
- Should NOT attempt to retrieve from non-existent documents
- Should NOT hallucinate or make up data
- Should suggest available companies: "I only have access to 10-K filings for Amazon and Uber from 2019-2021. Would you like information about one of these companies?"

**Example Response:**
```
I don't have access to Google's 10-K filings in my database. 
Currently, I can only answer questions about:
- Amazon (2019, 2020, 2021 10-K filings)
- Uber (2019, 2020, 2021 10-K filings)

Would you like to know about Amazon's or Uber's revenue instead?
```

### Edge Case 6: Company Not in Dataset - Real-Time Query
**Query:** "What is the current stock price of Google?" or "What is Tesla's stock price today?"

**Expected Behavior (API Query):**
- Should recognize this as a real-time price query
- Should route to API agent (not RAG)
- **Should successfully fetch the stock price** via RapidAPI YFinance
- API works for ANY publicly traded company, not just Amazon/Uber
- Should return current price for GOOGL, TSLA, AAPL, or any valid ticker

**Example Response:**
```
The current stock price of Google (GOOGL) is $142.50 as of [timestamp].

Note: I can fetch real-time stock prices for any publicly traded company, 
but I only have detailed 10-K filing information for Amazon and Uber (2019-2021).
```

### Edge Case 7: Company Not in Dataset - Mixed Query
**Query:** "Compare Google's 2020 revenue with its current stock price"

**Expected Behavior:**
- Should recognize this is a hybrid query (historical + real-time)
- Should inform user about RAG data limitations
- Should still provide the stock price via API
- Should explain what data is and isn't available

**Example Response:**
```
I can provide Google's current stock price ($142.50), but I don't have access 
to Google's 2020 10-K filing to retrieve historical revenue data.

My historical data is limited to:
- Amazon 10-K filings (2019-2021)
- Uber 10-K filings (2019-2021)

However, I can fetch real-time stock data for any publicly traded company.
Would you like to compare Amazon's or Uber's historical revenue with current prices?
```

---

## Testing Checklist

- [ ] All Problem 1 questions return correct, isolated data
- [ ] All Problem 2 questions properly decompose and synthesize
- [ ] All Problem 3 questions successfully fetch real-time data
- [ ] System handles edge cases gracefully
- [ ] All responses include proper citations/sources

---

## Notes for Testing

1. **Sequential Testing:** Test Problem 1 questions first to validate basic RAG functionality before testing complex decomposition.

2. **API Rate Limits:** Be mindful of RapidAPI rate limits when testing Problem 3 questions repeatedly.

3. **Citation Verification:** Always verify that citations match the actual source of information.

4. **Performance Metrics:** Track response time for each query to ensure acceptable performance.

5. **Consistency:** Run the same query multiple times to ensure consistent results (especially for RAG queries).
