"""
Groq Helper - Simple wrapper for Groq LLM API.

This file makes it easy to call Groq's LLM without worrying about the details.
Groq is fast and has a generous free tier!
"""

from groq import Groq
import sys
import os

# Add parent directory to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GROQ_API_KEY, GROQ_MODEL, TEMPERATURE, MAX_TOKENS


class GroqHelper:
    """
    Simple wrapper for Groq API.
    Makes it easy to ask questions and get answers.
    """
    
    def __init__(self):
        """Initialize the Groq client."""
        if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
            print("⚠️ WARNING: Groq API key not set!")
            self.client = None
        else:
            self.client = Groq(api_key=GROQ_API_KEY)
    
    def ask(
        self,
        question: str,
        context: str = "",
        system_prompt: str = "You are a helpful financial analyst assistant."
    ) -> str:
        """
        Ask Groq a question.
        
        Args:
            question: The question to ask
            context: Optional context/background information to help answer
            system_prompt: System message that sets the assistant's behavior
            
        Returns:
            Answer from Groq LLM
            
        Example:
            groq = GroqHelper()
            answer = groq.ask(
                "What is the revenue?",
                context="Revenue was $100M in 2020..."
            )
        """
        if not self.client:
            return "Error: Groq API key not configured"
        
        # Build the user message
        if context:
            user_message = f"Context:\n{context}\n\nQuestion: {question}"
        else:
            user_message = question
        
        try:
            # Call Groq API
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            
            # Extract the answer
            answer = response.choices[0].message.content
            return answer
        
        except Exception as e:
            return f"Error calling Groq: {str(e)}"
    
    def classify_intent(self, query: str) -> str:
        """
        Classify whether a query is about historical data (RAG) or real-time data (API).
        
        This is used by the intent router to decide which agent to use.
        
        Args:
            query: The user's question
            
        Returns:
            "RAG" for historical 10-K queries, "API" for real-time stock queries
        """
        system_prompt = """You are an intent classifier for a financial Q&A system.
Classify queries into two categories:

1. "RAG" - Questions about historical financial data from SEC 10-K filings
   Examples:
   - "What was Amazon's revenue in 2020?"
   - "Extract total liabilities from Uber's 2019 10-K"
   - "Compare Amazon's net sales in 2019 vs 2021"
   - Any question about specific fiscal years (2019, 2020, 2021)
   - Questions about: revenue, liabilities, employees, risk factors, etc.

2. "API" - Questions about current/real-time stock market data
   Examples:
   - "What is the current stock price of Amazon?"
   - "Get today's stock price for UBER"
   - "Show me stock prices for the last 7 days"
   - Questions with words like: current, today, latest, now, real-time

Respond with ONLY one word: "RAG" or "API"
"""
        
        response = self.ask(query, system_prompt=system_prompt)
        
        # Clean up response (remove whitespace, make uppercase)
        intent = response.strip().upper()
        
        # Ensure it's either RAG or API
        if "API" in intent:
            return "API"
        else:
            return "RAG"
    
    def decompose_query(self, query: str) -> list:
        """
        Break down a complex query into simpler sub-queries.
        
        Used for Problem 2 (multi-company/multi-year queries).
        
        Args:
            query: Complex query like "Compare Amazon 2019 vs 2021"
            
        Returns:
            List of sub-queries with metadata
        """
        system_prompt = """You are a query decomposer for financial questions.

Break down complex queries into simple sub-queries that can be answered individually.

For each sub-query, identify:
- The question
- The company (AMZN or UBER)
- The year (2019, 2020, or 2021)

Format your response as a Python list of dictionaries:
[
    {"query": "...", "company": "AMZN", "year": 2019},
    {"query": "...", "company": "AMZN", "year": 2021}
]

Examples:
Input: "Compare Amazon's revenue in 2019 vs 2021"
Output: [
    {"query": "What is Amazon's total revenue?", "company": "AMZN", "year": 2019},
    {"query": "What is Amazon's total revenue?", "company": "AMZN", "year": 2021}
]

Input: "Summarize risk factors for Uber 2021 and Amazon 2020"
Output: [
    {"query": "What are the major risk factors?", "company": "UBER", "year": 2021},
    {"query": "What are the major risk factors?", "company": "AMZN", "year": 2020}
]

Respond with ONLY the Python list, no explanation.
"""
        
        response = self.ask(query, system_prompt=system_prompt)
        
        try:
            # Parse the response as Python code
            sub_queries = eval(response)
            return sub_queries
        except:
            # If parsing fails, return a simple query
            return [{"query": query, "company": None, "year": None}]


if __name__ == "__main__":
    # Test the Groq helper
    groq = GroqHelper()
    
    print("🧪 Testing Groq Helper...\n")
    
    # Test 1: Simple question
    print("Test 1: Simple question")
    answer = groq.ask("What is 2 + 2?")
    print(f"Answer: {answer}\n")
    
    # Test 2: Question with context
    print("Test 2: Question with context")
    context = "Amazon's revenue in 2020 was $386 billion."
    question = "What was the revenue?"
    answer = groq.ask(question, context=context)
    print(f"Answer: {answer}\n")
    
    # Test 3: Intent classification
    print("Test 3: Intent classification")
    queries = [
        "What was Amazon's revenue in 2020?",
        "What is the current stock price of UBER?"
    ]
    for q in queries:
        intent = groq.classify_intent(q)
        print(f"Query: {q}")
        print(f"Intent: {intent}\n")
    
    # Test 4: Query decomposition
    print("Test 4: Query decomposition")
    complex_query = "Compare Amazon's revenue in 2019 vs 2021"
    sub_queries = groq.decompose_query(complex_query)
    print(f"Original: {complex_query}")
    print(f"Sub-queries: {sub_queries}")
