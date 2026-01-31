"""
Query Decomposer - Breaks complex queries into simpler sub-queries.

Handles Problem 2: Multi-company/multi-year queries.

Example:
"Compare Amazon's revenue in 2019 vs 2021"
→ [
    {"query": "total revenue", "company": "AMZN", "year": 2019},
    {"query": "total revenue", "company": "AMZN", "year": 2021}
  ]
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.groq_helper import GroqHelper
from typing import List, Dict


class QueryDecomposer:
    """
    Decomposes complex queries into simpler sub-queries.
    
    This enables answering comparative questions that span:
    - Multiple years
    - Multiple companies
    - Both
    """
    
    def __init__(self):
        """Initialize the decomposer."""
        self.groq = GroqHelper()
    
    def decompose(self, query: str) -> List[Dict]:
        """
        Break query into sub-queries.
        
        Args:
            query: Complex query to decompose
            
        Returns:
            List of sub-query dicts with 'query', 'company', 'year' keys
            
        Example:
            decomposer = QueryDecomposer()
            subs = decomposer.decompose("Compare Amazon 2019 vs 2021 revenue")
            # Returns:
            # [
            #     {"query": "total revenue", "company": "AMZN", "year": 2019},
            #     {"query": "total revenue", "company": "AMZN", "year": 2021}
            # ]
        """
        print(f"🔧 Decomposing query: {query}")
        
        # Use Groq to decompose the query
        sub_queries = self.groq.decompose_query(query)
        
        print(f"   Generated {len(sub_queries)} sub-queries")
        for sq in sub_queries:
            print(f"   - {sq}")
        
        return sub_queries
    
    def is_complex_query(self, query: str) -> bool:
        """
        Determine if a query needs decomposition.
        
        Args:
            query: The query to check
            
        Returns:
            True if query mentions multiple years/companies, False otherwise
            
        Example:
            is_complex = decomposer.is_complex_query("Compare Amazon 2019 vs 2021")
            # Returns: True
        """
        query_lower = query.lower()
        
        # Check for comparison words
        comparison_words = ["compare", "vs", "versus", "difference", "between"]
        has_comparison = any(word in query_lower for word in comparison_words)
        
        # Check for multiple years
        years = ["2019", "2020", "2021"]
        year_count = sum(1 for year in years if year in query)
        has_multiple_years = year_count > 1
        
        # Check for multiple companies
        companies = ["amazon", "uber", "amzn"]
        company_count = sum(1 for company in companies if company in query_lower)
        has_multiple_companies = company_count > 1
        
        # Complex if has comparison word OR multiple years/companies
        is_complex = has_comparison or has_multiple_years or has_multiple_companies
        
        if is_complex:
            print(f"   Detected complex query (comparison={has_comparison}, "
                  f"multi-year={has_multiple_years}, multi-company={has_multiple_companies})")
        
        return is_complex


if __name__ == "__main__":
    # Test the decomposer
    decomposer = QueryDecomposer()
    
    print("🧪 Testing Query Decomposer...\n")
    
    # Test 1: Simple query (shouldn't decompose)
    print("="*60)
    print("Test 1: Simple query")
    print("="*60)
    query1 = "What is Amazon's revenue in 2020?"
    is_complex = decomposer.is_complex_query(query1)
    print(f"Query: {query1}")
    print(f"Is complex: {is_complex}\n")
    
    # Test 2: Comparison query
    print("="*60)
    print("Test 2: Comparison query")
    print("="*60)
    query2 = "Compare Amazon's net sales in 2019 vs 2021"
    is_complex = decomposer.is_complex_query(query2)
    print(f"Query: {query2}")
    print(f"Is complex: {is_complex}")
    
    if is_complex:
        subs = decomposer.decompose(query2)
        print(f"\nSub-queries:")
        for i, sq in enumerate(subs):
            print(f"  {i+1}. {sq}")
