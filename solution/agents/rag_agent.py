"""
RAG Agent - Handles queries about historical 10-K financial documents.

This agent uses the retriever to find relevant chunks and Groq to generate answers.
Handles both Problem 1 (single company/year) and Problem 2 (multi-company/year).
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system.retriever import SimpleRetriever
from utils.groq_helper import GroqHelper
from typing import Dict, List
from config import GROQ_MODEL


class RAGAgent:
    """
    Agent for answering questions using RAG (Retrieval-Augmented Generation).

    This combines:
    1. Retrieval: Find relevant chunks from 10-K filings
    2. Generation: Use Groq LLM to answer based on retrieved context
    """

    # Year mapping: Fiscal year → Filing year
    # The 10-K filed in year X contains fiscal data for year X-1
    FISCAL_TO_FILING_YEAR = {
        2019: 2020,
        2020: 2021,
        2021: 2022
    }

    def __init__(self, retriever: SimpleRetriever):
        """
        Initialize the RAG agent.

        Args:
            retriever: Configured SimpleRetriever with all chunks loaded
        """
        self.retriever = retriever
        self.groq = GroqHelper()

    def _map_fiscal_to_filing_year(self, fiscal_year: int) -> int:
        """
        Convert a fiscal year to the corresponding filing year.

        IMPORTANT: The 10-K filed in year X contains fiscal year X-1 data.
        Example: 2021 filing contains 2020 fiscal data.

        Args:
            fiscal_year: The fiscal year you want (e.g., 2020)

        Returns:
            The filing year to search (e.g., 2021)
        """
        if fiscal_year in self.FISCAL_TO_FILING_YEAR:
            return self.FISCAL_TO_FILING_YEAR[fiscal_year]
        return fiscal_year  # Return as-is if not in mapping
    
    def extract_query_metadata(self, query: str) -> Dict:
        """
        Extract company name and year from a natural language query using LLM.
        
        Args:
            query: The natural language query (e.g., "What were Uber's liabilities in 2020?")
            
        Returns:
            Dictionary with 'company' and 'year' keys (None if not found)
            
        Example:
            metadata = agent.extract_query_metadata("What are Amazon's net sales in 2021?")
            # Returns: {'company': 'AMZN', 'year': 2021}
        """
        extraction_prompt = f"""Extract the company name and year from the following query.

Available companies: Amazon (AMZN), Uber (UBER)
Available years: 2019, 2020, 2021, 2022

Query: "{query}"

Return ONLY a JSON object in this exact format:
{{"company": "AMZN or UBER or null", "year": 2020 or null}}

If the company is mentioned as "Amazon", convert to "AMZN".
If the company is mentioned as "Uber", convert to "UBER".
If not found or ambiguous, use null.

JSON:"""

        try:
            import json
            
            # Use GroqHelper's client attribute
            if not self.groq.client:
                return {'company': None, 'year': None}
            
            response = self.groq.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0
            )
            
            # Parse the JSON response
            content = response.choices[0].message.content.strip()
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            
            metadata = json.loads(content)
            
            # Convert None strings to actual None
            if metadata.get('company') == 'null' or not metadata.get('company'):
                metadata['company'] = None
            if metadata.get('year') == 'null' or not metadata.get('year'):
                metadata['year'] = None
            
            return metadata
            
        except Exception as e:
            print(f"⚠️ Could not extract metadata: {e}")
            return {'company': None, 'year': None}
    
    def answer(
        self,
        query: str,
        company: str = None,
        year: int = None,
        auto_extract: bool = True
    ) -> Dict:
        """
        Answer a query using RAG with automatic company/year extraction.
        
        Args:
            query: The question to answer (can be natural language)
            company: Filter by company (AMZN or UBER) - optional, auto-extracts if None
            year: Filter by year (2019/2020/2021/2022) - optional, auto-extracts if None
            auto_extract: Whether to automatically extract company/year from query (default: True)
            
        Returns:
            Dictionary with answer, sources, and metadata
            
        Example:
            agent = RAGAgent(retriever)
            
            # Method 1: Natural language (auto-extraction)
            result = agent.answer("What were Uber's total liabilities in 2020?")
            
            # Method 2: Manual specification (backward compatible)
            result = agent.answer(
                "What are the total liabilities?",
                company="UBER",
                year=2020
            )
            
            print(result['answer'])
            print(f"Sources: {result['sources']}")
        """
        # Auto-extract company and year if not provided
        if auto_extract and (company is None or year is None):
            print(f"🔍 Analyzing query: {query}")
            metadata = self.extract_query_metadata(query)
            
            # Use extracted values if not manually specified
            if company is None:
                company = metadata.get('company')
            if year is None:
                year = metadata.get('year')
            
            if company or year:
                print(f"✨ Auto-extracted:")
                if company:
                    print(f"   Company: {company}")
                if year:
                    print(f"   Fiscal Year: {year}")

                    # IMPORTANT: Map fiscal year to filing year
                    filing_year = self._map_fiscal_to_filing_year(year)
                    if filing_year != year:
                        print(f"   → Searching in {filing_year} 10-K (which contains {year} fiscal data)")
                        year = filing_year

        # Apply year mapping only for manually specified years (not auto-extracted)
        if not auto_extract and year and year in self.FISCAL_TO_FILING_YEAR:
            # User manually specified a fiscal year that needs mapping
            original_year = year
            year = self._map_fiscal_to_filing_year(year)
            if year != original_year:
                print(f"📅 Mapping fiscal year {original_year} → filing year {year}")
                print(f"   (The {year} 10-K contains {original_year} fiscal data)\n")

        # Validate that we have company and year
        if not company or not year:
            return {
                'answer': f"⚠️ Could not identify company or year from query. Please specify explicitly or rephrase your query to include the company name (Amazon/Uber) and year (2019-2022).\n\nOriginal query: '{query}'",
                'sources': [],
                'query': query,
                'company': company,
                'year': year,
                'auto_extracted': auto_extract
            }
        
        print(f"🔍 Searching for: {query}")
        if company:
            print(f"   Company: {company}")
        if year:
            print(f"   Year: {year}")

        # Step 1: Retrieve relevant chunks
        # Use top_k=10 for complex queries to get more context
        results = self.retriever.search(query, company=company, year=year, top_k=10)

        # Check if we found any results
        if not results:
            # Try broader search without year filter
            if year:
                print(f"⚠️ No results with year filter. Trying broader search...")
                results = self.retriever.search(query, company=company, year=None, top_k=10)

                if not results and company:
                    # Try even broader - no filters
                    print(f"⚠️ Still no results. Searching across all documents...")
                    results = self.retriever.search(query, company=None, year=None, top_k=15)

        if not results:
            return {
                'answer': f"I couldn't find any relevant information for this query in the {company if company else ''} {year if year else ''} 10-K filings.",
                'sources': [],
                'query': query,
                'company': company,
                'year': year
            }

        # Step 1.5: Check confidence of top result
        top_similarity = results[0].get('similarity', 0)

        # Use hybrid reranking for better results on complex queries
        # This combines semantic similarity with keyword matching
        if hasattr(self.retriever, 'rerank_results'):
            original_top = top_similarity
            results = self.retriever.rerank_results(query, results, method="hybrid")
            new_top = results[0].get('rerank_score', top_similarity)
            if new_top > original_top:
                print(f"🔄 Hybrid reranking improved results (sim: {original_top:.3f} → {new_top:.3f})")

        if top_similarity < 0.4:
            print(f"⚠️ Low confidence search (similarity: {top_similarity:.3f})")
            print(f"   Top result may not be relevant. Consider rephrasing your query.")
        elif top_similarity < 0.5:
            print(f"⚠️ Medium confidence search (similarity: {top_similarity:.3f})")
        else:
            print(f"✅ High confidence search (similarity: {top_similarity:.3f})")
        
        # Step 2: Build context from retrieved chunks
        context_parts = []
        sources = []
        
        for i, result in enumerate(results):
            # Add chunk text to context
            context_parts.append(f"[Source {i+1}]\n{result['text']}\n")
            
            # Track source for citation
            sources.append({
                'company': result['company'],
                'year': result['year'],
                'page': result['page'],
                'source_file': result['source_file'],
                'similarity': result['similarity']
            })
        
        context = "\n".join(context_parts)
        
        # Step 3: Generate answer using Groq
        print("🤖 Generating answer...")

        # Enhanced prompt with few-shot examples for better extraction
        system_prompt = """You are a financial analyst extracting precise information from SEC 10-K filings.

CRITICAL RULES:
1. The context provided IS from the correct company and year - TRUST the metadata
2. Extract exact numbers, figures, and data points from the context
3. Look for financial metrics like: revenue, sales, liabilities, assets, employees, etc.
4. Report numbers with their units (million, billion, thousands)
5. Use dollar signs ($) for monetary values

FEW-SHOT EXAMPLES:

Example 1:
Context: "Total revenue $11,139 million for the fiscal year ended December 31, 2020"
Query: "What was the revenue in 2020?"
Answer: "According to the 10-K filing, the total revenue for 2020 was $11,139 million."

Example 2:
Context: "As of December 31, 2021, we had approximately 22,800 full-time employees globally."
Query: "How many employees in 2021?"
Answer: "As of December 31, 2021, the company had approximately 22,800 full-time employees globally."

Example 3:
Context: "Total liabilities were $23,988 million and $31,761 million as of December 31, 2020 and 2019, respectively."
Query: "What were total liabilities in 2020?"
Answer: "Total liabilities were $23,988 million as of December 31, 2020."

Example 4:
Context: "We recognized 34%, 31%, and 31% of our annual revenue during the fourth quarter of 2017, 2018, and 2019."
Query: "What percentage of annual revenue was recognized in Q4 2019?"
Answer: "According to the 10-K filing, 31% of annual revenue was recognized in the fourth quarter of 2019."

Example 5:
Context: "The number of employees does not appear in this section."
Query: "How many employees were there?"
Answer: "I could not find employee information in the provided context sections."

INSTRUCTIONS FOR YOUR ANSWER:
1. Answer based ONLY on the context below
2. Read through ALL the context sections carefully - the answer may be in any section
3. If you find the exact number, quote it directly
4. Include the unit (million, billion, thousands, percentage, etc.)
5. Be specific about the fiscal year if mentioned
6. For percentage queries, look for sentences with "%" or "percent"
7. If the context doesn't contain the answer, clearly state that
8. Keep answers concise but complete
"""
        
        answer = self.groq.ask(
            question=query,
            context=context,
            system_prompt=system_prompt
        )
        
        return {
            'answer': answer,
            'sources': sources,
            'query': query,
            'company': company,
            'year': year,
            'context_used': len(results)
        }
    
    def answer_comparative(
        self,
        sub_queries: List[Dict]
    ) -> Dict:
        """
        Answer a comparative query by combining multiple sub-queries.
        
        This is for Problem 2 (multi-company/multi-year).
        
        Args:
            sub_queries: List of dicts with 'query', 'company', 'year'
            
        Returns:
            Dictionary with comparative answer and all sources
            
        Example:
            sub_queries = [
                {'query': 'total revenue', 'company': 'AMZN', 'year': 2019},
                {'query': 'total revenue', 'company': 'AMZN', 'year': 2021}
            ]
            result = agent.answer_comparative(sub_queries)
        """
        print(f"🔍 Answering comparative query with {len(sub_queries)} sub-queries")
        
        # Answer each sub-query
        all_results = []
        all_sources = []
        
        for sq in sub_queries:
            result = self.answer(
                sq['query'],
                company=sq.get('company'),
                year=sq.get('year')
            )
            all_results.append(result)
            all_sources.extend(result['sources'])
        
        # Combine results into a comparative answer
        combined_context = []
        for i, result in enumerate(all_results):
            sq = sub_queries[i]
            combined_context.append(
                f"For {sq.get('company', 'the company')} in {sq.get('year', 'that year')}:\n{result['answer']}\n"
            )
        
        context = "\n".join(combined_context)
        
        # Generate final comparative answer
        original_query = " | ".join([sq['query'] for sq in sub_queries])

        # Enhanced comparative prompt with examples
        system_prompt = """You are a financial analyst creating comparative analyses from SEC filings.

COMPARATIVE ANALYSIS EXAMPLES:

Example 1:
Context A: "Amazon net sales were $280.5 billion in 2019"
Context B: "Amazon net sales were $469.8 billion in 2021"
Query: "Compare Amazon's net sales in 2019 vs 2021"
Answer: "Amazon's net sales grew significantly from $280.5 billion in 2019 to $469.8 billion in 2021, representing an increase of $189.3 billion (approximately 67.5% growth)."

Example 2:
Context A: "Uber 2021 risks include regulatory challenges and competition"
Context B: "Amazon 2020 risks include international operations and inventory"
Query: "Compare risk factors"
Answer: "Uber's 2021 risk factors focus on regulatory challenges and competitive pressures, while Amazon's 2020 risks emphasize international operations and inventory management. Both companies face market and competitive risks."

INSTRUCTIONS:
1. Synthesize the provided information into a coherent comparison
2. Highlight key differences and similarities
3. Be specific with numbers and calculate percentages when possible
4. Organize your answer clearly with comparisons side-by-side
5. Quote the exact figures from the context
"""
        
        final_answer = self.groq.ask(
            question=f"Synthesize this information: {original_query}",
            context=context,
            system_prompt=system_prompt
        )
        
        return {
            'answer': final_answer,
            'sub_results': all_results,
            'sources': all_sources,
            'query': original_query
        }


if __name__ == "__main__":
    # Test the RAG agent
    from rag_system.pdf_loader import load_pdfs
    from rag_system.chunking import chunk_documents
    
    print("Testing RAG Agent...")
    
    # Load and chunk documents
    docs = load_pdfs("../Assignment/10-k_docs")
    chunks = chunk_documents(docs)
    
    # Create retriever and agent
    retriever = SimpleRetriever(chunks)
    agent = RAGAgent(retriever)
    
    # Test single query
    print("\n" + "="*60)
    print("Test 1: Single query")
    print("="*60)
    result = agent.answer(
        "What are the total liabilities?",
        company="UBER",
        year=2020
    )
    
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources used: {result['context_used']}")
    for source in result['sources'][:2]:
        print(f"  - {source['company']} {source['year']}, Page {source['page']}")
