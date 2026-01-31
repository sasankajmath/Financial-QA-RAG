"""
Retriever - Search for relevant chunks with company/year filtering.

This is the KEY component for Problem 1: ensuring queries for "Amazon 2020"
only search within Amazon's 2020 filing.

HOW IT WORKS:
1. Convert all chunks to embeddings (vectors)
2. Store in FAISS index for fast similarity search
3. When querying: search + filter by metadata
"""

from typing import List, Dict, Optional
import faiss
import numpy as np
from rag_system.embeddings import get_embedding_model, create_embeddings
from rag_system.vector_cache import save_cache, load_cache


class SimpleRetriever:
    """
    Easy-to-understand retriever with metadata filtering and persistent caching.
    
    This class handles the core search functionality for the RAG system.
    Now with caching support for 90%+ speedup on subsequent runs!
    """
    
    def __init__(
        self, 
        chunks: List[Dict], 
        use_cache: bool = True,
        cache_dir: str = "data/faiss_index",
        pdf_directory: str = "Assignment/10-k_docs"
    ):
        """
        Initialize retriever with chunks.
        
        Tries to load from cache first. If cache doesn't exist or is invalid,
        creates embeddings and builds a FAISS index, then saves to cache.
        
        Args:
            chunks: List of chunk dicts from chunking.chunk_documents()
                   Each must have 'text' and metadata (company, year, page)
            use_cache: Whether to use caching (default: True)
            cache_dir: Directory for cache files
            pdf_directory: Directory containing source PDFs (for cache validation)
        """
        print("🔍 Initializing retriever...")
        
        self.chunks = chunks
        self.embedding_model = get_embedding_model()
        
        # Try to load from cache first
        if use_cache:
            cached_data = load_cache(cache_dir, pdf_directory)
            
            if cached_data is not None:
                # Cache hit! Load everything from cache
                self.index = cached_data["index"]
                self.embeddings = cached_data["embeddings"]
                self.chunks = cached_data["chunks"]
                print(f"✅ Retriever ready with {len(self.chunks)} chunks (loaded from cache)")
                return
        
        # Cache miss - build from scratch
        print(f"   Creating embeddings for {len(chunks)} chunks...")
        print("   (This will take a few minutes, but only on first run)")
        
        # Extract just the text from each chunk
        chunk_texts = [chunk['text'] for chunk in chunks]
        
        # Create embeddings for all chunks
        self.embeddings = create_embeddings(chunk_texts)
        
        # Build FAISS index for fast similarity search
        print("   Building FAISS index...")
        dimension = self.embeddings.shape[1]  # Usually 384 for all-MiniLM-L6-v2
        
        # IndexFlatL2 does exact search using L2 distance
        # (other indices can be faster but less accurate)
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"✅ Retriever ready with {len(chunks)} chunks")
        
        # Save to cache for next time
        if use_cache:
            save_cache(
                index=self.index,
                embeddings=self.embeddings,
                chunks=self.chunks,
                cache_dir=cache_dir,
                pdf_directory=pdf_directory
            )
    
    def search(
        self,
        query: str,
        company: Optional[str] = None,
        year: Optional[int] = None,
        top_k: int = 8
    ) -> List[Dict]:
        """
        Search for relevant chunks.
        
        This is where the magic happens! We search by meaning (semantic search)
        and filter by company/year to ensure isolation.
        
        Args:
            query: The question to search for
            company: Filter by company ('AMZN' or 'UBER') - optional
            year: Filter by year (2019, 2020, or 2021) - optional
            top_k: Number of results to return
            
        Returns:
            List of relevant chunks with scores, sorted by relevance
            
        Example:
            # Search only in Uber's 2020 filing
            results = retriever.search(
                "total liabilities",
                company="UBER",
                year=2020,
                top_k=5
            )
        """
        # Convert query to embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search in FAISS index
        # We search for more than top_k because we'll filter by metadata
        search_k = min(top_k * 10, len(self.chunks))
        distances, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
        # Filter results by metadata and collect top_k
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            chunk = self.chunks[idx]
            
            # Apply company filter
            if company and chunk['company'] != company:
                continue
            
            # Apply year filter
            if year and chunk['year'] != year:
                continue
            
            # Add to results with relevance score
            # Lower distance = more similar, so we use negative for sorting
            results.append({
                'text': chunk['text'],
                'company': chunk['company'],
                'year': chunk['year'],
                'page': chunk['page'],
                'source_file': chunk['source_file'],
                'score': float(distance),  # Lower is better
                'similarity': 1 / (1 + float(distance))  # Higher is better
            })
            
            # Stop when we have enough results
            if len(results) >= top_k:
                break
        
        # If we didn't find enough results, warn the user
        if len(results) < top_k:
            filter_desc = []
            if company:
                filter_desc.append(f"company={company}")
            if year:
                filter_desc.append(f"year={year}")
            filter_str = ", ".join(filter_desc) if filter_desc else "no filters"
            print(f"⚠️ Only found {len(results)} chunks (requested {top_k}) with {filter_str}")
        
        return results
    
    def search_multiple(
        self,
        queries: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """
        Search for multiple queries at once.
        
        Useful for Problem 2 (multi-company/multi-year queries).
        
        Args:
            queries: List of dicts with 'query', 'company', 'year' keys
            
        Returns:
            Dictionary mapping query text to results
            
        Example:
            queries = [
                {'query': 'net sales', 'company': 'AMZN', 'year': 2019},
                {'query': 'net sales', 'company': 'AMZN', 'year': 2021}
            ]
            results = retriever.search_multiple(queries)
        """
        all_results = {}
        
        for q in queries:
            results = self.search(
                query=q['query'],
                company=q.get('company'),
                year=q.get('year'),
                top_k=q.get('top_k', 5)
            )
            all_results[q['query']] = results
        
        return all_results

    def rerank_results(
        self,
        query: str,
        results: List[Dict],
        method: str = "similarity"
    ) -> List[Dict]:
        """
        Re-rank search results for better relevance.

        This can improve retrieval quality by re-scoring results based on
        more sophisticated metrics.

        Args:
            query: The original search query
            results: Initial search results from self.search()
            method: "similarity", "keyword", or "hybrid"

        Returns:
            Re-ranked list of results (same structure as input)
        """
        if not results:
            return results

        if method == "similarity":
            # Already sorted by similarity, just ensure descending order
            return sorted(results, key=lambda x: x.get('similarity', 0), reverse=True)

        elif method == "keyword":
            # Boost results with exact keyword matches
            query_lower = query.lower()
            query_words = set(query_lower.split())

            for result in results:
                text_lower = result['text'].lower()
                # Count matching words
                word_matches = sum(1 for word in query_words if word in text_lower)
                # Boost score based on keyword matches
                result['rerank_score'] = result.get('similarity', 0) + (word_matches * 0.1)

            return sorted(results, key=lambda x: x.get('rerank_score', 0), reverse=True)

        elif method == "hybrid":
            # Combine semantic similarity with keyword matching
            query_lower = query.lower()
            query_words = set(query_lower.split())

            for result in results:
                text_lower = result['text'].lower()
                word_matches = sum(1 for word in query_words if word in text_lower)
                # Weighted combination: 70% semantic, 30% keyword
                result['rerank_score'] = (
                    result.get('similarity', 0) * 0.7 +
                    (word_matches / len(query_words)) * 0.3
                )

            return sorted(results, key=lambda x: x.get('rerank_score', 0), reverse=True)

        return results


if __name__ == "__main__":
    # Test the retriever
    from pdf_loader import load_pdfs
    from chunking import chunk_documents
    
    print("Testing retriever...")
    
    # Load and chunk documents
    docs = load_pdfs("../Assignment/10-k_docs")
    chunks = chunk_documents(docs, chunk_size=1000, overlap=200)
    
    # Create retriever
    retriever = SimpleRetriever(chunks)
    
    # Test search with filters
    print("\n🔍 Test 1: Search for 'total revenue' in AMZN 2020")
    results = retriever.search("total revenue", company="AMZN", year=2020, top_k=3)
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Company: {result['company']}, Year: {result['year']}, Page: {result['page']}")
        print(f"  Score: {result['score']:.4f}")
        print(f"  Text preview: {result['text'][:200]}...")
    
    # Verify all results are from AMZN 2020
    assert all(r['company'] == 'AMZN' for r in results), "Found non-AMZN results!"
    assert all(r['year'] == 2020 for r in results), "Found non-2020 results!"
    print("\n✅ Filtering works correctly!")
