"""
Multi-Embedding Retriever - Dual FAISS indexes for optimal retrieval.

WHY:
- Different embedding models work better for different content types
- Text embeddings: Better for narrative content (risk factors, strategies)
- Numerical embeddings: Better for financial data (revenue, metrics, tables)

This retriever:
1. Builds TWO separate FAISS indexes (one for each embedding type)
2. Classifies incoming queries as TEXT or NUMERICAL
3. Searches the appropriate index
4. Provides better accuracy for different query types

ARCHITECTURE:
    Query → Classify (TEXT/NUMERICAL) → Search appropriate index → Results
"""

import sys
import os
from typing import List, Dict, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss
import numpy as np
from rag_system.embeddings import create_embeddings_by_type, get_model_by_type
from rag_system.vector_cache import save_cache, load_cache


class MultiEmbeddingRetriever:
    """
    Advanced retriever with dual FAISS indexes for text and numerical queries.
    
    Uses separate embedding models optimized for different content types.
    """
    
    def __init__(
        self,
        chunks: List[Dict],
        use_cache: bool = True,
        cache_dir: str = "data/faiss_index",
        pdf_directory: str = "Assignment/10-k_docs",
        use_llm_classification: bool = False
    ):
        """
        Initialize multi-embedding retriever.
        
        Builds two FAISS indexes:
        - text_index: For narrative content (384 dims)
        - numerical_index: For financial/numerical content (768 dims)
        
        Args:
            chunks: List of chunk dicts from chunking.chunk_documents()
            use_cache: Whether to use persistent caching
            cache_dir: Directory for cache files
            pdf_directory: PDF directory for cache validation
            use_llm_classification: Use LLM for query classification (slower but more accurate)
        """
        print("🔍 Initializing Multi-Embedding Retriever...")

        self.chunks = chunks
        self.use_llm_classification = use_llm_classification
        
        # Try to load from cache first
        if use_cache:
            cached_data = self._load_multi_cache(cache_dir, pdf_directory)
            
            if cached_data is not None:
                # Cache hit!
                self.text_index = cached_data["text_index"]
                self.numerical_index = cached_data["numerical_index"]
                self.text_embeddings = cached_data["text_embeddings"]
                self.numerical_embeddings = cached_data["numerical_embeddings"]
                self.chunks = cached_data["chunks"]
                print(f"✅ Multi-Embedding Retriever ready (loaded from cache)")
                print(f"   Text index: {self.text_index.ntotal} vectors (384 dims)")
                print(f"   Numerical index: {self.numerical_index.ntotal} vectors (768 dims)")
                return
        
        # Cache miss - build from scratch
        print(f"   Building dual embeddings for {len(chunks)} chunks...")
        print("   (This will take longer than single embedding, but only on first run)")
        
        # Extract chunk texts
        chunk_texts = [chunk['text'] for chunk in chunks]
        
        # Create TEXT embeddings (384 dimensions)
        print("\n   📝 Creating TEXT embeddings (all-MiniLM-L6-v2, 384 dims)...")
        self.text_embeddings = create_embeddings_by_type(chunk_texts, "TEXT")
        
        # Create NUMERICAL embeddings (768 dimensions)
        print("\n   🔢 Creating NUMERICAL embeddings (all-mpnet-base-v2, 768 dims)...")
        self.numerical_embeddings = create_embeddings_by_type(chunk_texts, "NUMERICAL")
        
        # Build TEXT FAISS index
        print("\n   Building TEXT FAISS index...")
        text_dimension = self.text_embeddings.shape[1]
        self.text_index = faiss.IndexFlatL2(text_dimension)
        self.text_index.add(self.text_embeddings.astype('float32'))
        
        # Build NUMERICAL FAISS index
        print("   Building NUMERICAL FAISS index...")
        numerical_dimension = self.numerical_embeddings.shape[1]
        self.numerical_index = faiss.IndexFlatL2(numerical_dimension)
        self.numerical_index.add(self.numerical_embeddings.astype('float32'))
        
        print(f"\n✅ Multi-Embedding Retriever ready")
        print(f"   Text index: {self.text_index.ntotal} vectors ({text_dimension} dims)")
        print(f"   Numerical index: {self.numerical_index.ntotal} vectors ({numerical_dimension} dims)")
        
        # Save to cache for next time
        if use_cache:
            self._save_multi_cache(cache_dir, pdf_directory)

    def _classify_query(self, query: str) -> str:
        """
        Classify query as TEXT or NUMERICAL based on keywords.

        Args:
            query: The search query

        Returns:
            "TEXT" or "NUMERICAL"
        """
        numerical_keywords = [
            'revenue', 'sales', 'net sales', 'income', 'net income',
            'profit', 'loss', 'operating income', 'operating loss',
            'gross profit', 'margin', 'gross', 'net', 'operating',
            'quarter', 'quarterly', 'fourth quarter', 'q4',
            'fiscal', 'fiscal year', 'earnings',
            'expenses', 'cost', 'cost of sales', 'cost of revenue',
            'fulfillment', 'shipping', 'marketing',
            'assets', 'total assets', 'liabilities',
            'current liabilities', 'long-term liabilities',
            'long-term debt', 'long-term obligations',
            'equity', 'shareholders equity', 'retained earnings',
            'cash', 'cash equivalents', 'marketable securities',
            'debt', 'capital expenditures', 'free cash flow',
            'employees', 'headcount',
            'growth', 'growth rate', 'year-over-year',
            'percentage', 'increase', 'decrease', 'decline',
            'amount', 'portion', 'total',
            'segment', 'segments', 'north america',
            'international', 'aws',
            'million', 'billion', 'thousand',
            'dollars', 'usd', 'per share',
            '$', '%',
            'market value', 'outstanding shares',
            '2018', '2019', '2020', '2021', '2022'
            ]


        query_lower = query.lower()

        # Count numerical keyword matches
        num_matches = sum(1 for keyword in numerical_keywords if keyword in query_lower)

        # If 2 or more numerical keywords, classify as NUMERICAL
        if num_matches >= 2:
            return "NUMERICAL"
        else:
            return "TEXT"
    
    def search(
        self,
        query: str,
        embedding_type: str = "AUTO",
        company: Optional[str] = None,
        year: Optional[int] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search for relevant chunks using appropriate embedding.
        
        Args:
            query: The question to search for
            embedding_type: "AUTO", "TEXT", or "NUMERICAL"
                           AUTO = classify query automatically (recommended)
            company: Filter by company ('AMZN' or 'UBER') - optional
            year: Filter by year (2019, 2020, or 2021) - optional
            top_k: Number of results to return
            
        Returns:
            List of relevant chunks with scores
            
        Example:
            # Automatic classification (recommended)
            results = retriever.search("What was the revenue?")
            
            # Force specific embedding type
            results = retriever.search("What are risks?", embedding_type="TEXT")
        """
        # Determine which embedding to use
        if embedding_type == "AUTO":
            classified_type = self._classify_query(query)
            print(f"🎯 Query classified as: {classified_type}")
        else:
            classified_type = embedding_type
        
        # Select appropriate index and model
        if classified_type == "TEXT":
            index = self.text_index
            model = get_model_by_type("TEXT")
            index_name = "TEXT (narrative content)"
        else:  # NUMERICAL
            index = self.numerical_index
            model = get_model_by_type("NUMERICAL")
            index_name = "NUMERICAL (financial data)"
        
        print(f"   Searching {index_name} index...")
        
        # Convert query to embedding using appropriate model
        query_embedding = model.encode([query])
        
        # Search in selected index
        search_k = min(top_k * 10, len(self.chunks))
        distances, indices = index.search(query_embedding.astype('float32'), search_k)
        
        # Filter results by metadata
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            chunk = self.chunks[idx]
            
            # Apply company filter
            if company and chunk['company'] != company:
                continue
            
            # Apply year filter
            if year and chunk['year'] != year:
                continue
            
            # Add to results
            results.append({
                'text': chunk['text'],
                'company': chunk['company'],
                'year': chunk['year'],
                'page': chunk['page'],
                'source_file': chunk['source_file'],
                'score': float(distance),
                'similarity': 1 / (1 + float(distance)),
                'embedding_type': classified_type  # Track which index was used
            })
            
            if len(results) >= top_k:
                break
        
        # Warn if not enough results
        if len(results) < top_k:
            filter_desc = []
            if company:
                filter_desc.append(f"company={company}")
            if year:
                filter_desc.append(f"year={year}")
            filter_str = ", ".join(filter_desc) if filter_desc else "no filters"
            print(f"⚠️ Only found {len(results)} chunks (requested {top_k}) with {filter_str}")
        
        return results
    
    def _save_multi_cache(self, cache_dir: str, pdf_directory: str):
        """Save both FAISS indexes and embeddings to cache."""
        import pickle
        import json
        from datetime import datetime
        from pathlib import Path
        
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving multi-embedding cache to {cache_dir}...")
        
        # Save TEXT index and embeddings
        text_index_file = cache_path / "text_index.faiss"
        faiss.write_index(self.text_index, str(text_index_file))
        print(f"   ✓ Saved TEXT FAISS index")
        
        text_emb_file = cache_path / "text_embeddings.npy"
        np.save(text_emb_file, self.text_embeddings)
        print(f"   ✓ Saved TEXT embeddings")
        
        # Save NUMERICAL index and embeddings
        num_index_file = cache_path / "numerical_index.faiss"
        faiss.write_index(self.numerical_index, str(num_index_file))
        print(f"   ✓ Saved NUMERICAL FAISS index")
        
        num_emb_file = cache_path / "numerical_embeddings.npy"
        np.save(num_emb_file, self.numerical_embeddings)
        print(f"   ✓ Saved NUMERICAL embeddings")
        
        # Save chunks
        chunks_file = cache_path / "chunks.pkl"
        with open(chunks_file, "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"   ✓ Saved chunks metadata")
        
        # Save metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "retriever_type": "multi_embedding",
            "text_model": "all-MiniLM-L6-v2",
            "numerical_model": "all-mpnet-base-v2",
            "num_chunks": len(self.chunks),
            "text_dimension": self.text_embeddings.shape[1],
            "numerical_dimension": self.numerical_embeddings.shape[1]
        }
        
        metadata_file = cache_path / "multi_cache_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✓ Saved cache metadata")
        
        # Calculate total size
        total_size = sum([
            text_index_file.stat().st_size,
            text_emb_file.stat().st_size,
            num_index_file.stat().st_size,
            num_emb_file.stat().st_size,
            chunks_file.stat().st_size
        ]) / 1024 / 1024
        
        print(f"\n✅ Multi-embedding cache saved! Total size: {total_size:.1f} MB")
    
    def _load_multi_cache(self, cache_dir: str, pdf_directory: str) -> Optional[Dict]:
        """Load both FAISS indexes and embeddings from cache."""
        import pickle
        import json
        from pathlib import Path
        
        cache_path = Path(cache_dir)
        
        # Check if all required files exist
        required_files = [
            cache_path / "text_index.faiss",
            cache_path / "numerical_index.faiss",
            cache_path / "text_embeddings.npy",
            cache_path / "numerical_embeddings.npy",
            cache_path / "chunks.pkl",
            cache_path / "multi_cache_metadata.json"
        ]
        
        if not all(f.exists() for f in required_files):
            print("📭 No multi-embedding cache found. Building from scratch...")
            return None
        
        print(f"\n📂 Loading multi-embedding cache from {cache_dir}...")
        
        try:
            # Load metadata
            with open(cache_path / "multi_cache_metadata.json", "r") as f:
                metadata = json.load(f)
            
            # Load TEXT index and embeddings
            text_index = faiss.read_index(str(cache_path / "text_index.faiss"))
            text_embeddings = np.load(cache_path / "text_embeddings.npy")
            print(f"   ✓ Loaded TEXT index and embeddings")
            
            # Load NUMERICAL index and embeddings
            numerical_index = faiss.read_index(str(cache_path / "numerical_index.faiss"))
            numerical_embeddings = np.load(cache_path / "numerical_embeddings.npy")
            print(f"   ✓ Loaded NUMERICAL index and embeddings")
            
            # Load chunks
            with open(cache_path / "chunks.pkl", "rb") as f:
                chunks = pickle.load(f)
            print(f"   ✓ Loaded {len(chunks)} chunks")
            
            print(f"\n✅ Multi-embedding cache loaded successfully!")
            print(f"   Created: {metadata['created_at']}")
            
            return {
                "text_index": text_index,
                "numerical_index": numerical_index,
                "text_embeddings": text_embeddings,
                "numerical_embeddings": numerical_embeddings,
                "chunks": chunks,
                "metadata": metadata
            }
            
        except Exception as e:
            print(f"❌ Error loading multi-embedding cache: {e}")
            print("   Will rebuild from scratch...")
            return None


if __name__ == "__main__":
    print("Multi-Embedding Retriever module loaded successfully!")
    print("\nThis module provides:")
    print("  - Dual FAISS indexes (TEXT + NUMERICAL)")
    print("  - Automatic query classification")
    print("  - Optimized retrieval for different content types")
