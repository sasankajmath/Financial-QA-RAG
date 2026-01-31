"""
Embeddings - Convert text into vector representations.

WHY:
- Computers can't directly understand text similarity
- Embeddings convert text to numbers (vectors)
- Similar text gets similar vectors
- This allows us to search semantically (by meaning, not just keywords)

We use sentence-transformers which runs locally (no API needed)
"""

from sentence_transformers import SentenceTransformer
from typing import List, Literal
import numpy as np


# Global variables to cache models (load once, use many times)
# Support for multiple models simultaneously
_embedding_models = {}

# Model type constants
ModelType = Literal["TEXT", "NUMERICAL"]


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Get or initialize the embedding model.
    
    Uses a global cache so models are only loaded once.
    
    Args:
        model_name: Name of the sentence-transformers model to use
        
    Returns:
        SentenceTransformer model ready to create embeddings
    
    Why all-MiniLM-L6-v2?
    - Fast (384 dimensions)
    - Good quality
    - Runs on CPU
    - No API key needed
    """
    global _embedding_models
    
    # If model not in cache, load it
    if model_name not in _embedding_models:
        print(f"🔢 Loading embedding model: {model_name}...")
        _embedding_models[model_name] = SentenceTransformer(model_name)
        print("✅ Embedding model loaded")
    
    return _embedding_models[model_name]


def get_model_by_type(model_type: ModelType = "TEXT") -> SentenceTransformer:
    """
    Get embedding model by type (TEXT or NUMERICAL).
    
    Loads the appropriate model based on type:
    - TEXT: all-MiniLM-L6-v2 (384 dims, fast)
    - NUMERICAL: all-mpnet-base-v2 (768 dims, accurate)
    
    Args:
        model_type: "TEXT" or "NUMERICAL"
        
    Returns:
        SentenceTransformer model
        
    Example:
        text_model = get_model_by_type("TEXT")
        num_model = get_model_by_type("NUMERICAL")
    """
    # Import here to avoid circular dependency
    try:
        from config import TEXT_EMBEDDING_MODEL, NUMERICAL_EMBEDDING_MODEL
    except ImportError:
        # Fallback defaults
        TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        NUMERICAL_EMBEDDING_MODEL = "all-mpnet-base-v2"
    
    if model_type == "TEXT":
        return get_embedding_model(TEXT_EMBEDDING_MODEL)
    elif model_type == "NUMERICAL":
        return get_embedding_model(NUMERICAL_EMBEDDING_MODEL)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'TEXT' or 'NUMERICAL'")


def create_embeddings(
    texts: List[str], 
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32
) -> np.ndarray:
    """
    Convert a list of texts into embeddings.
    
    Args:
        texts: List of text strings to embed
        model_name: Name of embedding model to use
        batch_size: How many texts to process at once (larger = faster but more memory)
        
    Returns:
        NumPy array of shape (n_texts, embedding_dimension)
        Each row is the embedding vector for one text
    
    Example:
        texts = ["Hello world", "Goodbye world"]
        embeddings = create_embeddings(texts)
        print(f"Shape: {embeddings.shape}")  # (2, 384)
    """
    model = get_embedding_model(model_name)
    
    # Convert texts to embeddings
    # show_progress_bar shows a nice progress bar
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    return embeddings


def create_embeddings_by_type(
    texts: List[str],
    model_type: ModelType = "TEXT",
    batch_size: int = 32
) -> np.ndarray:
    """
    Convert texts to embeddings using model type.
    
    Args:
        texts: List of text strings to embed
        model_type: "TEXT" or "NUMERICAL"
        batch_size: Batch size for processing
        
    Returns:
        NumPy array of embeddings
        
    Example:
        # For narrative content
        text_emb = create_embeddings_by_type(texts, "TEXT")
        
        # For financial/numerical content
        num_emb = create_embeddings_by_type(texts, "NUMERICAL")
    """
    model = get_model_by_type(model_type)
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    return embeddings


def get_embedding_dimension(model_name: str = "all-MiniLM-L6-v2") -> int:
    """
    Get the dimension of embeddings from this model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Number of dimensions in the embedding vector
    
    Example:
        dim = get_embedding_dimension()
        print(dim)  # 384
    """
    model = get_embedding_model(model_name)
    
    # Get dimension by encoding a test string
    test_embedding = model.encode(["test"])
    return test_embedding.shape[1]


if __name__ == "__main__":
    # Test embeddings
    print("Testing embeddings...")
    
    # Test with sample texts
    sample_texts = [
        "What is the total revenue?",
        "How much money did the company make?",
        "What is the weather like today?"
    ]
    
    embeddings = create_embeddings(sample_texts)
    
    print(f"\n📊 Embedding Statistics:")
    print(f"Number of texts: {len(sample_texts)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Dimension: {get_embedding_dimension()}")
    
    # Calculate similarity between first two texts
    # (they're similar questions, so similarity should be high)
    from numpy.linalg import norm
    
    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (norm(v1) * norm(v2))
    
    sim_12 = cosine_similarity(embeddings[0], embeddings[1])
    sim_13 = cosine_similarity(embeddings[0], embeddings[2])
    
    print(f"\nSimilarity between text 1 and 2: {sim_12:.3f} (similar questions)")
    print(f"Similarity between text 1 and 3: {sim_13:.3f} (different topics)")
