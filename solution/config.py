"""
Configuration file for the Financial Q&A system.
Loads API keys and sets up all constants in one place.

This file is the central hub for all settings - easy to understand and modify.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================
# API KEYS
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

# ============================================================
# GROQ LLM SETTINGS
# ============================================================

# Which Groq model to use
# Updated to llama-4 (latest supported model as of Jan 2026)
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Temperature: 0.0 = very deterministic, 1.0 = very creative
# We use low temperature for factual financial answers
TEMPERATURE = 0.1

# Maximum tokens in response
MAX_TOKENS = 2000

# ============================================================
# EMBEDDING SETTINGS
# ============================================================

# Local embedding model (no API key needed)
# all-MiniLM-L6-v2 is fast, small, and works well
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ============================================================
# RAG (Retrieval-Augmented Generation) SETTINGS
# ============================================================

# How many characters per chunk when splitting documents
CHUNK_SIZE = 1000

# How much overlap between chunks (helps preserve context)
CHUNK_OVERLAP = 200

# How many relevant chunks to retrieve for each query
TOP_K = 4

# ============================================================
# RAPIDAPI SETTINGS
# ============================================================

# YFinance API host on RapidAPI
RAPIDAPI_HOST = "yahoo-finance166.p.rapidapi.com"

# Default region for stock queries
STOCK_REGION = "US"

# ============================================================
# FILE PATHS
# ============================================================

# Where the PDF files are located
PDF_DIRECTORY = "Assignment/10-k_docs"

# Where to save the FAISS vector database
VECTOR_STORE_PATH = "data/faiss_index"

# ============================================================
# CACHING SETTINGS
# ============================================================

# Enable persistent caching of embeddings and FAISS indexes
# When True: First run takes 2-5 min, subsequent runs take 5-10 sec
# When False: Every run takes 2-5 min (no caching)
USE_VECTOR_CACHE = True

# Cache directory (where FAISS indexes and embeddings are saved)
CACHE_DIRECTORY = "data/faiss_index"

# ============================================================
# MULTI-EMBEDDING SETTINGS
# ============================================================

# Enable multi-embedding system (text + numerical embeddings)
# When True: Uses specialized embeddings for different query types
# When False: Uses only text embedding (original behavior)
USE_MULTI_EMBEDDING = True  # ENHANCED: Enabled for better numerical query accuracy

# Text embedding model for narrative content
# Good for: risk factors, descriptions, qualitative information
TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions, fast

# Numerical/tabular embedding model for financial data
# Good for: revenue, metrics, tables, calculations
NUMERICAL_EMBEDDING_MODEL = "all-mpnet-base-v2"  # 768 dimensions, more accurate

# ============================================================
# VALIDATION
# ============================================================

def validate_config():
    """
    Check if all required API keys are present.
    Raises error if any are missing.
    """
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        print("⚠️ WARNING: GROQ_API_KEY not set in .env file")
        print("   Get your key from: https://console.groq.com/")
        return False
    
    if not RAPIDAPI_KEY or RAPIDAPI_KEY == "your_rapidapi_key_here":
        print("⚠️ WARNING: RAPIDAPI_KEY not set in .env file")
        print("   Get your key from: https://rapidapi.com/")
        return False
    
    print("✅ Configuration loaded successfully!")
    return True

# Print configuration status when imported
if __name__ == "__main__":
    validate_config()
