# Financial Q&A System - Complete Solution

A multi-agent Retrieval-Augmented Generation (RAG) system for answering financial questions about SEC 10-K filings, with support for both LangGraph-based and traditional orchestration patterns.

## Problem Overview

This solution addresses three core problems:

1. **Problem 1 (RAG System)**: Query historical 10-K filings for specific companies and years
2. **Problem 2 (Query Decomposition)**: Handle multi-company, multi-year comparative queries
3. **Problem 3 (Real-Time Data)**: Fetch current stock prices via RapidAPI integration

## Project Structure

```
solution/
├── agents/                    # Agent implementations
│   ├── rag_agent.py          # RAG agent for 10-K queries (Problem 1 & 2)
│   ├── api_agent.py          # API agent for real-time stock data (Problem 3)
│   └── query_decomposer.py   # Query decomposition logic (Problem 2)
│
├── rag_system/               # RAG infrastructure
│   ├── pdf_loader.py         # PDF document loading
│   ├── chunking.py           # Document chunking strategies
│   ├── embeddings.py         # Text & numerical embedding models
│   ├── retriever.py          # Basic retrieval system
│   ├── multi_retriever.py    # Multi-embedding retrieval with caching
│   └── vector_cache.py       # FAISS vector database caching
│
├── utils/                    # Utility modules
│   └── groq_helper.py        # Groq LLM helper
│
├── config.py                 # Centralized configuration
├── lang_graph_qa.py          # LangGraph-based implementation
├── main.ipynb                # Traditional orchestrator notebook
├── update_notebook_langgraph.py  # LangGraph notebook updater
│
└── data/                     # Data & cache directory
    ├── faiss_index/          # Cached FAISS vector database
    └── chunks.pkl            # Serialized chunk metadata
```

## Two Implementation Approaches

### 1. Traditional Orchestrator (main.ipynb)

A notebook-based implementation using a manual orchestrator pattern:

- **Approach**: Sequential execution with Python-based orchestration
- **Entry Point**: `main.ipynb` Jupyter notebook
- **Characteristics**:
  - Explicit control flow
  - Easier to debug and modify
  - Suitable for interactive development
  - Direct agent instantiation and coordination

**Key Components:**
- `RAGAgent` (agents/rag_agent.py) - Handles 10-K queries
- `APIAgent` (agents/api_agent.py) - Handles real-time stock queries
- `QueryDecomposer` (agents/query_decomposer.py) - Breaks down complex queries
- `SimpleRetriever` / `MultiEmbeddingRetriever` - Document retrieval

### 2. LangGraph-Based (lang_graph_qa.py)

A state machine-based implementation using LangGraph:

- **Approach**: Declarative state graph with agent nodes
- **Entry Point**: `LangGraphFinancialQA` class in lang_graph_qa.py
- **Characteristics**:
  - State-driven architecture
  - Visual graph representation
  - Better for production deployment
  - Easier to extend with new agents
  - Built-in state management

**Key Components:**
- `FinancialQAState` - Typed state definition
- `FinancialQANodes` - Agent node implementations
- `build_financial_qa_graph()` - Graph construction
- `LangGraphFinancialQA` - Main interface class

## Quick Start

### Prerequisites

1. Python 3.9+
2. API keys (see `.env.template`):
   - `GROQ_API_KEY` - Get from https://console.groq.com/
   - `RAPIDAPI_KEY` - Get from https://rapidapi.com/

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.template .env
# Edit .env and add your API keys
```

### Running the Solutions

#### Option A: Traditional Implementation (main.ipynb)

```bash
# Start Jupyter notebook
jupyter notebook main.ipynb
```

#### Option B: LangGraph Implementation

```python
from rag_system.pdf_loader import load_pdfs
from rag_system.chunking import chunk_documents
from rag_system.multi_retriever import MultiEmbeddingRetriever
from agents.api_agent import APIAgent
from lang_graph_qa import LangGraphFinancialQA

# Load data
documents = load_pdfs("Assignment/10-k_docs")
chunks = chunk_documents(documents)

# Create retriever with caching (first run = 2-5 min, subsequent = 5-10 sec)
retriever = MultiEmbeddingRetriever(chunks, use_cache=True)

# Initialize system
qa_system = LangGraphFinancialQA(retriever, APIAgent())

# Query
result = qa_system.route("What was Amazon's revenue in 2019?")
print(result['answer'])
```

# Verify active configuration used in this run
import config
print("Active config:", {
    "CHUNK_SIZE": config.CHUNK_SIZE,
    "CHUNK_OVERLAP": config.CHUNK_OVERLAP,
    "TOP_K": config.TOP_K,
    "USE_MULTI_EMBEDDING": config.USE_MULTI_EMBEDDING
})

## Architecture Comparison

| Aspect | Traditional (main.ipynb) | LangGraph (lang_graph_qa.py) |
|--------|-------------------------|------------------------------|
| **Control Flow** | Manual orchestration | State machine |
| **State Management** | Explicit variables | Typed state dict |
| **Extensibility** | Add agents in notebook | Add nodes to graph |
| **Debugging** | Print statements, step-by-step | Graph visualization |
| **Production Ready** | Requires refactoring | Built for production |
| **Learning Curve** | Easier for beginners | Requires LangGraph knowledge |

## Core Features

### Multi-Embedding System
- **Text Embeddings** (all-MiniLM-L6-v2): For narrative content, risk factors
- **Numerical Embeddings** (all-mpnet-base-v2): For financial metrics, tables
- **Automatic Selection**: Chooses best embedding based on query type

### FAISS Vector Caching
- First run: Encodes all chunks (~2-5 minutes)
- Subsequent runs: Loads cached database (~5-10 seconds)
- Cache location: `data/faiss_index/`

### Fiscal Year Mapping
The system correctly maps fiscal years to filing years:
- Fiscal Year 2019 → Filing Year 2020 (file: AMZN_2020.pdf)
- Fiscal Year 2020 → Filing Year 2021 (file: AMZN_2021.pdf)
- Fiscal Year 2021 → Filing Year 2022 (file: AMZN_2022.pdf)

## Query Examples

### Problem 1: Single Company/Year
```python
# What were Amazon's net sales in 2019?
qa_system.route("What were Amazon's net sales in 2019?")
```

### Problem 2: Multi-Company/Multi-Year
```python
# Compare Amazon's revenue in 2019 vs 2021
qa_system.route("Compare Amazon's net sales in 2019 vs 2021")
```

### Problem 3: Real-Time Stock Price
```python
# What is Amazon's current stock price?
qa_system.route("What is Amazon's current stock price?")
```

## Configuration

Edit `config.py` to customize:

```python
# LLM Settings
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TEMPERATURE = 0.1

# RAG Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 300     # NOTE: Notebook examples / chunk counts were generated using overlap = 300
TOP_K = 5

# Multi-Embedding (recommended: True)
USE_MULTI_EMBEDDING = True

# Caching (recommended: True)
USE_VECTOR_CACHE = True
```

## Testing

Use `test_questions.md` for comprehensive testing:

```bash
# Run test queries from either implementation
# See test_questions.md for full test suite
```

## Performance Notes

- **First Run**: 2-5 minutes (document loading + embedding generation)
- **Subsequent Runs**: 5-10 seconds (loading cached FAISS database)
- **Query Time**: 2-5 seconds per query (retrieval + generation)

## Troubleshooting

### Low Accuracy Answers
- Increase `TOP_K` in config.py
- Try different embedding models
- Check fiscal year mapping in YEAR_MAPPING.md

### Slow Performance
- Enable `USE_VECTOR_CACHE = True`
- Reduce `CHUNK_SIZE` or `TOP_K`
- Use faster embedding model

### API Errors
- Verify API keys in `.env` file
- Check RapidAPI rate limits
- Ensure Groq API quota is available

## References

- [Assignment Problem Statement](../Assignment/Problem_Statement.docx)
- [Test Questions](../test_questions.md)
- [Year Mapping Documentation](YEAR_MAPPING.md)
- [RapidAPI Endpoints](../Assignment/rapidapi_endpoints.pdf)

## License

Educational project for Financial NLP course.
