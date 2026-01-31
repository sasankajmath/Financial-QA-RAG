# Traditional Implementation Workflow (main.ipynb)

This document describes the complete workflow for the **Traditional/Notebook-based** Financial Q&A system.

## Overview

The traditional implementation uses a Jupyter notebook (`main.ipynb`) with explicit Python orchestration. Agents are manually coordinated, and flow control is handled through Python code rather than a state machine.

```
┌──────────────────────────────────────────────────────────────────────┐
│                    TRADITIONAL NOTEBOOK WORKFLOW                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  User Query                                                           │
│      │                                                                │
│      ▼                                                                │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  STEP 1: Setup                                                 │  │
│  │  - Import libraries                                            │  │
│  │  - Validate configuration (.env file)                          │  │
│  └────────────────────────────────────────────────────────────────┘  │
│      │                                                                │
│      ▼                                                                │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  STEP 2: Data Loading                                          │  │
│  │  - Load PDF documents (load_pdfs)                              │  │
│  │  - Chunk documents (chunk_documents)                           │  │
│  │  - Create retriever (MultiEmbeddingRetriever)                  │  │
│  └────────────────────────────────────────────────────────────────┘  │
│      │                                                                │
│      ▼                                                                │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  STEP 3: Agent Initialization                                  │  │
│  │  - RAGAgent (for Problems 1 & 2)                               │  │
│  │  - APIAgent (for Problem 3)                                    │  │
│  │  - QueryDecomposer (for Problem 2)                             │  │
│  └────────────────────────────────────────────────────────────────┘  │
│      │                                                                │
│      ▼                                                                │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  STEP 4: Problem Detection (Manual)                            │  │
│  │  - User selects which problem to solve                         │  │
│  │  - Or uses query type detection                                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│      │                                                                │
│      ├──────────────────┬──────────────────┬──────────────────┐       │
│      ▼                  ▼                  ▼                  ▼       │
│  ┌────────┐      ┌────────────┐      ┌────────────┐      ┌────────┐  │
│  │PROBLEM 1│      │ PROBLEM 2  │      │ PROBLEM 3  │      │LANGGRAPH│  │
│  │  RAG    │      │  Complex   │      │    API     │      │SYSTEM   │  │
│  │ Single  │      │ Multi-Co/  │      │ Real-time  │      │Optional │  │
│  │ Co/Year │      │ Year       │      │ Stock      │      │         │  │
│  └────────┘      └────────────┘      └────────────┘      └────────┘  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Workflow

### STEP 1: Setup

**File**: `main.ipynb` - Cells 0-3

**Purpose**: Initialize the notebook environment

```python
# Set working directory
os.chdir(solution_dir)

# Import configuration
from config import validate_config, USE_MULTI_EMBEDDING

# Import RAG system modules
from rag_system.pdf_loader import load_pdfs
from rag_system.chunking import chunk_documents
from rag_system.multi_retriever import MultiEmbeddingRetriever

# Import agents
from agents.rag_agent import RAGAgent
from agents.api_agent import APIAgent
from agents.query_decomposer import QueryDecomposer
```

**Configuration Validation**:
- Checks for `GROQ_API_KEY` in `.env`
- Checks for `RAPIDAPI_KEY` in `.env`
- Displays warnings if keys are missing

---

### STEP 2: Data Loading

**File**: `main.ipynb` - Cells 4-9

#### 2.1 Load PDF Documents

```python
pdf_path = "../Assignment/10-k_docs"
documents = load_pdfs(pdf_path)
```

**Output**:
- 1069 pages from 6 PDF files
- 3 Amazon PDFs (2020, 2021, 2022)
- 3 Uber PDFs (2020, 2021, 2022)

**Document Statistics**:
```
Total pages: 1069
Companies: ['AMZN', 'UBER']
Years: [2020, 2021, 2022]

AMZN:
  2020: 102 pages
  2021: 100 pages
  2022: 96 pages

UBER:
  2020: 239 pages
  2021: 272 pages
  2022: 260 pages
```

#### 2.2 Chunk Documents

```python
chunks = chunk_documents(documents, chunk_size=1000, overlap=300)
```

**Output**:
- 4962 chunks created
- Average chunk length: 836 characters

#### 2.3 Create Retriever

```python
if USE_MULTI_EMBEDDING:
    retriever = MultiEmbeddingRetriever(chunks, use_cache=True)
else:
    retriever = SimpleRetriever(chunks)
```

**Multi-Embedding Retriever**:
- **Text Index**: 4962 vectors (384 dims, MiniLM)
- **Numerical Index**: 4962 vectors (768 dims, mpnet)
- **Cached at**: `data/faiss_index/`

**Cache Loading**:
```
📂 Loading multi-embedding cache from data/faiss_index/
   ✓ Loaded TEXT index and embeddings
   ✓ Loaded NUMERICAL index and embeddings
   ✓ Loaded 4962 chunks
```

---

### STEP 3: Agent Initialization

**File**: `main.ipynb` - Cells 10-16

#### 3.1 RAG Agent

```python
from agents.rag_agent import RAGAgent
rag_agent = RAGAgent(retriever)
```

**Features**:
- Automatic company/year extraction from queries
- Fiscal to filing year mapping
- Multi-embedding support (TEXT/NUMERICAL)
- Answer validation with confidence scoring

**Year Mapping** (Built-in):
```
Fiscal 2019 → Filing 2020 (AMZN_2020.pdf)
Fiscal 2020 → Filing 2021 (AMZN_2021.pdf)
Fiscal 2021 → Filing 2022 (AMZN_2022.pdf)
```

#### 3.2 API Agent

```python
from agents.api_agent import APIAgent
api_agent = APIAgent()
```

**Features**:
- Real-time stock price fetching
- Historical price data (7-day trends)
- Multi-symbol support
- YFinance API via RapidAPI

#### 3.3 Query Decomposer

```python
from agents.query_decomposer import QueryDecomposer
decomposer = QueryDecomposer()
```

**Purpose**: Breaks complex queries into sub-queries

---

### STEP 4: Problem 1 - Single Company/Year Queries

**File**: `main.ipynb` - Cells 17-19

#### Workflow

```
User Query (Natural Language)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│ RAGAgent.answer(query)                               │
│                                                     │
│ 1. Extract company & year (LLM-based)               │
│    - "Amazon" → "AMZN"                              │
│    - "2020" → 2021 (fiscal→filing mapping)          │
│                                                     │
│ 2. Classify query type (TEXT vs NUMERICAL)          │
│    - Numerical keywords: revenue, sales, assets     │
│    - Narrative keywords: risk, description, overview│
│                                                     │
│ 3. Search FAISS database                            │
│    - Filter by company and year                     │
│    - Use appropriate embedding (TEXT/NUMERICAL)     │
│    - Retrieve top_k chunks (default: 10)            │
│                                                     │
│ 4. Generate answer (Groq LLM)                       │
│    - Use retrieved context                          │
│    - Apply few-shot prompting                       │
│    - Validate with confidence scoring               │
│                                                     │
│ 5. Return result                                    │
└─────────────────────────────────────────────────────┘
```

#### Example Usage

```python
# Query: Uber 2020 total revenue
result = rag_agent.answer("What is the total revenue for Uber in 2020?")
```

**Output**:
```
🔍 Analyzing query: What is the total revenue for Uber in 2020?
✨ Auto-extracted:
   Company: UBER
   Fiscal Year: 2020
   → Searching in 2021 10-K (which contains 2020 fiscal data)

🎯 Query classified as: NUMERICAL
🔢 Searching NUMERICAL (financial data) index...
✅ High confidence search (similarity: 0.677)

📝 Answer: According to the 10-K filing, the total revenue for Uber in 2020
           was $11,139 million.

📚 Sources:
  1. UBER 2021, Page 109 (similarity: 0.677)
  2. UBER 2021, Page 100 (similarity: 0.651)
  3. UBER 2021, Page 109 (similarity: 0.642)
```

#### Key Features

1. **Auto-Extraction**:
   - No need to manually specify `company="UBER"` and `year=2020`
   - LLM-based extraction from natural language

2. **Year Mapping**:
   - Automatically maps fiscal years to filing years
   - User asks for "2020 data" → System searches 2021 filing

3. **Query Classification**:
   - Automatically chooses TEXT or NUMERICAL embedding
   - Numerical queries get mpnet embeddings (768 dims)
   - Narrative queries get MiniLM embeddings (384 dims)

4. **Confidence Scoring**:
   - HIGH: similarity > 0.5, contains numbers
   - MEDIUM: 0.4 < similarity < 0.5
   - LOW: similarity < 0.4 or no answer found

---

### STEP 5: Problem 2 - Multi-Company/Multi-Year Queries

**File**: `main.ipynb` - Cells 20-23

#### Workflow

```
Complex Query
      │
      ▼
┌─────────────────────────────────────────────────────┐
│ QueryDecomposer.decompose(query)                     │
│                                                     │
│ 1. Parse query for entities                         │
│    - Extract companies                              │
│    - Extract years                                  │
│    - Identify comparison keywords                   │
│                                                     │
│ 2. Generate sub-queries                             │
│    - One sub-query per (company, year) combination  │
│    - Each with query, company, year                 │
│                                                     │
│ Output: [                                           │
│   {'query': '...', 'company': 'AMZN', 'year': 2019},│
│   {'query': '...', 'company': 'AMZN', 'year': 2021} │
│ ]                                                   │
└─────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────┐
│ RAGAgent.answer_comparative(sub_queries)            │
│                                                     │
│ 1. For each sub-query:                              │
│    - Call RAGAgent.answer()                         │
│    - Collect results and sources                    │
│                                                     │
│ 2. Combine contexts                                 │
│    - Format as "For AMZN in 2019: [answer]"         │
│    - Format as "For AMZN in 2021: [answer]"         │
│                                                     │
│ 3. Generate comparative answer                      │
│    - Use specialized comparison prompt              │
│    - Highlight differences and similarities         │
│    - Calculate percentages/growth when possible     │
│                                                     │
│ 4. Return result                                    │
└─────────────────────────────────────────────────────┘
```

#### Example Usage

```python
# Query: Compare Amazon 2019 vs 2021
complex_query = "Compare Amazon's net sales in 2019 vs 2021"

# Decompose
sub_queries = decomposer.decompose(complex_query)
# Output:
# [
#   {'query': "What are Amazon's net sales?", 'company': 'AMZN', 'year': 2019},
#   {'query': "What are Amazon's net sales?", 'company': 'AMZN', 'year': 2021}
# ]

# Answer
result = rag_agent.answer_comparative(sub_queries)
```

**Output**:
```
🔧 Decomposed into 2 sub-queries:
  1. {'query': "What are Amazon's net sales?", 'company': 'AMZN', 'year': 2019}
  2. {'query': "What are Amazon's net sales?", 'company': 'AMZN', 'year': 2021}

🔍 Answering comparative query with 2 sub-queries...

📝 Comparative Answer:

## Comparative Analysis of Amazon's Net Sales

| Year | Net Sales | Growth from Previous |
|------|-----------|---------------------|
| 2019 | $280,522M | -                   |
| 2021 | $469,822M | +$189,300M (67.5%)  |

Amazon's net sales grew from $280,522 million in 2019 to $469,822 million in 2021,
representing an increase of $189,300 million (approximately 67.5% growth).

📚 Sources used:
  1. AMZN 2020
  2. AMZN 2022
```

#### Decomposition Patterns

| Query Pattern | Sub-queries Generated |
|---------------|----------------------|
| "Compare Amazon 2019 vs 2021" | 2 (same company, different years) |
| "Compare Amazon and Uber in 2020" | 2 (different companies, same year) |
| "Summarize risks in Uber 2021 and Amazon 2020" | 2 (different companies, different years) |

---

### STEP 6: Problem 3 - Real-Time Stock Prices

**File**: `main.ipynb` - Cells 24-29

#### Workflow

```
User Query (Stock Price)
      │
      ▼
┌─────────────────────────────────────────────────────┐
│ APIAgent.answer(query)                               │
│                                                     │
│ 1. Extract ticker symbols                           │
│    - "Amazon" → "AMZN"                              │
│    - "Uber" → "UBER"                                │
│                                                     │
│ 2. Determine query type                             │
│    - Current price: "current", "today", "now"       │
│    - Historical: "last 7 days", "historical"        │
│                                                     │
│ 3. Call RapidAPI YFinance endpoint                  │
│    - GET /stock/get-historical                      │
│    - Or: GET /stock/get-stats                       │
│                                                     │
│ 4. Format response                                  │
│    - Extract current price, day high/low            │
│    - Extract historical closes (if applicable)      │
│                                                     │
│ 5. Generate natural language answer                 │
└─────────────────────────────────────────────────────┘
```

#### Example Usage

```python
# Query: Current stock prices
result = api_agent.answer("What is the current stock price of Amazon and Uber?")
```

**Output**:
```
📈 Fetching real-time stock data for query...
   Identified symbols: ['AMZN', 'UBER']

📝 Answer:
The current stock prices are:

* Amazon (AMZN): **$247.38**
* Uber (UBER): **$85.44**

💰 Raw Data:
AMZN:
  Current Price: $247.38
  Currency: USD
  Revenue: $691,330,023,424

UBER:
  Current Price: $85.44
  Currency: USD
  Revenue: $49,609,998,336
```

#### Query Types Supported

1. **Current Price**: "What is the current stock price of Amazon?"
2. **Multiple Symbols**: "What are the stock prices of Amazon and Uber?"
3. **Historical**: "Extract stock prices of Uber for the last 7 days"

---

### STEP 7: Quick Test Cells

**File**: `main.ipynb` - Cells 43-48

The notebook provides three pre-built test cells for easy testing:

#### Problem 1 Test Cell (Cell 44)

```python
# Just change the query and run
query = "What was Uber's total revenue in 2020?"
result = rag_agent.answer(query)
```

#### Problem 2 Test Cell (Cell 46)

```python
# Just change the query and run
query = "Compare Amazon's net sales in 2019 vs 2021"
sub_queries = decomposer.decompose(query)
result = rag_agent.answer_comparative(sub_queries)
```

#### Problem 3 Test Cell (Cell 48)

```python
# Just change the query and run
query = "What is the current stock price of Amazon?"
result = api_agent.answer(query)
```

---

## Notebook Cell Organization

| Cell Range | Purpose | Key Functions |
|------------|---------|---------------|
| 0-3 | Setup | Import, validate config |
| 4-9 | Data Loading | `load_pdfs()`, `chunk_documents()`, `MultiEmbeddingRetriever()` |
| 10-16 | Agent Init | `RAGAgent()`, `APIAgent()`, `QueryDecomposer()` |
| 17-19 | Problem 1 | `rag_agent.answer()` - Single company/year |
| 20-23 | Problem 2 | `decomposer.decompose()`, `rag_agent.answer_comparative()` |
| 24-29 | Problem 3 | `api_agent.answer()` - Real-time stock |
| 30-34 | Router (Optional) | Query routing logic |
| 35-42 | LangGraph (Optional) | Alternative LangGraph implementation |
| 43-48 | Quick Test Cells | Easy-to-use test interfaces |

---

## Code Architecture

### Agent Classes

#### RAGAgent (`agents/rag_agent.py`)

```python
class RAGAgent:
    FISCAL_TO_FILING_YEAR = {2019: 2020, 2020: 2021, 2021: 2022}

    def extract_query_metadata(self, query: str) -> Dict
    def answer(self, query, company=None, year=None, auto_extract=True) -> Dict
    def answer_comparative(self, sub_queries: List[Dict]) -> Dict
```

#### APIAgent (`agents/api_agent.py`)

```python
class APIAgent:
    def _extract_ticker_symbols(self, query: str) -> List[str]
    def _get_current_price(self, symbols: List[str]) -> Dict
    def _get_historical_prices(self, symbols: List[str]) -> Dict
    def answer(self, query: str) -> Dict
```

#### QueryDecomposer (`agents/query_decomposer.py`)

```python
class QueryDecomposer:
    def _extract_entities(self, query: str) -> Dict
    def decompose(self, query: str) -> List[Dict]
```

### Retriever Classes

#### MultiEmbeddingRetriever (`rag_system/multi_retriever.py`)

```python
class MultiEmbeddingRetriever:
    def __init__(self, chunks, use_cache=True)
    def _classify_query_type(self, query: str) -> str
    def search(self, query, company=None, year=None, top_k=10, embedding_type="AUTO") -> List
    def rerank_results(self, query, results, method="hybrid") -> List
```

---

## Usage Patterns

### Pattern 1: Direct RAG Query

```python
result = rag_agent.answer("What was Amazon's revenue in 2019?")
print(result['answer'])
print(f"Sources: {result['sources']}")
```

### Pattern 2: Manual Company/Year Specification

```python
result = rag_agent.answer(
    "What are the total liabilities?",
    company="UBER",
    year=2020
)
```

### Pattern 3: Comparative Query

```python
query = "Compare Amazon's net sales in 2019 vs 2021"
sub_queries = decomposer.decompose(query)
result = rag_agent.answer_comparative(sub_queries)
```

### Pattern 4: Real-Time Stock Query

```python
result = api_agent.answer("What is the current stock price of Amazon?")
print(result['answer'])
```

---

## Key Differences from LangGraph Implementation

| Aspect | Traditional (main.ipynb) | LangGraph |
|--------|-------------------------|-----------|
| **Entry Point** | Jupyter notebook | `lang_graph_qa.py` |
| **Control Flow** | Manual Python code | LangGraph StateGraph |
| **State Management** | Variables in cells | TypedDict state |
| **Agent Coordination** | Direct function calls | Graph nodes + edges |
| **Testing** | Run individual cells | `qa_system.route(query)` |
| **Debugging** | Print statements, cell-by-cell | Graph visualization |
| **Extensibility** | Edit notebook cells | Add nodes/edges to graph |
| **Production Deployment** | Requires conversion | Already production-ready |

---

## Running the Notebook

```bash
# From the project root
jupyter notebook solution/main.ipynb
```

**Recommended Workflow**:
1. Run cells 0-9 (Setup + Data Loading) - First time only
2. Run cells 10-16 (Agent Initialization) - Each session
3. Use cells 43-48 (Quick Test Cells) for testing

---

## File Dependencies

```
main.ipynb depends on:
├── config.py (configuration)
├── agents/
│   ├── rag_agent.py
│   ├── api_agent.py
│   └── query_decomposer.py
├── rag_system/
│   ├── pdf_loader.py
│   ├── chunking.py
│   ├── multi_retriever.py
│   └── vector_cache.py
└── utils/
    └── groq_helper.py

External data:
├── Assignment/10-k_docs/*.pdf (PDF files)
└── data/faiss_index/ (Cached FAISS database)
```
