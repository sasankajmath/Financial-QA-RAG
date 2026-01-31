# LangGraph Implementation Workflow

This document describes the complete workflow for the **LangGraph-based** Financial Q&A system.

## Overview

The LangGraph implementation uses a state machine architecture where queries flow through defined nodes (agents) with state transitions managed by LangGraph's `StateGraph`.

```
┌──────────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH WORKFLOW                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  User Query                                                           │
│      │                                                                │
│      ▼                                                                │
│  ┌──────────────────┐                                                │
│  │  ENTRY POINT     │                                                │
│  └──────────────────┘                                                │
│      │                                                                │
│      ▼                                                                │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  NODE 1: problem_classifier                                     │  │
│  │  - Classifies query type: PROBLEM_1, PROBLEM_2, or PROBLEM_3  │  │
│  │  - Updates state["problem_type"]                               │  │
│  └────────────────────────────────────────────────────────────────┘  │
│      │                                                                │
│      ▼                                                                │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  NODE 2: complexity_classifier                                  │  │
│  │  - Determines: SIMPLE, COMPLEX, REASONING, SUMMARIZATION       │  │
│  │  - Extracts: company, year, info_type                          │  │
│  └────────────────────────────────────────────────────────────────┘  │
│      │                                                                │
│      ▼                                                                │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  NODE 3: entity_extraction                                      │  │
│  │  - Extracts company (AMZN/UBER)                                │  │
│  │  - Maps fiscal year → filing year                              │  │
│  └────────────────────────────────────────────────────────────────┘  │
│      │                                                                │
│      ▼                                                                │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  NODE 4: retrieval (uses cached FAISS database)                 │  │
│  │  - Selects embedding type (TEXT/NUMERICAL)                     │  │
│  │  - Determines top_k based on complexity                        │  │
│  │  - Queries FAISS vector database                               │  │
│  └────────────────────────────────────────────────────────────────┘  │
│      │                                                                │
│      ▼                                                                │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  NODE 5: answer_generation                                      │  │
│  │  - Generates answer using LLM                                   │  │
│  │  - Validates answer quality                                     │  │
│  │  - Assigns confidence score                                    │  │
│  └────────────────────────────────────────────────────────────────┘  │
│      │                                                                │
│      ▼                                                                │
│  ┌──────────────────┐                                                │
│  │  END (Return)    │                                                │
│  └──────────────────┘                                                │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## State Definition

### FinancialQAState

```python
class FinancialQAState(TypedDict):
    # Input
    query: str
    user_query: str

    # Agent 1: Problem Classification
    problem_type: Literal["PROBLEM_1", "PROBLEM_2", "PROBLEM_3"]
    problem_reasoning: str

    # Agent 2: Complexity Classification
    complexity: Literal["SIMPLE", "COMPLEX", "REASONING", "SUMMARIZATION"]
    info_type: Literal["numerical", "narrative", "mixed"]
    requires_decomposition: bool

    # Extraction
    company: str | None      # "AMZN" or "UBER"
    year: int | None         # 2019, 2020, 2021, 2022

    # Agent 3: Retrieval
    retrieved_chunks: list
    context: str
    retrieval_strategy: str

    # Agent 4: Answer Generation
    answer: str
    has_answer: bool
    confidence: Literal["HIGH", "MEDIUM", "LOW"]

    # Metadata
    sources: list
    agent_trace: dict
    next_step: str
```

---

## Node Details

### NODE 1: Problem Classifier

**Purpose**: Classify the query into one of three problem types

**Input**:
- `state["user_query"]`

**Processing**:
```python
# Uses LLM to classify query
system_prompt = """
Classify the query into problem type:
PROBLEM_1: Single company historical query from SEC 10-K
PROBLEM_2: Multi-company/multi-year comparison
PROBLEM_3: Real-time stock price query
"""
```

**Output**:
- `state["problem_type"]` → `"PROBLEM_1"` | `"PROBLEM_2"` | `"PROBLEM_3"`
- `state["problem_reasoning"]` → Explanation

**Example Classifications**:
| Query | Problem Type | Reasoning |
|-------|-------------|-----------|
| "What was Amazon's revenue in 2019?" | PROBLEM_1 | Single company, single year - RAG from 10-K |
| "Compare Amazon and Uber revenue in 2020" | PROBLEM_2 | Multi-company comparison |
| "What is Amazon's current stock price?" | PROBLEM_3 | Real-time stock price - API call |

---

### NODE 2: Complexity Classifier

**Purpose**: Determine query complexity to optimize retrieval strategy

**Input**:
- `state["user_query"]`

**Processing**:
```python
# Keyword-based + LLM classification
reasoning_keywords = ['calculate', 'percentage', 'growth', 'rate']
summary_keywords = ['summarize', 'overview', 'highlights']
comparison_keywords = ['compare', 'versus', 'vs', 'difference']
```

**Output**:
- `state["complexity"]` → `"SIMPLE"` | `"COMPLEX"` | `"REASONING"` | `"SUMMARIZATION"`
- `state["info_type"]` → `"numerical"` | `"narrative"` | `"mixed"`
- `state["requires_decomposition"]` → `True` | `False`
- `state["company"]` → `"AMZN"` | `"UBER"` | `None`
- `state["year"]` → `2019` | `2020` | `2021` | `2022` | `None`

---

### NODE 3: Entity Extraction & Year Mapping

**Purpose**: Extract entities and map fiscal years to filing years

**Input**:
- `state["user_query"]`
- `state["company"]`
- `state["year"]`

**Processing**:
```python
# Fiscal Year → Filing Year Mapping
fiscal_to_filing = {
    2019: 2020,  # Fiscal 2019 → Filed in 2020
    2020: 2021,  # Fiscal 2020 → Filed in 2021
    2021: 2022   # Fiscal 2021 → Filed in 2022
}
```

**Output**:
- `state["company"]` → Mapped to `"AMZN"` or `"UBER"`
- `state["year"]` → Mapped to filing year (2020, 2021, 2022)

**Why This Matters**:
The file `AMZN_2020.pdf` contains fiscal year 2019 data, not 2020 data. This mapping ensures we retrieve from the correct document.

---

### NODE 4: Retrieval

**Purpose**: Query the cached FAISS database for relevant chunks

**Input**:
- `state["query"]`
- `state["company"]`
- `state["year"]`
- `state["complexity"]`
- `state["info_type"]`

**Retrieval Strategy Selection**:

| Complexity | Info Type | Embedding | Top-K | Purpose |
|------------|-----------|-----------|-------|---------|
| SIMPLE | numerical | NUMERICAL | 5 | Fast, accurate |
| SIMPLE | narrative | TEXT | 5 | Semantic search |
| COMPLEX | any | NUMERICAL | 15 | Extended + rerank |
| REASONING | any | NUMERICAL | 12 | Context-aware |
| SUMMARIZATION | any | TEXT | 20 | Broad retrieval |

**Processing**:
```python
retrieval_tool(
    query=state["query"],
    company=state["company"],     # "AMZN" or "UBER"
    year=state["year"],           # 2020, 2021, 2022
    top_k=top_k,                 # 5, 12, 15, or 20
    embedding_type=embedding      # "TEXT", "NUMERICAL", or "AUTO"
)
```

**FAISS Database** (Cached at `data/faiss_index/`):
- `text_index.faiss` (384-dim, MiniLM embeddings)
- `numerical_index.faiss` (768-dim, mpnet embeddings)
- `chunks.pkl` (chunk metadata)

**Output**:
- `state["retrieved_chunks"]` → List of retrieved chunks
- `state["context"]` → Formatted context with sources
- `state["sources"]` → Source metadata
- `state["retrieval_strategy"]` → Strategy used

---

### NODE 5: Answer Generation

**Purpose**: Generate and validate final answer

**Input**:
- `state["query"]`
- `state["context"]`
- `state["complexity"]`

**Prompt Selection**:

| Complexity | Prompt Type | System Prompt Focus |
|------------|-------------|---------------------|
| SIMPLE | direct | Extract answer directly, be concise |
| COMPLEX | comparison | Compare values, highlight differences |
| REASONING | analysis | Show reasoning/calculation steps |
| SUMMARIZATION | summary | Synthesize key points |

**Processing**:
```python
system_prompt = f"""
You are a financial analyst answering questions based on SEC 10-K filings.

CRITICAL RULES:
1. The context provided IS from the correct company and year - TRUST it
2. Extract exact numbers with units (million, billion, percentage)
3. {system_prompts[prompt_type]}

Context from {state.get('company', '')} {state.get('year', '')} 10-K:
{state['context']}

Question: {state['query']}
"""
```

**Validation**:
```python
has_answer = any(char.isdigit() for char in answer)  # For numerical queries
negative_phrases = ['could not find', 'does not contain', 'not available']
has_negative = any(phrase in answer.lower() for phrase in negative_phrases)

confidence = "HIGH" if has_answer and not has_negative else "MEDIUM" if has_answer else "LOW"
```

**Output**:
- `state["answer"]` → Generated answer text
- `state["has_answer"]` → `True` | `False`
- `state["confidence"]` → `"HIGH"` | `"MEDIUM"` | `"LOW"`

---

## Graph Construction

### Building the Graph

```python
def build_financial_qa_graph(retriever, api_agent=None):
    # Create state graph
    graph = StateGraph(FinancialQAState)

    # Add nodes
    graph.add_node("problem_classifier", nodes.problem_classifier_node)
    graph.add_node("complexity_classifier", nodes.complexity_classifier_node)
    graph.add_node("entity_extraction", nodes.entity_extraction_node)
    graph.add_node("retrieval", nodes.retrieval_node)
    graph.add_node("answer_generation", nodes.answer_generation_node)

    # Add edges (flow)
    graph.set_entry_point("problem_classifier")

    graph.add_edge("problem_classifier", "complexity_classifier")
    graph.add_edge("complexity_classifier", "entity_extraction")
    graph.add_edge("entity_extraction", "retrieval")
    graph.add_edge("retrieval", "answer_generation")
    graph.add_edge("answer_generation", END)

    # Compile
    app = graph.compile()
    return app
```

### Graph Visualization

The graph can be visualized using:

```python
qa_system = LangGraphFinancialQA(retriever, api_agent)
qa_system.visualize_graph()
```

Output:
```
┌─────────────────────────────────────────────────────────────┐
│  QUERY → problem_classifier → complexity_classifier        │
│         → entity_extraction → retrieval → answer_generation │
│         → END                                                │
└─────────────────────────────────────────────────────────────┘
```

---

## Main Interface

### LangGraphFinancialQA Class

```python
class LangGraphFinancialQA:
    def __init__(self, retriever, api_agent=None):
        self.retriever = retriever
        self.api_agent = api_agent
        self.graph = build_financial_qa_graph(retriever, api_agent)

    def route(self, query: str) -> dict:
        # Initialize state
        initial_state = FinancialQAState(...)

        # Check for API query (PROBLEM_3)
        if is_api_query(query):
            return self.api_agent.answer(query)

        # Run through LangGraph
        final_state = self.graph.invoke(initial_state)

        return {
            'answer': final_state['answer'],
            'has_answer': final_state['has_answer'],
            'confidence': final_state['confidence'],
            'sources': final_state['sources'],
            'method': 'LANGGRAPH_RAG',
            'agent_trace': {...}
        }
```

---

## Usage Example

```python
from lang_graph_qa import LangGraphFinancialQA
from rag_system.multi_retriever import MultiEmbeddingRetriever
from agents.api_agent import APIAgent

# Initialize
retriever = MultiEmbeddingRetriever(chunks, use_cache=True)
qa_system = LangGraphFinancialQA(retriever, APIAgent())

# Query
result = qa_system.route("What was Amazon's revenue in 2019?")

# Output
print(result['answer'])
# "According to the 10-K filing, Amazon's net sales for fiscal year 2019
#  were $280,522 million..."

print(result['agent_trace'])
# {
#   'problem_type': 'PROBLEM_1',
#   'complexity': 'SIMPLE',
#   'company': 'AMZN',
#   'year': 2020,  # Mapped from fiscal 2019
#   'retrieval_count': 5
# }
```

---

## Key Differences from Traditional Implementation

| Aspect | Traditional | LangGraph |
|--------|-------------|-----------|
| **State** | Variables in notebook | TypedDict state |
| **Flow Control** | If/else statements | Graph edges |
| **Agent Coordination** | Manual function calls | State transitions |
| **Debugging** | Print statements | Graph visualization |
| **Scalability** | Add to notebook | Add node + edge |

---

## File Locations

- **Main File**: `solution/lang_graph_qa.py`
- **State Definition**: Line 27-60
- **Nodes Implementation**: Line 146-427
- **Graph Builder**: Line 448-497
- **Main Interface**: Line 504-883

---

## Testing the LangGraph Implementation

```python
# See test_questions.md for comprehensive tests
test_queries = [
    # Problem 1
    "What were the total liabilities reported by Uber in 2020?",

    # Problem 2
    "Compare Amazon's net sales in 2019 vs 2021",

    # Problem 3
    "What is Amazon's current stock price?"
]

for query in test_queries:
    result = qa_system.route(query)
    print(f"Query: {query}")
    print(f"Answer: {result['answer'][:200]}...")
    print(f"Confidence: {result['confidence']}\n")
```
