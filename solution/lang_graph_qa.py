"""
LangGraph-based Multi-Agent System for Financial Q&A

Architecture:
- State: Holds conversation data (query, context, results, etc.)
- Tools: Retrieval tool (queries FAISS database)
- Nodes: Agent nodes (classify, retrieve, generate_answer)
- Edges: Conditional routing based on state

This replaces the orchestrator with a proper LangGraph state machine.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict, Annotated, Sequence, Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from utils.groq_helper import GroqHelper
from config import GROQ_MODEL, USE_MULTI_EMBEDDING

# ============================================================
# STATE DEFINITION
# ============================================================

class FinancialQAState(TypedDict):
    """State for the financial Q&A multi-agent system."""

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
    company: str | None
    year: int | None

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


# ============================================================
# RETRIEVAL TOOL (Uses saved FAISS database)
# ============================================================

from rag_system.multi_retriever import MultiEmbeddingRetriever


def create_retrieval_tool(retriever):
    """
    Create a LangGraph-compatible retrieval tool that uses saved FAISS database.

    This is a TOOL that agents can call to retrieve relevant chunks.
    """

    def retrieval_tool(
        query: str,
        company: str = None,
        year: int = None,
        top_k: int = 10,
        embedding_type: str = "AUTO"
    ) -> dict:
        """
        Retrieve relevant chunks from FAISS database.

        Args:
            query: The search query
            company: Filter by company (AMZN/UBER)
            year: Filter by year (2019/2020/2021/2022)
            top_k: Number of chunks to retrieve
            embedding_type: AUTO, TEXT, or NUMERICAL

        Returns:
            Dictionary with retrieved chunks and metadata
        """
        print(f"🔍 Retrieval Tool Called:")
        print(f"   Query: {query}")
        print(f"   Company: {company}, Year: {year}, Top-K: {top_k}")

        # Use the retriever (has cached FAISS database)
        results = retriever.search(
            query=query,
            company=company,
            year=year,
            top_k=top_k,
            embedding_type=embedding_type
        )

        # Format results for LangGraph
        chunks = []
        sources = []
        context_parts = []

        for i, r in enumerate(results):
            chunk = {
                'text': r['text'],
                'company': r['company'],
                'year': r['year'],
                'page': r['page'],
                'similarity': r['similarity']
            }
            chunks.append(chunk)
            sources.append({
                'company': r['company'],
                'year': r['year'],
                'page': r['page'],
                'similarity': r['similarity']
            })
            context_parts.append(f"[Source {i+1} - {r['company']} {r['year']} Pg {r['page']}]\n{r['text']}\n")

        return {
            'chunks': chunks,
            'sources': sources,
            'context': "\n".join(context_parts),
            'count': len(chunks)
        }

    return retrieval_tool


# ============================================================
# LANGGRAPH NODES (Agents)
# ============================================================

class FinancialQANodes:
    """All agent nodes for the LangGraph system."""

    def __init__(self, retriever, api_agent=None):
        self.retriever = retriever
        self.api_agent = api_agent
        self.groq = GroqHelper()

    # Node 1: Problem Type Classifier
    def problem_classifier_node(self, state: FinancialQAState) -> FinancialQAState:
        """Classify problem type (Problem 1/2/3)."""
        print("\n" + "="*70)
        print("🔷 NODE 1: Problem Type Classification")
        print("="*70)

        query = state["user_query"]
        query_lower = query.lower()

        # Classify using LLM with structured output
        system_prompt = """Classify the query into problem type:

PROBLEM_1: Single company historical query from SEC 10-K
PROBLEM_2: Multi-company/multi-year comparison
PROBLEM_3: Real-time stock price query

Respond with ONLY: PROBLEM_1 or PROBLEM_2 or PROBLEM_3"""

        try:
            response = self.groq.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}"}
                ],
                temperature=0
            )

            result = response.choices[0].message.content.strip()
            problem_type = result if result in ["PROBLEM_1", "PROBLEM_2", "PROBLEM_3"] else "PROBLEM_1"

        except:
            # Fallback: keyword-based classification
            realtime_keywords = ['current', 'today', 'latest', 'now', 'stock price']
            if any(kw in query_lower for kw in realtime_keywords):
                problem_type = "PROBLEM_3"
            elif any(kw in query_lower for kw in ['compare', 'versus', 'vs', 'and']) + \
                     (1 if 'amazon' in query_lower else 0) + (1 if 'uber' in query_lower else 0) >= 2:
                problem_type = "PROBLEM_2"
            else:
                problem_type = "PROBLEM_1"

        reasoning = {
            "PROBLEM_1": "Single company, single year - RAG from 10-K",
            "PROBLEM_2": "Multi-company/multi-year - Complex RAG",
            "PROBLEM_3": "Real-time stock price - API call"
        }[problem_type]

        print(f"Problem Type: {problem_type}")
        print(f"Reasoning: {reasoning}\n")

        # Update state
        state["problem_type"] = problem_type
        state["problem_reasoning"] = reasoning
        state["next_step"] = "complexity_classifier"

        return state

    # Node 2: Complexity Classifier
    def complexity_classifier_node(self, state: FinancialQAState) -> FinancialQAState:
        """Classify query complexity."""
        print("🔷 NODE 2: Complexity Classification")
        print("-"*70)

        query = state["user_query"]
        query_lower = query.lower()

        # Classify complexity
        reasoning_keywords = ['calculate', 'percentage', 'growth', 'rate', 'increase', 'decrease', 'margin']
        summary_keywords = ['summarize', 'overview', 'highlights', 'explain']
        comparison_keywords = ['compare', 'versus', 'vs', 'difference']
        numerical_keywords = ['revenue', 'sales', 'liabilities', 'assets', 'employees', 'income']

        has_reasoning = any(kw in query_lower for kw in reasoning_keywords)
        has_summary = any(kw in query_lower for kw in summary_keywords)
        has_comparison = any(kw in query_lower for kw in comparison_keywords)
        has_numerical = any(kw in query_lower for kw in numerical_keywords)

        if has_reasoning:
            complexity = "REASONING"
            requires_decomp = False
            info_type = "numerical"
        elif has_summary:
            complexity = "SUMMARIZATION"
            requires_decomp = False
            info_type = "mixed"
        elif has_comparison:
            complexity = "COMPLEX"
            requires_decomp = True
            info_type = "numerical" if has_numerical else "narrative"
        else:
            complexity = "SIMPLE"
            requires_decomp = False
            info_type = "numerical" if has_numerical else "narrative"

        print(f"Complexity: {complexity}")
        print(f"Information Type: {info_type}")
        print(f"Requires Decomposition: {requires_decomp}\n")

        # Extract entities
        company = None
        if 'amazon' in query_lower or 'amzn' in query_lower:
            company = 'AMZN'
        elif 'uber' in query_lower:
            company = 'UBER'

        year = None
        for y in [2019, 2020, 2021, 2022]:
            if str(y) in query:
                year = y

        print(f"Extracted - Company: {company}, Year: {year}\n")

        state["complexity"] = complexity
        state["info_type"] = info_type
        state["requires_decomposition"] = requires_decomp
        state["company"] = company
        state["year"] = year
        state["next_step"] = "check_problem_type"

        return state

    # Node 3: Entity Extraction
    def entity_extraction_node(self, state: FinancialQAState) -> FinancialQAState:
        """Extract company and year with fiscal year mapping."""
        print("🔷 NODE 3: Entity Extraction & Year Mapping")
        print("-"*70)

        query = state["user_query"]
        query_lower = query.lower()

        # Extract company
        company = None
        if 'amazon' in query_lower or 'amzn' in query_lower:
            company = 'AMZN'
        elif 'uber' in query_lower:
            company = 'UBER'

        # Extract year
        year = None
        for y in [2019, 2020, 2021, 2022]:
            if str(y) in query:
                year = y

        # Apply fiscal to filing year mapping
        fiscal_to_filing = {2019: 2020, 2020: 2021, 2021: 2022}
        if year and year in fiscal_to_filing:
            fiscal_year = year
            filing_year = fiscal_to_filing[year]
            print(f"Fiscal Year: {fiscal_year}")
            print(f"Mapped to Filing Year: {filing_year}")
            year = filing_year

        print(f"Final - Company: {company}, Year: {year}\n")

        state["company"] = company
        state["year"] = year
        state["next_step"] = "retrieval"

        return state

    # Node 4: Retrieval
    def retrieval_node(self, state: FinancialQAState) -> FinancialQAState:
        """Retrieve relevant chunks from FAISS database."""
        print("🔷 NODE 4: Retrieval (FAISS Database)")
        print("-"*70)

        # Determine retrieval strategy based on complexity
        if state["complexity"] == "SIMPLE":
            top_k = 5
            embedding = "NUMERICAL" if state["info_type"] == "numerical" else "TEXT"
        elif state["complexity"] == "COMPLEX":
            top_k = 15
            embedding = "NUMERICAL"
        elif state["complexity"] == "REASONING":
            top_k = 12
            embedding = "NUMERICAL"
        else:  # SUMMARIZATION
            top_k = 20
            embedding = "TEXT"

        print(f"Strategy: {state['complexity']} query")
        print(f"Top-K: {top_k}, Embedding: {embedding}")

        # Create retrieval tool and call it
        retrieval_tool = create_retrieval_tool(self.retriever)

        retrieval_result = retrieval_tool(
            query=state["query"],
            company=state["company"],
            year=state["year"],
            top_k=top_k,
            embedding_type=embedding
        )

        print(f"Retrieved {retrieval_result['count']} chunks\n")

        state["retrieved_chunks"] = retrieval_result["chunks"]
        state["context"] = retrieval_result["context"]
        state["sources"] = retrieval_result["sources"]
        state["retrieval_strategy"] = f"{state['complexity']}_{state['info_type']}"
        state["next_step"] = "answer_generation"

        return state

    # Node 5: Answer Generation
    def answer_generation_node(self, state: FinancialQAState) -> FinancialQAState:
        """Generate and validate final answer."""
        print("🔷 NODE 5: Answer Generation & Validation")
        print("-"*70)

        # Select appropriate prompt based on complexity
        if state["complexity"] == "SIMPLE":
            prompt_type = "direct"
        elif state["complexity"] == "COMPLEX":
            prompt_type = "comparison"
        elif state["complexity"] == "REASONING":
            prompt_type = "analysis"
        else:  # SUMMARIZATION
            prompt_type = "summary"

        system_prompts = {
            "direct": """Extract the answer directly from context. Be concise and specific with numbers.""",
            "comparison": """Compare the values mentioned. Highlight differences and similarities clearly.""",
            "analysis": """Show your reasoning or calculation steps clearly. Provide final answer with methodology.""",
            "summary": """Synthesize key points from context. Use bullet points or structured format."""
        }

        system_prompt = f"""You are a financial analyst answering questions based on SEC 10-K filings.

CRITICAL RULES:
1. The context provided IS from the correct company and year - TRUST it
2. Extract exact numbers with units (million, billion, percentage)
3. {system_prompts[prompt_type]}

Context from {state.get('company', '')} {state.get('year', '')} 10-K:
{state['context']}

Question: {state['query']}

Answer:"""

        try:
            response = self.groq.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "system", "content": system_prompt}],
                temperature=0.1
            )

            answer = response.choices[0].message.content

            # Validate answer
            has_answer = any(char.isdigit() for char in answer) if state['info_type'] == 'numerical' else len(answer) > 50
            negative_phrases = ['could not find', 'does not contain', 'not available', 'no information']
            has_negative = any(phrase in answer.lower() for phrase in negative_phrases)

            confidence = "HIGH" if has_answer and not has_negative else "MEDIUM" if has_answer else "LOW"

            print(f"Answer Generated: {confidence} confidence")
            print(f"Validation: {'✓ Valid' if has_answer else '✗ Check needed'}\n")

            state["answer"] = answer
            state["has_answer"] = has_answer and not has_negative
            state["confidence"] = confidence

        except Exception as e:
            print(f"Error: {e}")
            state["answer"] = "I encountered an error generating the answer."
            state["has_answer"] = False
            state["confidence"] = "LOW"

        state["next_step"] = END
        return state


# ============================================================
# CONDITIONAL EDGES (Routing)
# ============================================================

def should_use_api(state: FinancialQAState) -> bool:
    """Check if query should go to API (real-time stock)."""
    return state.get("problem_type") == "PROBLEM_3"


def should_decompose(state: FinancialQAState) -> bool:
    """Check if query needs decomposition (complex comparison)."""
    return state.get("requires_decomposition", False)


# ============================================================
# BUILD LANGGRAPH
# ============================================================

def build_financial_qa_graph(retriever, api_agent=None):
    """
    Build the LangGraph state graph for financial Q&A.

    Args:
        retriever: MultiEmbeddingRetriever with cached FAISS database
        api_agent: Optional APIAgent for real-time stock queries

    Returns:
        Compiled LangGraph
    """
    # Initialize nodes
    nodes = FinancialQANodes(retriever, api_agent)

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

    # Compile the graph
    app = graph.compile()

    print("="*70)
    print("LANGGRAPH FINANCIAL Q&A SYSTEM")
    print("="*70)
    print("\nNodes:")
    print("  1. problem_classifier → Problem Type (1/2/3)")
    print("  2. complexity_classifier → Complexity (Simple/Complex/Reasoning/Summary)")
    print("  3. entity_extraction → Extract company/year with mapping")
    print("  4. retrieval → Query FAISS database (saved embeddings)")
    print("  5. answer_generation → Generate validated answer")
    print("\nTool:")
    print("  - retrieval_tool (uses cached FAISS database)")
    print("\n✅ Graph compiled successfully!\n")

    return app


# ============================================================
# MAIN ENTRY POINT
# ============================================================

class LangGraphFinancialQA:
    """Main interface for the LangGraph-based financial Q&A system."""

    def __init__(self, retriever, api_agent=None):
        """
        Initialize the LangGraph system.

        Args:
            retriever: MultiEmbeddingRetriever with cached FAISS database
            api_agent: Optional APIAgent for real-time stock
        """
        print("\n" + "="*70)
        print("INITIALIZING LANGGRAPH MULTI-AGENT SYSTEM")
        print("="*70)

        self.retriever = retriever
        self.api_agent = api_agent
        self.graph = build_financial_qa_graph(retriever, api_agent)

        print("✅ LangGraph system ready!\n")

    def visualize_graph(self):
        """
        Display the LangGraph state machine visualization.

        Shows the complete workflow with nodes and edges as a visual graph.
        """
        try:
            from IPython.display import Image, display

            # Generate the graph visualization
            graph_image = self.graph.get_graph().draw_mermaid_png()

            print("\n" + "="*70)
            print("LANGGRAPH WORKFLOW VISUALIZATION")
            print("="*70)
            print("\n📊 This graph shows the complete query processing pipeline:\n")
            print("   • Nodes: Agent processing steps")
            print("   • Edges: Flow between nodes")
            print("   • State: Data passed through the pipeline\n")

            display(Image(graph_image))

            print("\n" + "="*70)
            print("✅ Graph visualization complete!")
            print("="*70 + "\n")

        except ImportError:
            print("\n⚠️ IPython display not available. Using text description instead:\n")
            self._print_text_description()
        except Exception as e:
            print(f"\n⚠️ Could not display graph: {e}")
            print("Using text description instead:\n")
            self._print_text_description()

    def _print_text_description(self):
        """Print a text description of the graph flow."""
        print("\n" + "="*70)
        print("LANGGRAPH WORKFLOW (Text Description)")
        print("="*70)
        print("""
┌─────────────────────────────────────────────────────────────┐
│  QUERY → problem_classifier → complexity_classifier        │
│         → entity_extraction → retrieval → answer_generation │
│         → END                                                │
└─────────────────────────────────────────────────────────────┘

Nodes:
  1. problem_classifier     → PROBLEM_1/2/3
  2. complexity_classifier  → SIMPLE/COMPLEX/REASONING/SUMMARY
  3. entity_extraction      → Company & Year with mapping
  4. retrieval              → Query FAISS database
  5. answer_generation      → Generate validated answer

Tool:
  - retrieval_tool (queries cached FAISS database)
""")

    def print_architecture(self):
        """Print the complete LangGraph architecture before query execution."""
        print("\n" + "╔" + "═"*68 + "╗")
        print("║" + " "*20 + "LANGGRAPH ARCHITECTURE" + " "*26 + "║")
        print("╚" + "═"*68 + "╝")

        print("\n┌" + "─"*68 + "┐")
        print("│" + " "*15 + "USER QUERY ENTERS SYSTEM" + " "*29 + "│")
        print("│" + " "*68 + "│")
        print("│  Example: \"What was Amazon's revenue in 2019?\"" + " "*20 + "│")
        print("└" + "─"*68 + "┘")
        print("\n" + "│"*70)
        print("│"*70)
        print("│"*70)

        # STATE
        print("┌" + "─"*68 + "┐")
        print("│" + " "*22 + "INITIALIZE STATE" + " "*30 + "│")
        print("└" + "─"*68 + "┘")
        print("""
┌──────────────────────────────────────────────────────────────────────┐
│  FinancialQAState (TypedDict)                                        │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  query: str           → User's question                         │  │
│  │  problem_type: None    → Will be classified (PROBLEM_1/2/3)    │  │
│  │  complexity: None      → Will be classified                    │  │
│  │  company: None         → Will be extracted                      │  │
│  │  year: None            → Will be extracted + mapped            │  │
│  │  retrieved_chunks: []  → Will be populated from FAISS          │  │
│  │  context: ""           → Will be built from chunks             │  │
│  │  answer: ""            → Will be generated by LLM              │  │
│  │  confidence: "LOW"     → Will be scored (HIGH/MEDIUM/LOW)      │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
""")

        print("│"*70)
        print("│"*70)
        print("│"*70)

        # NODE 1
        print("┌" + "─"*68 + "┐")
        print("│" + " "*8 + "NODE 1: PROBLEM CLASSIFIER AGENT" + " "*24 + "│")
        print("└" + "─"*68 + "┘")
        print("""
┌──────────────────────────────────────────────────────────────────────┐
│  INPUT:  state["user_query"]                                         │
│                                                                          │
│  PROCESS:                                                               │
│  • Analyze query using LLM (llama-3.3-70b-versatile)                  │
│  • Classify into problem type:                                         │
│      - PROBLEM_1: Single company, single year (RAG from 10-K)        │
│      - PROBLEM_2: Multi-company or multi-year comparison              │
│      - PROBLEM_3: Real-time stock price (API call)                    │
│                                                                          │
│  OUTPUT:                                                                 │
│  → state["problem_type"] = "PROBLEM_1" | "PROBLEM_2" | "PROBLEM_3"    │
│  → state["problem_reasoning"] = explanation                            │
│  → state["next_step"] = "complexity_classifier"                       │
└──────────────────────────────────────────────────────────────────────┘
""")

        print("│"*70)
        print("│"*70)
        print("│"*70)

        # NODE 2
        print("┌" + "─"*68 + "┐")
        print("│" + " "*6 + "NODE 2: COMPLEXITY CLASSIFIER AGENT" + " "*24 + "│")
        print("└" + "─"*68 + "┘")
        print("""
┌──────────────────────────────────────────────────────────────────────┐
│  INPUT:  state["user_query"]                                         │
│                                                                          │
│  PROCESS:                                                               │
│  • Analyze query complexity using keyword + LLM                        │
│  • Classify into:                                                       │
│      - SIMPLE: Direct fact retrieval                                   │
│      - COMPLEX: Comparison or multi-part query                         │
│      - REASONING: Calculation or analysis needed                       │
│      - SUMMARIZATION: Overview or synthesis                            │
│  • Determine information type: numerical | narrative | mixed          │
│  • Extract company (AMZN | UBER) and year (2019-2022)                 │
│                                                                          │
│  OUTPUT:                                                                 │
│  → state["complexity"] = "SIMPLE" | "COMPLEX" | "REASONING" | "SUMMARIZATION"│
│  → state["info_type"] = "numerical" | "narrative" | "mixed"           │
│  → state["requires_decomposition"] = True | False                      │
│  → state["company"] = "AMZN" | "UBER" | None                           │
│  → state["year"] = 2019 | 2020 | 2021 | 2022 | None                   │
│  → state["next_step"] = "entity_extraction"                            │
└──────────────────────────────────────────────────────────────────────┘
""")

        print("│"*70)
        print("│"*70)
        print("│"*70)

        # NODE 3
        print("┌" + "─"*68 + "┐")
        print("│" + " "*10 + "NODE 3: ENTITY EXTRACTION AGENT" + " "*22 + "│")
        print("└" + "─"*68 + "┘")
        print("""
┌──────────────────────────────────────────────────────────────────────┐
│  INPUT:  state["user_query"], state["company"], state["year"]        │
│                                                                          │
│  PROCESS:                                                               │
│  • Extract company name from query                                      │
│  • Extract fiscal year from query                                       │
│  • APPLY FISCAL → FILING YEAR MAPPING:                                  │
│      Fiscal Year 2019 → Filing Year 2020 (file: AMZN_2020.pdf)       │
│      Fiscal Year 2020 → Filing Year 2021 (file: AMZN_2021.pdf)       │
│      Fiscal Year 2021 → Filing Year 2022 (file: AMZN_2022.pdf)       │
│                                                                          │
│  OUTPUT:                                                                 │
│  → state["company"] = "AMZN" | "UBER"                                  │
│  → state["year"] = filing_year (2020 | 2021 | 2022)                    │
│  → state["next_step"] = "retrieval"                                    │
└──────────────────────────────────────────────────────────────────────┘
""")

        print("│"*70)
        print("│"*70)
        print("│"*70)

        # NODE 4
        print("┌" + "─"*68 + "┐")
        print("│" + " "*18 + "NODE 4: RETRIEVAL AGENT" + " "*25 + "│")
        print("└" + "─"*68 + "┘")
        print("""
┌──────────────────────────────────────────────────────────────────────┐
│  INPUT:  state["query"], state["company"], state["year"],             │
│          state["complexity"], state["info_type"]                       │
│                                                                          │
│  PROCESS:                                                               │
│  • DETERMINE RETRIEVAL STRATEGY:                                        │
│                                                                          │
│      Complexity        | Embedding     | Top-K | Purpose              │
│      ──────────────────────────────────────────────────────────────    │
│      SIMPLE + numerical | NUMERICAL     | 5     | Fast, accurate       │
│      SIMPLE + narrative | TEXT          | 5     | Semantic search      │
│      COMPLEX            | NUMERICAL     | 15    | Extended + rerank    │
│      REASONING          | NUMERICAL     | 12    | Context-aware        │
│      SUMMARIZATION      | TEXT          | 20    | Broad retrieval      │
│                                                                          │
│  • CALL RETRIEVAL TOOL:                                                  │
│      → retrieval_tool(query, company, year, top_k, embedding_type)      │
│      → Queries cached FAISS database (no re-encoding!)                  │
│                                                                          │
│  FAISS Database (Cached):                                                │
│  • data/faiss_index/text_index.faiss (384-dim, MiniLM)                  │
│  • data/faiss_index/numerical_index.faiss (768-dim, mpnet)             │
│  • data/faiss_index/chunks.pkl (chunk metadata)                         │
│                                                                          │
│  OUTPUT:                                                                 │
│  → state["retrieved_chunks"] = [chunk1, chunk2, ...]                    │
│  → state["context"] = formatted context with sources                    │
│  → state["sources"] = [{company, year, page, similarity}, ...]          │
│  → state["retrieval_strategy"] = "SIMPLE_numerical"                     │
│  → state["next_step"] = "answer_generation"                             │
└──────────────────────────────────────────────────────────────────────┘
""")

        print("│"*70)
        print("│"*70)
        print("│"*70)

        # NODE 5
        print("┌" + "─"*68 + "┐")
        print("│" + " "*8 + "NODE 5: ANSWER GENERATION AGENT" + " "*22 + "│")
        print("└" + "─"*68 + "┘")
        print("""
┌──────────────────────────────────────────────────────────────────────┐
│  INPUT:  state["query"], state["context"], state["complexity"]        │
│                                                                          │
│  PROCESS:                                                               │
│  • Select prompt based on complexity:                                    │
│      - direct: Extract specific fact                                    │
│      - comparison: Highlight differences/similarities                   │
│      - analysis: Show reasoning/calculation                             │
│      - summary: Synthesize key points                                   │
│                                                                          │
│  • Generate answer using LLM with context                                │
│  • Validate answer:                                                      │
│      ✓ Check for numerical values (if numerical query)                  │
│      ✓ Check for negative phrases ("not found", "no information")       │
│      ✓ Assign confidence score                                          │
│                                                                          │
│  OUTPUT:                                                                 │
│  → state["answer"] = generated answer text                              │
│  → state["has_answer"] = True | False                                   │
│  → state["confidence"] = "HIGH" | "MEDIUM" | "LOW"                      │
│  → state["next_step"] = END                                             │
└──────────────────────────────────────────────────────────────────────┘
""")

        print("│"*70)
        print("│"*70)
        print("│"*70)

        # FINAL OUTPUT
        print("┌" + "─"*68 + "┐")
        print("│" + " "*23 + "FINAL OUTPUT" + " "*31 + "│")
        print("└" + "─"*68 + "┘")
        print("""
┌──────────────────────────────────────────────────────────────────────┐
│  RETURN FORMAT:                                                       │
│  {                                                                     │
│    'answer': "The net sales were $280,522 million...",                │
│    'has_answer': True,                                                 │
│    'confidence': "HIGH",                                               │
│    'sources': [                                                        │
│      {'company': 'AMZN', 'year': 2020, 'page': 22, 'similarity': 0.89}│
│    ],                                                                  │
│    'method': 'LANGGRAPH_RAG',                                         │
│    'agent_trace': {                                                    │
│      'problem_type': 'PROBLEM_1',                                      │
│      'complexity': 'SIMPLE',                                           │
│      'company': 'AMZN',                                                │
│      'year': 2020,                                                     │
│      'retrieval_count': 5                                              │
│    }                                                                   │
│  }                                                                     │
└──────────────────────────────────────────────────────────────────────┘
""")

        print("="*70)
        print("✅ PIPELINE READY - QUERY EXECUTION STARTING")
        print("="*70)
        print()

    def route(self, query: str) -> dict:
        """
        Process a query through the LangGraph pipeline.

        Args:
            query: User's question

        Returns:
            Dictionary with answer and metadata
        """
        # Initialize state
        initial_state = FinancialQAState(
            query=query,
            user_query=query,
            problem_type=None,
            complexity=None,
            company=None,
            year=None,
            retrieved_chunks=[],
            context="",
            answer="",
            has_answer=False,
            confidence="LOW",
            sources=[],
            agent_trace={},
            next_step="problem_classifier"
        )

        # Check if it's an API query (Problem 3)
        query_lower = query.lower()
        api_keywords = ['current', 'today', 'latest', 'now', 'stock price']

        if any(kw in query_lower for kw in api_keywords):
            if self.api_agent:
                print("\n" + "="*70)
                print("🔷 ROUTING TO API (Real-Time Stock)")
                print("="*70)
                print(f"Query: {query}\n")
                api_result = self.api_agent.answer(query)
                return {
                    'answer': api_result['answer'],
                    'has_answer': True,
                    'confidence': 'HIGH',
                    'sources': [],
                    'method': 'API_REALTIME'
                }

        # Run through LangGraph
        print("\n" + "="*70)
        print("🚀 RUNNING LANGGRAPH PIPELINE")
        print("="*70)
        print(f"Query: {query}\n")

        # Invoke the graph
        final_state = self.graph.invoke(initial_state)

        # Return formatted result
        return {
            'answer': final_state['answer'],
            'has_answer': final_state['has_answer'],
            'confidence': final_state['confidence'],
            'sources': final_state['sources'],
            'method': 'LANGGRAPH_RAG',
            'agent_trace': {
                'problem_type': final_state['problem_type'],
                'complexity': final_state['complexity'],
                'company': final_state['company'],
                'year': final_state['year'],
                'retrieval_count': len(final_state['retrieved_chunks'])
            }
        }


if __name__ == "__main__":
    # Test the LangGraph system
    from rag_system.pdf_loader import load_pdfs
    from rag_system.chunking import chunk_documents
    from agents.api_agent import APIAgent

    print("Loading data...")
    pdf_path = os.path.join(os.path.dirname(os.getcwd()), "Assignment", "10-k_docs")
    documents = load_pdfs(pdf_path)
    chunks = chunk_documents(documents)

    print("Creating retriever...")
    from rag_system.multi_retriever import MultiEmbeddingRetriever
    retriever = MultiEmbeddingRetriever(chunks, use_cache=True)
    api_agent = APIAgent()

    # Create LangGraph system
    qa_system = LangGraphFinancialQA(retriever, api_agent)

    # Test
    test_queries = [
        "What was Amazon's revenue in 2019?",
        "What percentage of Amazon's revenue was in Q4 2019?",
        "Compare Amazon's net sales in 2019 vs 2021"
    ]

    for query in test_queries:
        result = qa_system.route(query)
        print(f"\n📝 Answer: {result['answer'][:200]}...")
        print(f"Confidence: {result['confidence']}")
        print("="*70)
        print()
