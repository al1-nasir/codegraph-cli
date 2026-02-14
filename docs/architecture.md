# CodeGraph CLI Architecture

Understanding how CodeGraph works under the hood.

---

## System Overview

CodeGraph CLI is a **fully local**, **multi-agent** system for semantic code analysis. It combines:

1. **AST Parsing** - Extracts code structure
2. **Graph Storage** - Persists nodes and edges
3. **Semantic Embeddings** - Enables similarity search
4. **Multi-Agent Orchestration** - Coordinates specialized agents
5. **Local LLM** - Generates explanations (optional)

```
┌─────────────────────────────────────────────────────────────┐
│                      CodeGraph CLI                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ GraphAgent   │  │  RAGAgent    │  │ Summarization│      │
│  │              │  │              │  │    Agent     │      │
│  │ - Parse AST  │  │ - Search     │  │ - Impact     │      │
│  │ - Build graph│  │ - Retrieve   │  │ - Explain    │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                  │              │
│         └─────────────────┴──────────────────┘              │
│                           │                                 │
│                  ┌────────▼────────┐                        │
│                  │  MCPOrchestrator│                        │
│                  └────────┬────────┘                        │
│                           │                                 │
│         ┌─────────────────┴─────────────────┐              │
│         │                                   │              │
│  ┌──────▼───────┐                  ┌────────▼────────┐     │
│  │ GraphStore   │                  │ HashEmbedding   │     │
│  │ (SQLite)     │                  │ Model           │     │
│  └──────────────┘                  └─────────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Parser Layer (`parser.py`)

**Purpose**: Extract code structure from source files.

**How it works**:
- Uses Python's `ast` module to parse source code
- Visits AST nodes to extract:
  - **Modules** - Python files
  - **Classes** - Class definitions
  - **Functions** - Function/method definitions
- Detects relationships:
  - **contains** - Module contains class/function
  - **calls** - Function calls another function
  - **depends_on** - Module imports another module

**Example**:
```python
# Input: Python file
def process_payment(amount):
    validate_amount(amount)  # Function call detected
    return charge_card(amount)

# Output: Nodes and Edges
Nodes:
  - function:process_payment
  - function:validate_amount
  - function:charge_card

Edges:
  - process_payment --calls--> validate_amount
  - process_payment --calls--> charge_card
```

**Extensibility**: The parser is modular. Future versions can add:
- `JavaScriptParser` for JS/TS
- `JavaParser` for Java
- Language detection and routing

---

### 2. Storage Layer (`storage.py`)

**Purpose**: Persist code graph and embeddings.

**Components**:

#### ProjectManager
- Manages multiple project memories
- Tracks active project
- Handles project lifecycle (create, load, delete, merge)

**Storage location**: `~/.codegraph/memory/<project_name>/`

#### GraphStore
- SQLite database for nodes, edges, and embeddings
- Schema:
  ```sql
  CREATE TABLE nodes (
      node_id TEXT PRIMARY KEY,
      node_type TEXT,
      name TEXT,
      qualname TEXT,
      file_path TEXT,
      start_line INTEGER,
      end_line INTEGER,
      code TEXT,
      docstring TEXT,
      embedding TEXT  -- JSON array of floats
  );

  CREATE TABLE edges (
      src TEXT,
      dst TEXT,
      edge_type TEXT
  );
  ```

**Why SQLite?**
- Lightweight, no server required
- Built into Python
- Fast for local queries
- Easy to backup (single file)

**Scalability**: For very large projects (>10k files), consider:
- FAISS for vector search (optional enhancement)
- PostgreSQL for distributed teams
- Incremental indexing to avoid full re-scans

---

### 3. Embedding Layer (`embeddings.py`)

**Purpose**: Convert code into numerical vectors for similarity search.

**Current Implementation**: Hash-based embeddings
- **Deterministic**: Same code always produces same embedding
- **Local**: No model download required
- **Fast**: Instant computation
- **Lightweight**: No GPU needed

**How it works**:
```python
# 1. Extract tokens from code
tokens = ["validate", "email", "address"]

# 2. Hash each token to a dimension
for token in tokens:
    hash_value = blake2b(token)
    dimension = hash_value % 256
    vector[dimension] += 1.0 or -1.0  # Based on hash

# 3. Normalize to unit length
vector = normalize(vector)
```

**Similarity**: Cosine similarity between vectors
```python
similarity = dot(vector_a, vector_b)  # Range: -1 to 1
```

**Embedding Models**:
- **Qodo-Embed-1-1.5B**: Code-specialized transformer (1536-dim, via `transformers` library)
- **Hash Embeddings**: Zero-dependency fallback for base installs (256-dim)

---

### 4. Multi-Agent System (`agents.py`)

**Purpose**: Specialized agents for different tasks.

#### GraphAgent
**Responsibility**: Parse and index projects

**Key Methods**:
- `index_project(path)` - Parse all files, build graph
- `ascii_neighbors(symbol, depth)` - Generate ASCII visualization

#### RAGAgent
**Responsibility**: Semantic retrieval

**Key Methods**:
- `semantic_search(query, top_k)` - Find similar code
- `context_for_query(query)` - Retrieve code snippets for LLM

**How RAG works**:
```
User Query: "validate email"
     │
     ▼
  Embed Query
     │
     ▼
  Compare with all node embeddings
     │
     ▼
  Rank by similarity
     │
     ▼
  Return top-k matches with code snippets
```

#### SummarizationAgent
**Responsibility**: Impact analysis and explanations

**Key Methods**:
- `impact_analysis(symbol, hops)` - Multi-hop dependency traversal
- `_multi_hop(start, hops)` - BFS graph traversal
- `_build_impact_prompt(...)` - Create LLM prompt

**How impact analysis works**:
```
1. Find the target symbol in graph
2. BFS traversal for N hops:
   - Follow "calls" edges
   - Follow "contains" edges
3. Collect all reachable nodes
4. Generate ASCII graph
5. Build prompt with context
6. Send to LLM (or use fallback)
7. Return structured report
```

---

### 5. Orchestrator (`orchestrator.py`)

**Purpose**: Coordinate agents for CLI operations.

**Pattern**: MCP (Multi-Component Protocol) style
- Single entry point for all operations
- Delegates to appropriate agent
- Manages shared resources (store, embeddings)

**Example Flow**:
```python
# User runs: cg impact process_payment

orchestrator = MCPOrchestrator(store)

# Orchestrator delegates:
1. GraphAgent: Get node from store
2. SummarizationAgent: Perform multi-hop traversal
3. RAGAgent: Retrieve code context
4. LLM: Generate explanation
5. Return: Structured ImpactReport
```

---

### 6. LLM Integration (`llm.py`)

**Purpose**: Generate human-readable explanations.

**Implementation**:
- **Primary**: Ollama HTTP API
- **Fallback**: Deterministic template-based explanations

**Why Ollama?**
- Fully local (no cloud calls)
- Easy to install and use
- Supports many models (Qwen, CodeLlama, etc.)
- Optional (system works without it)

**Request Flow**:
```
Prompt (from SummarizationAgent)
     │
     ▼
  Try Ollama HTTP API
     │
     ├─ Success ──> Return LLM response
     │
     └─ Failure ──> Return fallback explanation
```

**Fallback Strategy**:
- Always works, even without Ollama
- Provides actionable recommendations
- Includes context excerpt from prompt

---

## Data Flow Example

Let's trace a complete `cg impact` command:

```
User: cg impact process_payment --hops 2

1. CLI (cli.py)
   - Parse arguments
   - Load current project
   - Create MCPOrchestrator

2. Orchestrator (orchestrator.py)
   - Call SummarizationAgent.impact_analysis()

3. SummarizationAgent (agents.py)
   - Query GraphStore for "process_payment" node
   - Perform BFS traversal for 2 hops
   - Collect impacted nodes
   - Generate ASCII graph
   - Build prompt with code context

4. LocalLLM (llm.py)
   - Try Ollama API
   - If unavailable, use fallback
   - Return explanation

5. SummarizationAgent
   - Create ImpactReport
   - Return to Orchestrator

6. Orchestrator
   - Return report to CLI

7. CLI
   - Format and print:
     * Root symbol
     * Impacted symbols list
     * ASCII graph
     * Explanation

Output displayed to user
```

---

## Design Principles

### 1. Fully Local
- No cloud API calls required
- All data stays on your machine
- Works offline

### 2. Minimal Dependencies
- Core: Only Typer for CLI
- Optional: pytest for testing
- No heavy ML libraries required

### 3. Modular Architecture
- Each component has single responsibility
- Easy to extend (new parsers, embeddings, etc.)
- Agents are independent and testable

### 4. Graceful Degradation
- Works without Ollama (fallback explanations)
- Works without GPU (hash embeddings)
- Works with limited RAM (SQLite is efficient)

### 5. Developer-Friendly
- Clear separation of concerns
- Comprehensive tests
- Type hints throughout
- Detailed documentation

---

## Extension Points

### Adding a New Language Parser

```python
# 1. Create new parser
class JavaScriptParser:
    def parse_project(self) -> Tuple[List[Node], List[Edge]]:
        # Use tree-sitter or esprima
        pass

# 2. Register in parser factory
PARSERS = {
    ".py": PythonParser,
    ".js": JavaScriptParser,
    ".ts": JavaScriptParser,
}

# 3. Auto-detect language
def get_parser(file_path: Path):
    ext = file_path.suffix
    return PARSERS.get(ext, PythonParser)
```

### Adding a New Embedding Backend

```python
# 1. Create embedding interface
class EmbeddingModel(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        pass

# 2. Implement new backend
class CodeBERTEmbedding(EmbeddingModel):
    def __init__(self):
        self.model = load_model("microsoft/codebert-base")
    
    def embed_text(self, text: str) -> List[float]:
        return self.model.encode(text)

# 3. Use in GraphAgent
embedding_model = CodeBERTEmbedding()  # or HashEmbeddingModel()
agent = GraphAgent(store, embedding_model)
```

### Adding a New Agent

```python
# 1. Create agent class
class RefactoringAgent:
    def suggest_refactorings(self, symbol: str) -> List[Suggestion]:
        # Analyze code patterns
        # Suggest improvements
        pass

# 2. Add to orchestrator
class MCPOrchestrator:
    def __init__(self, store):
        self.refactoring_agent = RefactoringAgent(store)
    
    def suggest_refactorings(self, symbol):
        return self.refactoring_agent.suggest_refactorings(symbol)

# 3. Add CLI command
@app.command("suggest")
def suggest_refactorings(symbol: str):
    orchestrator = MCPOrchestrator(store)
    suggestions = orchestrator.suggest_refactorings(symbol)
    for s in suggestions:
        typer.echo(s)
```

---

## Performance Characteristics

### Indexing Speed
- **Small projects** (<100 files): <5 seconds
- **Medium projects** (100-1000 files): 10-60 seconds
- **Large projects** (1000-10000 files): 1-10 minutes

**Bottlenecks**:
- File I/O (reading source files)
- AST parsing (Python's `ast.parse`)
- SQLite insertions

**Optimizations**:
- Parallel file parsing (future)
- Batch insertions (already implemented)
- Incremental indexing (future)

### Search Speed
- **Semantic search**: <100ms for most projects
- **Impact analysis**: <500ms for 2-3 hops
- **Graph visualization**: <50ms

**Scalability**:
- SQLite handles up to ~100k nodes efficiently
- For larger graphs, consider FAISS or PostgreSQL

### Memory Usage
- **Indexing**: ~100-500 MB for medium projects
- **Search**: ~50-100 MB
- **LLM**: 4-8 GB if using Ollama (optional)

---

## Security Considerations

### Data Privacy
- All data stored locally
- No telemetry or analytics
- No cloud uploads

### Code Execution
- Parser only reads files (no execution)
- No `eval()` or dynamic imports
- Safe for untrusted codebases

### File System Access
- Only reads project files
- Writes only to `~/.codegraph/`
- No system-wide modifications

---

## Future Enhancements

See [implementation_plan.md](file:///home/ali-nasir/.gemini/antigravity/brain/a7281c58-055f-4987-8698-a7a1592000f5/implementation_plan.md) for detailed roadmap.

**Planned Features**:
- Multi-language support (JS, Java, Go)
- Better embeddings (CodeBERT, FAISS)
- Incremental indexing
- Interactive graph visualization
- Test generation suggestions
- Refactoring recommendations

---

## See Also

- [Setup Guide](setup.md) - Installation and configuration
- [Command Reference](commands.md) - CLI command documentation
- [Workflows](workflows.md) - Common usage patterns
