# Reddit Post Draft

**Subreddit:** r/Python | **Flair:** "I Made This"

---

## Title

**I built CodeGraph CLI — parses your codebase into a semantic graph with tree-sitter, does RAG-powered search over LanceDB vectors, and lets you chat with multi-agent AI from the terminal**

---

## Body

Hey r/Python 👋

I've been building **CodeGraph CLI** (`cg`) — an open-source, local-first code intelligence tool. It parses your project into an AST with tree-sitter, builds a directed dependency graph in SQLite, embeds every symbol into vectors stored in LanceDB, then layers RAG, impact analysis, and a multi-agent system on top.

**GitHub:** [https://github.com/al1-nasir/codegraph-cli](https://github.com/al1-nasir/codegraph-cli) | **PyPI:** `pip install codegraph-cli`

---

### How it works under the hood

**1. Parsing → Semantic Graph (tree-sitter + SQLite)**

When you run `cg project index ./my-project`, the parser walks every `.py`, `.js`, `.ts` file using tree-sitter grammars. Tree-sitter gives us a concrete syntax tree — it's error-tolerant, so even broken/incomplete files get parsed instead of crashing.

From the CST, we extract:

- **Nodes**: every module, class, function — with qualified names, line ranges, docstrings, and full source code
- **Edges**: imports, function calls, class inheritance — resolved into a directed graph

All of this goes into SQLite (`graph.db`) with proper indexes. Graph traversal (BFS for impact analysis, neighbor lookups) is just SQL queries.

**2. Embedding Engine (5 models, raw transformers)**

Each node gets embedded using a structured chunk that combines file path + symbol name + docstring + code body. Import lines are stripped and module-level nodes get truncated to avoid diluting embeddings with boilerplate.

```
file: src/auth.py
symbol: AuthService.validate_token
type: function
doc: Validate JWT token and return user claims.
def validate_token(self, token: str) -> dict:
    ...
```

5 embedding models available — you pick based on your hardware:

| Model | Size | Dim | Quality |
|-------|------|-----|---------|
| hash | 0 bytes | 256 | Keyword-only (BLAKE2b hash of tokens) |
| minilm | ~80 MB | 384 | Decent |
| bge-base | ~440 MB | 768 | Solid general-purpose |
| jina-code | ~550 MB | 768 | Code-aware |
| qodo-1.5b | ~6.2 GB | 1536 | Best quality |

The **hash model** is zero-dependency — it tokenizes with regex, hashes each token with BLAKE2b, and maps to a 256-dim vector. No torch, no downloads. The neural models use raw `transformers` + `torch` with configurable pooling (CLS, mean, last-token) — no `sentence-transformers` dependency. Models are cached in `~/.codegraph/models/` after first download from HuggingFace.

Each embedding model gets its own LanceDB table (`code_nodes_{model_key}`) so you can switch models without dimension mismatch crashes. If you change the embedding model, re-ingestion from SQLite happens automatically and transparently.

**3. Vector Store (LanceDB — "SQLite for vectors")**

I chose LanceDB over Chroma/FAISS because:

- **Zero-server** — embedded, just like SQLite. No Docker, no process management
- **Hybrid search** — vector similarity + SQL WHERE in one query (`file_path LIKE 'src/%'` AND semantic similarity)
- **Lance columnar format** — fast scans, efficient storage on disk
- Everything lives under `~/.codegraph/<project>/lancedb/`

Search uses cosine metric. Distance values are true cosine distances (`1 - cos_sim`), converted to similarity scores clamped to [0, 1].

**4. RAG Pipeline (Graph-Augmented Retrieval)**

This is where it gets interesting. The RAG retriever doesn't just do a basic top-k vector search:

1. **Semantic top-k** via LanceDB (or brute-force cosine fallback if LanceDB is unavailable)
2. **Graph-neighbour augmentation** — for the top 3 hits, we fetch their direct dependency neighbours from the SQLite graph (both incoming and outgoing edges) and score those neighbours against the query too. This means if you search for "authentication", you don't just get `validate_token` — you also get the caller `login_handler` and the dependency `TokenStore` that vector search alone might have missed.
3. **Minimum score threshold** — low-quality results are dropped before they reach the LLM
4. **LRU cache** (64 entries) — identical queries within a session skip re-computation
5. **Context compression** — before injecting into the LLM prompt, snippets get import lines stripped, blank lines collapsed, and long code truncated. The LLM gets clean, information-dense context instead of 500 lines of imports.

**5. Impact Analysis (Graph BFS + RAG + LLM)**

`cg analyze impact UserService --hops 3` does a multi-hop BFS traversal on the dependency graph, collects all reachable symbols, pulls RAG context for the root symbol, then sends everything to the LLM to generate a human-readable explanation of what would break.

If the symbol isn't found, it falls back to fuzzy matching via semantic search and suggests similar symbols.

**6. Multi-Agent System (CrewAI)**

`cg chat start --crew` launches 4 specialized agents via CrewAI:

| Agent | Tools | Max Iterations |
|-------|-------|---------------|
| **Coordinator** | All tools, can delegate | 25 |
| **File System Engineer** | list_directory, read_file, write_file, patch_file, delete_file, rollback_file, file_tree, backup | 15 |
| **Senior Developer** | All 11 tools (file ops + code analysis) | 20 |
| **Code Intelligence Analyst** | search_code, grep_in_project, read_file, get_project_summary | 15 |

Every file write/patch automatically creates a timestamped backup in `~/.codegraph/backups/` with JSON metadata. Rollback to any previous state with `/rollback` in chat.

The agents have detailed backstories and rules — the coordinator knows to check conversation history for follow-up requests ("apply those changes you suggested"), and the developer knows to always read the existing file before patching to match code style.

**7. LLM Adapter (6 providers, zero env vars)**

One unified interface supporting Ollama, Groq, OpenAI, Anthropic, Gemini, and OpenRouter. Each provider has its own class handling auth, payload format, and error handling. All config lives in `~/.codegraph/config.toml` — no env vars needed.

For CrewAI, models route through LiteLLM automatically.

**8. Chat with Real File Access + Symbol Memory**

The chat agent isn't just an LLM wrapper. It has:

- **Intent detection** — classifies your message (read, list, search, impact, generate, refactor, general chat) and routes to the right handler
- **Symbol memory** — tracks recently discussed symbols and files so it doesn't re-run redundant RAG queries
- **Auto-context injection** — the system prompt includes project name, indexed file count, symbol breakdown, and recently modified files so the LLM has awareness from the first message
- **Code proposals** — when you ask it to generate code, it creates a diffable proposal you can preview and apply (or reject)

---

### What you actually get as a user

```bash
pip install codegraph-cli
cg config setup                          # pick your LLM
cg project index ./my-project            # parse + build graph + embed

# Find code by meaning
cg analyze search "how does authentication work"

# Trace what breaks before you change something
cg analyze impact login_handler --hops 3

# Project health dashboard
cg analyze health

# See indexed tree with function/class breakdown
cg analyze tree --full

# Incremental sync (much faster than re-index)
cg analyze sync

# Chat with your codebase
cg chat start                            # standard mode with RAG
cg chat start --crew                     # 4-agent mode

# Visual code explorer in browser (Starlette + Uvicorn)
cg explore open

# Generate DOCX docs with Mermaid architecture diagrams
cg export docx --enhanced --include-code

# Auto-generate README from the code graph
cg onboard --save
```

### Full command structure

```
cg config    — LLM & embedding setup (6 providers, 5 embedding models)
cg project   — Index, load, and manage project memories
cg analyze   — Semantic search, impact analysis, dependency graphs, health dashboard
cg chat      — Conversational coding sessions with RAG context (+ multi-agent mode)
cg explore   — Visual code explorer that opens in your browser
cg export    — Generate DOCX documentation with architecture diagrams
cg onboard   — Auto-generate a README from your code graph
```

### Tech stack

- **CLI:** Typer + Rich (grouped command hierarchy)
- **Parsing:** tree-sitter (Python, JavaScript, TypeScript)
- **Graph storage:** SQLite (nodes + edges + metadata)
- **Vector search:** LanceDB (cosine metric, hybrid search)
- **Embeddings:** raw transformers + torch (5 models, no sentence-transformers)
- **RAG:** Graph-augmented retrieval with context compression + LRU cache
- **Browser explorer:** Starlette + Uvicorn (self-contained HTML UI)
- **Multi-agent:** CrewAI + LiteLLM (4 specialized agents, 11 tools)
- **Docs export:** python-docx + Mermaid Ink (PNG diagrams)
- **License:** MIT

### Install

```bash
pip install codegraph-cli              # core (tree-sitter + SQLite + LanceDB)
pip install codegraph-cli[embeddings]  # + neural embedding models (torch + transformers)
pip install codegraph-cli[crew]        # + CrewAI multi-agent system
pip install codegraph-cli[all]         # everything
```

Python 3.9+ | MIT license

**GitHub:** [https://github.com/al1-nasir/codegraph-cli](https://github.com/al1-nasir/codegraph-cli) | **PyPI:** [https://pypi.org/project/codegraph-cli/](https://pypi.org/project/codegraph-cli/)

---

Would love technical feedback on:

1. The graph-augmented RAG approach — is augmenting with dependency neighbours actually useful for code search, or just noise?
2. LanceDB vs FAISS/Chroma for this use case — anyone have strong opinions?
3. What languages should be next? (Go, Rust, Java grammars exist for tree-sitter)
4. Is the multi-agent approach actually useful vs. a single well-prompted agent?

Thanks for reading! Happy to deep-dive into any of the internals. ⭐ if you find it useful!

---

## Cross-post suggestions

| Subreddit | Angle |
|-----------|-------|
| **r/Python** | Primary — use "I Made This" flair |
| **r/LocalLLaMA** | Lead with Ollama/local-first + RAG pipeline |
| **r/MachineLearning** | Lead with graph-augmented RAG + embedding engine |
| **r/commandline** | Focus on the CLI UX and grouped commands |
| **r/opensource** | MIT project showcase |
