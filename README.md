# CodeGraph CLI

**Code intelligence from the terminal. Semantic search, impact analysis, multi-agent code generation, and conversational coding — all backed by your choice of LLM.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org)
[![Version](https://img.shields.io/badge/version-2.1.1-blue.svg)](https://github.com/al1-nasir/codegraph-cli)
[![CI](https://github.com/al1-nasir/codegraph-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/al1-nasir/codegraph-cli/actions/workflows/ci.yml)

---

## Overview

CodeGraph CLI (`cg`) parses your codebase into a semantic graph, then exposes that graph through search, impact analysis, visualization, and a conversational interface. It supports six LLM providers and optionally runs a CrewAI multi-agent system that can read, write, patch, and rollback files autonomously.

Core capabilities:

- **Semantic Search** — find code by meaning, not string matching
- **Impact Analysis** — trace multi-hop dependencies before making changes
- **Graph Visualization** — interactive HTML and Graphviz DOT exports
- **Conversational Chat** — natural language coding sessions with RAG context
- **Multi-Agent System** — CrewAI-powered agents for code generation, refactoring, and analysis
- **File Rollback** — automatic backups before every file modification

---

## Installation

```bash
pip install codegraph-cli
```

With neural embedding models (semantic code search):

```bash
pip install codegraph-cli[embeddings]
```

With CrewAI multi-agent support:

```bash
pip install codegraph-cli[crew]
```

Everything:

```bash
pip install codegraph-cli[all]
```

For development:

```bash
git clone https://github.com/al1-nasir/codegraph-cli.git
cd codegraph-cli
pip install -e ".[dev]"
```

---

## Quick Start

### 1. Configure your LLM provider

```bash
cg config setup
```

This runs an interactive wizard that writes configuration to `~/.codegraph/config.toml`. Alternatively, switch providers directly:

```bash
cg config set-llm openrouter
cg config set-llm groq
cg config set-llm ollama
```

### 2. Index a project

```bash
cg index /path/to/project --name myproject
```

This parses the source tree using tree-sitter, builds a dependency graph in SQLite, and generates embeddings for semantic search.

### 3. Use it

```bash
cg search "authentication logic"
cg impact UserService --hops 3
cg graph process_payment --depth 2
cg chat start
cg chat start --crew    # multi-agent mode
```

---

## Supported LLM Providers

| Provider | Type | Configuration |
|----------|------|---------------|
| Ollama | Local, free | `cg config set-llm ollama` |
| Groq | Cloud, free tier | `cg config set-llm groq` |
| OpenAI | Cloud | `cg config set-llm openai` |
| Anthropic | Cloud | `cg config set-llm anthropic` |
| Gemini | Cloud | `cg config set-llm gemini` |
| OpenRouter | Cloud, multi-model | `cg config set-llm openrouter` |

All configuration is stored in `~/.codegraph/config.toml`. No environment variables required.

```bash
cg config show-llm        # view current provider, model, and endpoint
cg config unset-llm       # reset to defaults
```

---

## Embedding Models

CodeGraph supports configurable embedding models for semantic code search. Choose based on your hardware and quality needs:

| Model | Download | Dim | Quality | Command |
|-------|----------|-----|---------|---------|
| hash | 0 bytes | 256 | Keyword-only | `cg config set-embedding hash` |
| minilm | ~80 MB | 384 | Decent | `cg config set-embedding minilm` |
| bge-base | ~440 MB | 768 | Good | `cg config set-embedding bge-base` |
| jina-code | ~550 MB | 768 | Code-aware | `cg config set-embedding jina-code` |
| qodo-1.5b | ~6.2 GB | 1536 | Best | `cg config set-embedding qodo-1.5b` |

The default is `hash` (zero-dependency, no download). Neural models require the `[embeddings]` extra and are downloaded on first use from HuggingFace.

```bash
cg config set-embedding jina-code    # switch to a neural model
cg config show-embedding             # view current model and all options
cg config unset-embedding            # reset to hash default
```

After changing the embedding model, re-index your project:

```bash
cg index /path/to/project
```

---

## Commands

### Project Management

```bash
cg index <path> [--name NAME]       # parse and index a codebase
cg list-projects                     # list all indexed projects
cg load-project <name>               # switch active project
cg current-project                   # print active project name
cg delete-project <name>             # remove a project index
cg merge-projects <src> <dst>        # merge two project graphs
cg unload-project                    # unload without deleting
```

### Search and Analysis

```bash
cg search <query> [--top-k N]       # semantic search across the graph
cg impact <symbol> [--hops N]       # multi-hop dependency impact analysis
cg graph <symbol> [--depth N]       # ASCII dependency graph
cg rag-context <query> [--top-k N]  # raw RAG retrieval for debugging
```

### Graph Export

```bash
cg export-graph --format html        # interactive vis.js visualization
cg export-graph --format dot         # Graphviz DOT format
cg export-graph MyClass -f html -o out.html  # focused subgraph
```

### Interactive Chat

```bash
cg chat start                        # start or resume a session
cg chat start --new                  # force a new session
cg chat start --crew                 # multi-agent mode (CrewAI)
cg chat start -s <id>                # resume a specific session
cg chat list                         # list all sessions
cg chat delete <id>                  # delete a session
```

In-chat commands:

| Command | Mode | Description |
|---------|------|-------------|
| `/help` | Both | Show available commands |
| `/clear` | Both | Clear conversation history |
| `/new` | Both | Start a fresh session |
| `/exit` | Both | Save and exit |
| `/apply` | Standard | Apply pending code proposal |
| `/preview` | Standard | Preview pending file changes |
| `/backups` | Crew | List all file backups |
| `/rollback <file>` | Crew | Restore a file from backup |
| `/undo <file>` | Crew | Alias for rollback |

### Code Generation (v2)

```bash
cg v2 generate "add a REST endpoint for user deletion"
cg v2 review src/auth.py --check-security
cg v2 refactor rename-symbol OldName NewName
cg v2 refactor extract-function target_fn 45 60
cg v2 test unit process_payment
cg v2 diagnose check src/
cg v2 diagnose fix src/auth.py
cg v2 rollback <file>
cg v2 list-backups
```

---

## Multi-Agent System

When you run `cg chat start --crew`, CodeGraph launches a CrewAI pipeline with four specialized agents:

| Agent | Role | Tools |
|-------|------|-------|
| Project Coordinator | Routes tasks to the right specialist | Delegation only |
| File System Engineer | File I/O, directory traversal, backups | list_directory, read_file, write_file, patch_file, delete_file, rollback_file, file_tree |
| Senior Software Developer | Code generation, refactoring, bug fixes | All tools (file ops + code analysis) |
| Code Intelligence Analyst | Search, dependency tracing, explanations | search_code, grep, project_summary, read_file |

Every file modification automatically creates a timestamped backup in `~/.codegraph/backups/`. Files can be rolled back to any previous state via `/rollback` or `cg v2 rollback`.

---

## Architecture

```
CLI Layer (Typer)
    |
    +-- MCPOrchestrator ----------> GraphStore (SQLite)
    |       |                           |
    |       +-- Parser (tree-sitter)    +-- VectorStore (LanceDB)
    |       +-- RAGRetriever            |
    |       +-- LLM Adapter             +-- Embeddings (configurable)
    |                                       hash | minilm | bge-base
    |                                       jina-code | qodo-1.5b
    +-- ChatAgent (standard mode)
    |
    +-- CrewChatAgent (--crew mode)
            |
            +-- Coordinator Agent
            +-- File System Agent -----> 8 file operation tools
            +-- Code Gen Agent --------> all 11 tools
            +-- Code Analysis Agent ---> 3 search/analysis tools
```

**Embeddings**: Five models available via `cg config set-embedding`. Hash (default, zero-dependency) through Qodo-Embed-1-1.5B (best quality, 6 GB). Neural models use raw `transformers` + `torch` — no sentence-transformers overhead. Models are cached in `~/.codegraph/models/`.

**Parser**: tree-sitter grammars for Python, JavaScript, and TypeScript. Extracts modules, classes, functions, imports, and call relationships into a directed graph.

**Storage**: SQLite for the code graph (nodes + edges), LanceDB for vector embeddings. All data stored under `~/.codegraph/`.

**LLM Adapter**: Unified interface across six providers. For CrewAI, models are routed through LiteLLM. Configuration is read exclusively from `~/.codegraph/config.toml`.

---

## Project Structure

```
codegraph_cli/
    cli.py               # main Typer application, all top-level commands
    cli_chat.py           # interactive chat REPL with styled output
    cli_setup.py          # setup wizard, set-llm, unset-llm, set-embedding
    cli_v2.py             # v2 code generation commands
    config.py             # loads config from TOML
    config_manager.py     # TOML read/write, provider and embedding config
    llm.py                # multi-provider LLM adapter
    parser.py             # tree-sitter AST parsing
    storage.py            # SQLite graph store
    embeddings.py         # configurable embedding engine (5 models)
    rag.py                # RAG retriever
    vector_store.py       # LanceDB vector store
    orchestrator.py       # coordinates parsing, search, impact
    graph_export.py       # DOT and HTML export
    project_context.py    # unified file access layer
    crew_tools.py         # 11 CrewAI tools (file ops + analysis)
    crew_agents.py        # 4 specialized CrewAI agents
    crew_chat.py          # CrewAI orchestrator with rollback
    chat_agent.py         # standard chat agent
    chat_session.py       # session persistence
    models.py             # core data models
    models_v2.py          # v2 models (ChatSession, CodeProposal)
    templates/
        graph_interactive.html  # vis.js graph template
```

---

## Development

```bash
git clone https://github.com/al1-nasir/codegraph-cli.git
cd codegraph-cli
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,crew,embeddings]"
pytest
```

---

## License

MIT. See [LICENSE](LICENSE).
