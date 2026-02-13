# CodeGraph CLI

<div align="center">

**Local-first AI coding assistant with semantic search, impact analysis, and interactive chat**

[![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)](https://github.com/yourusername/codegraph-cli)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Quick Start](#-quick-start) • [Features](#-features) • [Installation](INSTALL.md) • [Examples](EXAMPLES.md) • [FAQ](FAQ.md)

</div>

---

## 🎯 What is CodeGraph?

CodeGraph CLI is a **fully local, open-source** AI coding assistant that understands your codebase through semantic analysis. Think of it as having an expert developer who knows every line of your code, can predict what breaks before you change it, and helps you code through natural conversation.

**The only coding assistant that understands what breaks before you break it.**

---

## ✨ Features

### 🤖 Interactive Chat Mode (NEW!)
**Conversational coding assistance with smart RAG-based context**

```bash
cg chat start
```

- Natural language coding sessions
- Search, analyze, generate, and refactor through conversation
- Smart context management (70% fewer tokens than naive approaches)
- ChromaDB-powered fast retrieval (10-100x faster)
- Session persistence - resume anytime
- Preview changes before applying

**Example conversation:**
```
You: Find the authentication code
You: Add a password reset endpoint
You: /preview
You: /apply
```

### 🔍 Semantic Search
**Find code by meaning, not exact matches**

```bash
cg search "database writes"
cg search "payment processing" --top-k 10
```

- Hash-based embeddings (no heavy ML models)
- ChromaDB vector store for blazing fast search
- Filter by node type (function, class, module)
- Relevance scoring

### 📊 Impact Analysis
**See what breaks BEFORE you change it**

```bash
cg impact process_payment --hops 2
cg impact UserService --hops 3 --llm-provider groq
```

- Multi-hop dependency traversal
- AI-powered explanations
- ASCII dependency graphs
- Identifies affected tests and integration points

### 🗺️ Dependency Graphs
**Visualize how your code connects**

```bash
cg graph UserService --depth 2
cg export-graph --format html --output graph.html
cg export-graph MyClass --format dot
```

- Interactive HTML visualizations
- Graphviz DOT export
- Focus on specific symbols
- Full project or local subgraphs

### 🛠️ Code Generation & Refactoring (v2.0)
**AI-powered code transformations**

```bash
# Generate code
cg v2 generate "Add password reset endpoint"

# Refactor operations
cg v2 refactor rename-symbol OldName NewName
cg v2 refactor extract-function process_payment 45 60
cg v2 refactor extract-service payment_functions

# Code review
cg v2 review src/auth.py --check-security --check-performance

# Test generation
cg v2 test unit process_payment
cg v2 test integration "user registration flow"

# Error fixing
cg v2 diagnose check src/
cg v2 diagnose fix src/auth.py
```

### 🤖 Multi-LLM Support
**Choose your AI provider**

- **Ollama** (local, private, free) - Default
- **Groq** (fast, cloud, free tier)
- **OpenAI** (GPT-4, GPT-3.5)
- **Anthropic** (Claude)

```bash
# Configure provider
export LLM_PROVIDER=groq
export LLM_API_KEY=your_key_here
export LLM_MODEL=llama-3.3-70b-versatile

# Or use flags
cg impact MyClass --llm-provider groq --llm-api-key $GROQ_KEY
```

### 🔒 Privacy & Performance
- **100% Local Option** - Works offline with Ollama
- **Your code never leaves your machine** (with local LLM)
- **Fast** - Hash-based embeddings, ChromaDB vector store
- **Persistent Memory** - Index once, query forever
- **SQLite + ChromaDB** - Lightweight, no database server needed

---

## 🚀 Quick Start

### 1. Install

```bash
pip install codegraph-cli
```

**Optional: Install ChromaDB for 10-100x faster search**
```bash
pip install chromadb
```

See [INSTALL.md](INSTALL.md) for detailed instructions and LLM setup.

### 2. Index Your Project

```bash
cg index /path/to/your/project --name MyProject
```

This creates a semantic graph of your codebase with:
- All modules, classes, and functions
- Dependencies and call relationships
- Embeddings for semantic search
- ChromaDB vector store (if installed)

### 3. Start Coding!

#### Interactive Chat
```bash
cg chat start

You: Find the authentication code
Assistant: Found 5 results:
1. [function] auth.login_endpoint
   Location: src/auth.py:45
   Relevance: 0.85

You: Add a password reset endpoint
Assistant: I've created a code proposal: Add password reset endpoint
Files to change: 2
  - New files: 1
  - Modified files: 1

To apply these changes, say 'apply' or '/apply'.

You: /apply
Assistant: ✅ Successfully applied changes to 2 file(s).
```

#### Semantic Search
```bash
cg search "database writes"
# [function] save_user  score=0.823
#   src/models.py:45-67
#   def save_user(user_data): ...
```

#### Impact Analysis
```bash
cg impact process_payment --hops 2
# Root: process_payment
# Impacted symbols:
# - checkout_handler
# - order_service
# - payment_webhook
#
# Explanation:
# Changing process_payment will affect 3 downstream functions...
```

#### Dependency Graph
```bash
cg graph UserService --depth 2
# UserService
# ├── AuthService
# │   └── TokenManager
# └── DatabaseService
#     └── ConnectionPool
```

---

## 📚 Complete Command Reference

### Project Management
```bash
cg index <path> [--name NAME]          # Index a project
cg list-projects                        # List all indexed projects
cg load-project <name>                  # Switch active project
cg current-project                      # Show active project
cg delete-project <name>                # Delete project index
cg merge-projects <source> <target>     # Merge two projects
```

### Interactive Chat
```bash
cg chat start                           # Start/resume chat
cg chat start --new                     # Force new session
cg chat start --session <id>            # Resume specific session
cg chat list                            # List all sessions
cg chat list --project MyProject        # Filter by project
cg chat delete <session-id>             # Delete a session
```

**In-Chat Commands:**
- `/exit` - Exit and save session
- `/help` - Show help
- `/apply` - Apply pending code proposal
- `/preview` - Preview pending changes
- `/clear` - Clear conversation history

### Search & Analysis
```bash
cg search <query> [--top-k N]           # Semantic search
cg impact <symbol> [--hops N]           # Impact analysis
cg graph <symbol> [--depth N]           # Dependency graph
cg rag-context <query> [--top-k N]      # Raw RAG context
```

### Visualization
```bash
cg export-graph [--format html|dot]     # Export full graph
cg export-graph <symbol> [--format]     # Export local subgraph
cg export-graph --output graph.html     # Custom output path
```

### Code Generation (v2.0)
```bash
cg v2 generate <prompt>                 # Generate code
cg v2 generate <prompt> --max-files N   # Limit files changed
```

### Refactoring (v2.0)
```bash
cg v2 refactor rename-symbol <old> <new>
cg v2 refactor extract-function <symbol> <start> <end>
cg v2 refactor extract-service <pattern>
```

### Code Review (v2.0)
```bash
cg v2 review <file>                     # Full review
cg v2 review <file> --check-security    # Security scan
cg v2 review <file> --check-performance # Performance analysis
cg v2 review <file> --check-bugs        # Bug detection
```

### Testing (v2.0)
```bash
cg v2 test unit <symbol>                # Generate unit tests
cg v2 test integration <description>    # Generate integration tests
cg v2 test coverage <symbol>            # Predict coverage impact
```

### Diagnostics (v2.0)
```bash
cg v2 diagnose check <path>             # Check for errors
cg v2 diagnose fix <file>               # Auto-fix errors
```

---

## 💡 Use Cases

### 1. Understanding Unfamiliar Code
```bash
# "What does this function do?"
cg search "payment processing"
cg impact process_payment --hops 2

# Or use chat
cg chat start
You: What does process_payment do?
```

### 2. Safe Refactoring
```bash
# Before changing anything
cg impact UserService --hops 3

# See what breaks
# Root: UserService
# Impacted: AuthController, ProfileView, AdminDashboard

# Now refactor safely
cg v2 refactor rename-symbol UserService UserManager
```

### 3. Code Review
```bash
# Review a pull request
cg v2 review src/new_feature.py --check-security --check-bugs

# Or in chat
You: Review the authentication changes
You: Are there any security issues?
```

### 4. Feature Development
```bash
# Interactive development
cg chat start

You: Add a password reset endpoint
Assistant: [Creates proposal]

You: /preview
Assistant: [Shows diff]

You: /apply
Assistant: ✅ Applied changes
```

### 5. Finding Dead Code
```bash
# Find unused functions
cg search "legacy" --top-k 20
cg impact old_function --hops 1

# If no impacted symbols → safe to delete
```

---

## 🏗️ Architecture

CodeGraph uses a **multi-agent architecture**:

```
┌─────────────────────────────────────────┐
│         CLI Interface                    │
│  (Typer-based commands + Chat REPL)     │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│      MCPOrchestrator                     │
│  (Coordinates all agents)                │
└──┬───────┬──────────┬──────────┬────────┘
   │       │          │          │
┌──▼──┐ ┌─▼───┐  ┌───▼───┐  ┌──▼─────┐
│Graph│ │ RAG │  │Codegen│  │Refactor│
│Agent│ │Agent│  │ Agent │  │ Agent  │
└──┬──┘ └─┬───┘  └───┬───┘  └──┬─────┘
   │      │          │         │
┌──▼──────▼──────────▼─────────▼────────┐
│         GraphStore (SQLite)            │
│  + VectorStore (ChromaDB - optional)   │
└────────────────────────────────────────┘
```

**Key Components:**
- **Parser** - AST-based code analysis (Python, JS, TS, Go, Java, C++)
- **GraphStore** - SQLite for code graph + ChromaDB for vectors
- **Embeddings** - Fast hash-based semantic embeddings
- **RAGAgent** - Semantic search with ChromaDB (10-100x faster)
- **ChatAgent** - Smart context management with RAG
- **CodeGenAgent** - AI-powered code generation
- **RefactorAgent** - Safe code transformations

See [docs/architecture.md](docs/architecture.md) for details.

---

## 🔧 Configuration

### Environment Variables

```bash
# LLM Provider (default: ollama)
export LLM_PROVIDER=groq              # ollama, groq, openai, anthropic
export LLM_MODEL=llama-3.3-70b-versatile
export LLM_API_KEY=your_key_here
export LLM_ENDPOINT=http://localhost:11434/api/generate  # For Ollama

# Storage (default: ~/.codegraph/)
export CODEGRAPH_HOME=/custom/path
```

### LLM Provider Setup

**Ollama (Local, Free)**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull qwen2.5-coder:7b

# Use with CodeGraph
export LLM_PROVIDER=ollama
```

**Groq (Cloud, Fast, Free Tier)**
```bash
# Get API key from https://console.groq.com
export LLM_PROVIDER=groq
export LLM_API_KEY=gsk_...
export LLM_MODEL=llama-3.3-70b-versatile
```

**OpenAI**
```bash
export LLM_PROVIDER=openai
export LLM_API_KEY=sk-...
export LLM_MODEL=gpt-4
```

**Anthropic**
```bash
export LLM_PROVIDER=anthropic
export LLM_API_KEY=sk-ant-...
export LLM_MODEL=claude-3-5-sonnet-20241022
```

---

## 📖 Documentation

- **[INSTALL.md](INSTALL.md)** - Installation guide with LLM setup
- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute tutorial
- **[EXAMPLES.md](EXAMPLES.md)** - Real-world use cases
- **[FAQ.md](FAQ.md)** - Common questions and troubleshooting
- **[docs/architecture.md](docs/architecture.md)** - Technical architecture
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guide

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas we'd love help with:**
- Additional language parsers (Rust, Ruby, PHP, etc.)
- UI improvements for graph visualization
- Performance optimizations
- Documentation and examples
- Bug reports and feature requests

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Built with:
- [Tree-sitter](https://tree-sitter.github.io/) for parsing
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Typer](https://typer.tiangolo.com/) for CLI
- [Ollama](https://ollama.com/), [Groq](https://groq.com/), [OpenAI](https://openai.com/), [Anthropic](https://anthropic.com/) for LLMs

---

<div align="center">

**Made with ❤️ for developers who care about code quality**

[⭐ Star us on GitHub](https://github.com/yourusername/codegraph-cli) • [🐛 Report a bug](https://github.com/yourusername/codegraph-cli/issues) • [💡 Request a feature](https://github.com/yourusername/codegraph-cli/issues)

</div>
