# CodeGraph CLI User Guide

A comprehensive guide to using all features of CodeGraph CLI - your local-first code intelligence assistant with AI-powered analysis.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Setup & Configuration](#setup--configuration)
3. [Project Management](#project-management)
4. [Code Search & Discovery](#code-search--discovery)
5. [Impact Analysis](#impact-analysis)
6. [Interactive Chat Mode](#interactive-chat-mode)
7. [Code Generation (v2)](#code-generation-v2)
8. [Refactoring Operations](#refactoring-operations)
9. [Test Generation](#test-generation)
10. [Diagnostics & Error Fixing](#diagnostics--error-fixing)
11. [Graph Visualization](#graph-visualization)
12. [Tips & Best Practices](#tips--best-practices)

---

## Getting Started

### Installation

```bash
# Install via pip
pip install codegraph-cli

# Or install from source
git clone https://github.com/al1-nasir/codegraph-cli
cd codegraph-cli
pip install -e .
```

### Verify Installation

```bash
cg --version
cg --help
```

---

## Setup & Configuration

### Interactive Setup Wizard

The easiest way to configure CodeGraph CLI is using the interactive setup wizard:

```bash
cg config setup
```

This will guide you through:

1. **Selecting an LLM Provider:**
   - **Ollama** (local, free) - Recommended for privacy
   - **Groq** (cloud, fast, free tier) - Great performance
   - **OpenAI** (cloud, paid) - GPT-4 models
   - **Anthropic** (cloud, paid) - Claude models

2. **Configuring the Provider:**
   - For Ollama: Select endpoint and available models
   - For cloud providers: Enter API key and select model

3. **Saving Configuration:**
   - Configuration is saved to `~/.codegraph/config.toml`

### Manual Configuration

You can also specify LLM settings per command:

```bash
# Use specific provider and model
cg index ./myproject --llm-provider groq --llm-model llama-3.3-70b-versatile

# With API key for cloud providers
cg impact my_function --llm-provider openai --llm-api-key sk-xxx
```

### Supported LLM Providers

| Provider | Type | Models | API Key Required |
|----------|------|--------|------------------|
| Ollama | Local | qwen2.5-coder, codellama, etc. | No |
| Groq | Cloud | llama-3.3-70b-versatile, mixtral-8x7b | Yes (free tier) |
| OpenAI | Cloud | gpt-4, gpt-4-turbo, gpt-3.5-turbo | Yes (paid) |
| Anthropic | Cloud | claude-3-5-sonnet, claude-3-opus | Yes (paid) |

---

## Project Management

### Index a Project

Before using most features, index your project to build the semantic code graph:

```bash
# Index current directory
cg index .

# Index with custom name
cg index /path/to/project --name MyProject

# Index with specific LLM for reasoning
cg index ./myproject --llm-model qwen2.5-coder:7b
```

**Output:**
```
Indexed '/path/to/project' as project 'MyProject'.
Nodes: 245 | Edges: 512
```

### List Projects

View all indexed projects:

```bash
cg list-projects
```

**Output:**
```
* MyProject      # * indicates currently loaded project
  OtherProject
  TestProject
```

### Load a Project

Switch between indexed projects:

```bash
cg load-project MyProject
```

### Check Current Project

```bash
cg current-project
```

### Unload Project

Unload without deleting data:

```bash
cg unload-project
```

### Delete Project

Permanently remove a project from memory:

```bash
cg delete-project OldProject
```

⚠️ **Warning:** This permanently deletes all indexed data for the project.

### Merge Projects

Combine two project memories:

```bash
cg merge-projects Frontend FullStack
```

---

## Code Search & Discovery

### Semantic Search

Search your codebase using natural language:

```bash
# Find database-related code
cg search "database connection"

# Find authentication functions
cg search "user authentication" --top-k 10

# Find error handling patterns
cg search "exception handling"

# Find API endpoints
cg search "REST API endpoints"
```

**Options:**
- `--top-k INTEGER` - Maximum number of results (default: 5, range: 1-30)

**Output:**
```
[function] db.connect  score=0.842
  db/connection.py:15-32
  def connect(host: str, port: int) -> Connection:

[class] DatabaseManager  score=0.756
  db/manager.py:8-45
  class DatabaseManager:
```

### RAG Context Retrieval

Retrieve semantic code context for debugging:

```bash
# Get context about payment processing
cg rag-context "payment processing logic"

# Get more context snippets
cg rag-context "database queries" --top-k 10
```

---

## Impact Analysis

### Analyze Change Impact

Understand what will be affected before making changes:

```bash
# Analyze impact of changing a function
cg impact process_payment

# Deep analysis with more hops
cg impact UserService --hops 3

# Without ASCII graph
cg impact calculate_total --no-graph
```

**Options:**
- `--hops INTEGER` - Dependency traversal depth (default: 2, range: 1-6)
- `--show-graph / --no-graph` - Include ASCII graph (default: show)

#### How Impact Analysis Works

The `cg impact` command uses a **hybrid approach**:

1. **Graph Traversal (No LLM)**: First, it performs multi-hop graph traversal using BFS (Breadth-First Search) to find all dependent symbols. This uses the pre-built code graph stored during indexing.

2. **LLM Explanation**: Then, it uses the configured LLM to generate a human-readable explanation of the impact, including:
   - Main risks of making the change
   - Most likely breakpoints
   - Test recommendations

**Data Flow:**
```
cg impact <symbol>
    │
    ├─► Graph Store (SQLite) ─► Multi-hop BFS traversal ─► Find impacted nodes
    │                                                      │
    │                                                      ▼
    └─► LLM (Ollama/Groq/OpenAI/Anthropic) ◄──────── Build prompt with:
           │                                              - Root symbol
           │                                              - Impacted symbols
           ▼                                              - Dependency graph
        Generate explanation
        (risks, breakpoints, test recommendations)
```

**Note:** Embeddings are **NOT** used during impact analysis. Embeddings are only used during:
- **Indexing**: To create vector embeddings for each code node
- **Semantic Search** (`cg search`): To find similar code based on meaning

**Output:**
```
Root: processor.OrderProcessor.create_order
Impacted symbols:
- models.Order.total
- models.Order
- processor.UserProcessor.get_user

ASCII graph:
processor.OrderProcessor.create_order
  |- calls -> processor.UserProcessor.get_user
  |- calls -> models.Order

Explanation:
Main risks:
- Changing create_order() may affect order validation logic
- Dependencies on User model could break if signature changes

Test recommendations:
- Add unit tests for order creation edge cases
- Test user validation scenarios
```

### View Dependency Graph

Show ASCII dependency graph around a symbol:

```bash
# Show dependencies of a class
cg graph UserProcessor

# Deep graph with 3 levels
cg graph PaymentService --depth 3
```

**Options:**
- `--depth INTEGER` - Traversal depth (default: 2, range: 1-6)

---

## Interactive Chat Mode

Start an interactive chat session with your codebase:

### Start Chat Session

```bash
# Start new chat session
cg chat start

# Resume specific session
cg chat start --session abc123

# Use CrewAI multi-agent system (experimental)
cg chat start --crew
```

**Options:**
- `--session, -s TEXT` - Resume specific session ID
- `--llm-model TEXT` - LLM model to use
- `--llm-provider TEXT` - LLM provider
- `--crew` - Use CrewAI multi-agent system (experimental)

### Chat Commands

Inside the chat REPL:

| Command | Description |
|---------|-------------|
| `/exit` | Exit chat session |
| `/help` | Show available commands |
| `/clear` | Clear conversation history |
| `/apply` | Apply pending proposal (non-crew mode) |
| `/preview` | Preview pending changes (non-crew mode) |

### Example Chat Session

```
💬 CodeGraph Chat Mode
============================================================
Ask me to search code, analyze impact, generate features, or refactor.

Special commands:
  /exit     - Exit chat
  /help     - Show help
  /apply    - Apply pending proposal
  /preview  - Preview pending changes
  /clear    - Clear conversation history
============================================================

You: Find all functions related to payment processing

Assistant: I found 5 functions related to payment processing:

1. `process_payment()` in payment/processor.py:45
   - Main payment processing function
   - Handles credit card and PayPal payments

2. `validate_payment()` in payment/validator.py:12
   - Validates payment details
   - Checks card number format and expiry

...

You: What would happen if I change the signature of process_payment?

Assistant: Let me analyze the impact of changing `process_payment()`...

[Provides detailed impact analysis]
```

### List Chat Sessions

```bash
# List all sessions
cg chat list

# Filter by project
cg chat list --project MyProject
```

### Delete Chat Session

```bash
cg chat delete abc123
```

---

## Code Generation (v2)

Generate code from natural language descriptions:

### Generate Code

```bash
# Generate code from description
cg v2 generate "Create a function to validate email addresses"

# Generate with context file
cg v2 generate "Add error handling" --file src/processor.py

# Preview only (don't apply)
cg v2 generate "Add logging" --preview

# Auto-apply without confirmation
cg v2 generate "Add docstrings" --auto-apply

# Specify output file
cg v2 generate "Create utility functions" --output src/utils.py
```

**Options:**
- `--file, -f TEXT` - File to use as context
- `--output, -o TEXT` - Output file path or directory
- `--preview, -p` - Preview changes without applying
- `--auto-apply, -y` - Apply changes without confirmation

### Code Review

Run AI-powered code review:

```bash
# Full review
cg v2 review src/processor.py

# Check specific issues
cg v2 review src/processor.py --check bugs
cg v2 review src/processor.py --check security
cg v2 review src/processor.py --check performance

# Filter by severity
cg v2 review src/processor.py --severity high

# With LLM analysis
cg v2 review src/processor.py --llm

# Show auto-fix suggestions
cg v2 review src/processor.py --fix
```

**Options:**
- `--check TEXT` - Check type: bugs, security, performance, all (default: all)
- `--severity TEXT` - Minimum severity: low, medium, high, critical
- `--verbose, -v` - Show detailed output
- `--llm` - Use LLM for deeper analysis
- `--fix` - Show auto-fix suggestions

### Rollback Changes

```bash
# Rollback to a previous backup
cg v2 rollback backup_123

# List available backups
cg v2 list-backups
```

---

## Refactoring Operations

Safe refactoring with automatic dependency tracking:

### Rename Symbol

Rename a function/class and update all references:

```bash
# Rename with preview
cg v2 refactor rename old_name new_name --preview

# Auto-apply rename
cg v2 refactor rename old_name new_name --auto-apply
```

### Extract Function

Extract code into a new function:

```bash
# Extract lines 50-75 into a new function
cg v2 refactor extract-function src/processor.py 50 75 process_data --preview

# Auto-apply
cg v2 refactor extract-function src/processor.py 50 75 process_data --auto-apply
```

### Extract Service

Extract multiple functions to a new service file:

```bash
# Extract functions to new service
cg v2 refactor extract-service func1 func2 func3 --target services/new_service.py --preview

# Auto-apply
cg v2 refactor extract-service func1 func2 --target services/api.py --auto-apply
```

---

## Test Generation

Generate tests from your code graph:

### Generate Unit Tests

```bash
# Generate tests for a function
cg v2 test unit process_payment

# Save to file
cg v2 test unit process_payment --output tests/test_payment.py
```

### Generate Integration Tests

```bash
# Generate integration tests for a user flow
cg v2 test integration "user registration and login"

# Save to file
cg v2 test integration "checkout flow" --output tests/test_checkout.py
```

### Coverage Prediction

```bash
# Predict coverage impact
cg v2 test coverage process_payment
```

**Output:**
```
📊 Coverage Analysis:
   Current coverage: 45.2%
   Estimated after tests: 62.8%
   Increase: +17.6%
   Tests to generate: 5
   Functions covered: 3
```

---

## Diagnostics & Error Fixing

### Check for Errors

Scan project for syntax errors:

```bash
# Check current directory
cg v2 diagnose check

# Check specific path
cg v2 diagnose check ./src
```

**Output:**
```
📋 Found 3 error(s):

1. src/processor.py:45
   SyntaxError: invalid syntax

2. src/utils.py:102
   IndentationError: expected an indented block
```

### Auto-Fix Errors

Automatically fix common syntax errors:

```bash
# Preview fixes
cg v2 diagnose fix --preview

# Auto-apply fixes
cg v2 diagnose fix --auto-apply
```

---

## Graph Visualization

### Export Graph

Export dependency graph to HTML or DOT format:

```bash
# Export full project graph as HTML
cg export-graph --format html

# Export subgraph around a symbol
cg export-graph PaymentService --format html --output payment.html

# Export as Graphviz DOT
cg export-graph --format dot --output graph.dot
```

**Options:**
- `--format, -f TEXT` - Export format: `html` or `dot` (default: `html`)
- `--output, -o PATH` - Output file path

### Render DOT Files

```bash
# Render DOT to PNG
dot -Tpng graph.dot -o graph.png

# Render DOT to SVG
dot -Tsvg graph.dot -o graph.svg
```

---

## Tips & Best Practices

### When Are Embeddings vs LLM Used?

Understanding when embeddings and LLM are used helps you optimize performance and costs:

| Command | Embeddings | LLM | Description |
|---------|------------|-----|-------------|
| `cg index` | ✅ Yes | ❌ No | Creates vector embeddings for each code node |
| `cg search` | ✅ Yes | ❌ No | Semantic similarity search using embeddings |
| `cg rag-context` | ✅ Yes | ❌ No | Retrieves context using embeddings |
| `cg impact` | ❌ No | ✅ Yes | Graph traversal + LLM explanation |
| `cg graph` | ❌ No | ❌ No | Pure graph traversal (no AI) |
| `cg chat start` | ✅ Yes | ✅ Yes | Both embeddings (search) and LLM (response) |
| `cg v2 generate` | ❌ No | ✅ Yes | LLM generates code |
| `cg v2 review` | ❌ No | ✅ Yes* | LLM analysis (*optional with `--llm`) |
| `cg v2 refactor` | ❌ No | ❌ No | Pure graph-based refactoring |
| `cg v2 test` | ❌ No | ✅ Yes | LLM generates test code |
| `cg v2 diagnose` | ❌ No | ❌ No | Static analysis (no AI) |

**Key Points:**
- **Embeddings** are computed once during indexing and stored locally
- **LLM** is called only when generating explanations, code, or tests
- **Graph operations** (traversal, refactoring) don't require AI - they use the stored graph structure

### Workflow Recommendations

1. **Start with Setup:**
   ```bash
   cg config setup  # Configure your LLM provider
   ```

2. **Index Your Project:**
   ```bash
   cg index . --name MyProject
   ```

3. **Explore with Search:**
   ```bash
   cg search "main entry point"
   ```

4. **Analyze Before Changes:**
   ```bash
   cg impact function_name --hops 3
   ```

5. **Use Chat for Complex Tasks:**
   ```bash
   cg chat start
   ```

### Chaining Commands

```bash
# Index and immediately search
cg index ~/myproject --name MyProj && cg search "authentication"

# Switch project and analyze
cg load-project API && cg impact handle_request

# Export and render
cg export-graph --format dot && dot -Tpng project_graph.dot -o graph.png
```

### Using with Git Hooks

```bash
# Pre-commit check for errors
cg v2 diagnose check

# Generate tests for new functions
cg v2 test unit new_function --output tests/test_new.py
```

### Performance Tips

- Use `--hops 1` or `--hops 2` for faster impact analysis
- Use `--top-k 3` for quicker searches when you know what you're looking for
- Index smaller subdirectories for large monorepos

### Keyboard Shortcuts in Chat

| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Exit chat (saves session) |
| `Ctrl+D` | Exit chat |
| `Up/Down` | Navigate command history |

---

## Command Quick Reference

| Command | Description |
|---------|-------------|
| `cg config setup` | Interactive LLM configuration |
| `cg index <path>` | Index a project |
| `cg list-projects` | List all projects |
| `cg load-project <name>` | Load a project |
| `cg current-project` | Show current project |
| `cg search <query>` | Semantic code search |
| `cg impact <symbol>` | Impact analysis |
| `cg graph <symbol>` | Show dependency graph |
| `cg export-graph` | Export graph to HTML/DOT |
| `cg chat start` | Start interactive chat |
| `cg chat list` | List chat sessions |
| `cg v2 generate <prompt>` | Generate code |
| `cg v2 review <file>` | Code review |
| `cg v2 refactor rename <old> <new>` | Rename symbol |
| `cg v2 refactor extract-function` | Extract function |
| `cg v2 test unit <symbol>` | Generate unit tests |
| `cg v2 diagnose check` | Check for errors |
| `cg v2 diagnose fix` | Auto-fix errors |
| `cg v2 rollback <backup>` | Rollback changes |
| `cg v2 list-backups` | List backups |

---

## Getting Help

- Use `cg --help` for global options
- Use `cg <command> --help` for command-specific help
- Use `cg <group> <command> --help` for grouped commands

```bash
cg --help
cg index --help
cg chat start --help
cg v2 generate --help
cg v2 refactor rename --help
```

---

## Troubleshooting

### No Project Loaded

```
❌ No project loaded. Use 'cg load-project <name>' first.
```

**Solution:** Index or load a project first:
```bash
cg index .  # or
cg load-project MyProject
```

### Symbol Not Found

```
❌ Symbol 'xyz' not found in current project.
```

**Solution:** Use search to find the correct symbol name:
```bash
cg search "xyz"
```

### LLM Connection Issues

```
❌ Cannot connect to Ollama!
```

**Solution:** Ensure Ollama is running:
```bash
ollama serve
```

### API Key Issues

```
❌ API key validation failed
```

**Solution:** Run setup again with correct API key:
```bash
cg config setup
```

---

## License

CodeGraph CLI is open source. See [LICENSE](LICENSE) for details.
