# Setup Guide

## Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: Minimum 8 GB RAM (16-32 GB recommended for large projects)
- **Optional**: Ollama for local LLM reasoning (recommended but not required)

## Installation

### Option 1: Install from Source (Development)

```bash
# Clone the repository
git clone https://github.com/yourusername/CodeGraph-CLI.git
cd CodeGraph-CLI

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode
pip install -e .

# Install development dependencies (for testing)
pip install -e ".[dev]"
```

### Option 2: Install from PyPI (Coming Soon)

```bash
pip install codegraph-cli
```

## Verify Installation

```bash
# Check that the CLI is available
cg --help

# Or use direct module invocation
python -m codegraph_cli.cli --help
```

You should see the CodeGraph CLI help message with all available commands.

## Optional: Setup Ollama for LLM Reasoning

CodeGraph CLI works perfectly fine without Ollama, using deterministic fallback explanations. However, for richer impact analysis and code explanations, you can install Ollama:

### Install Ollama

```bash
# Linux
curl https://ollama.ai/install.sh | sh

# macOS
brew install ollama

# Windows
# Download from https://ollama.ai/download
```

### Start Ollama Service

```bash
ollama serve
```

### Pull a Code Model

```bash
# Recommended: Qwen 2.5 Coder (7B parameters, good balance)
ollama pull qwen2.5-coder:7b

# Alternative: CodeLlama
ollama pull codellama:7b

# For better quality (requires more RAM/GPU):
ollama pull qwen2.5-coder:14b
```

### Verify Ollama is Running

```bash
curl http://127.0.0.1:11434/api/tags
```

## Configuration

### Environment Variables

```bash
# Optional: Set custom memory location
export CODEGRAPH_HOME="$HOME/.codegraph"

# Optional: Set custom LLM endpoint
export CODEGRAPH_LLM_ENDPOINT="http://127.0.0.1:11434/api/generate"
```

### Memory Storage Location

By default, CodeGraph stores project memories in:
- Linux/macOS: `~/.codegraph/memory/`
- Windows: `%USERPROFILE%\.codegraph\memory\`

Each project gets its own subdirectory with:
- `graph.db` - SQLite database with nodes, edges, and embeddings
- `project.json` - Project metadata

## Quick Start

### 1. Index Your First Project

```bash
# Index a Python project
cg index /path/to/your/project --name MyProject

# Example output:
# Indexed '/path/to/your/project' as project 'MyProject'.
# Nodes: 245 | Edges: 512
```

### 2. Search for Code

```bash
# Semantic search
cg search "database connection"

# Example output:
# [function] db.connect  score=0.842
#   db/connection.py:15-32
#   def connect(host: str, port: int) -> Connection:
```

### 3. Analyze Impact

```bash
# See what would be affected by changing a function
cg impact process_payment --hops 2

# Shows:
# - Impacted symbols
# - ASCII dependency graph
# - LLM-generated explanation of risks
```

### 4. Visualize Dependencies

```bash
# ASCII graph in terminal
cg graph UserService --depth 2

# Export to HTML for interactive viewing
cg export-graph UserService --format html --output graph.html
```

## Troubleshooting

### Command Not Found: `cg`

If `cg` command is not found after installation:

```bash
# Option 1: Use direct module invocation
python -m codegraph_cli.cli --help

# Option 2: Add to PATH (if using venv)
export PATH="$PATH:$(pwd)/.venv/bin"

# Option 3: Reinstall in editable mode
pip install -e .
```

### Ollama Connection Failed

If you see "Local LLM endpoint was unavailable":

1. **Check if Ollama is running**:
   ```bash
   curl http://127.0.0.1:11434/api/tags
   ```

2. **Start Ollama**:
   ```bash
   ollama serve
   ```

3. **Verify model is pulled**:
   ```bash
   ollama list
   ```

**Note**: CodeGraph works without Ollama, using fallback explanations.

### Memory/Performance Issues

For large projects (>1000 files):

1. **Increase system resources**: Use a machine with more RAM
2. **Index incrementally**: Index subdirectories separately
3. **Exclude directories**: Skip test files, vendor code, etc.

### Permission Errors

If you get permission errors writing to `~/.codegraph`:

```bash
# Use a local directory instead
export CODEGRAPH_HOME="$(pwd)/.codegraph_local"
```

## Next Steps

- Read the [Command Reference](commands.md) for detailed command usage
- Check out [Workflows](workflows.md) for common use cases
- See [Architecture](architecture.md) to understand how it works

## Getting Help

- Run `cg COMMAND --help` for command-specific help
- Check the [GitHub Issues](https://github.com/yourusername/CodeGraph-CLI/issues)
- Read the full documentation in the `docs/` directory
