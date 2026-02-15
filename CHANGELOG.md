# Changelog

All notable changes to CodeGraph CLI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.1] - 2026-02-16

### Fixed
- Version consistency across package, README badge, and `__init__.py`
- Placeholder GitHub URLs in documentation
- `config.json` references corrected to `config.toml` in user guide
- GroqProvider now uses `requests` library instead of shelling out to `curl`

### Added
- GitHub Actions CI workflow (test + lint on push/PR)
- `CONTRIBUTING.md` for open-source contributors
- Rich-based terminal output in chat REPL (replaces raw ANSI escapes)

### Changed
- Architecture docs updated to reflect tree-sitter parser (replaces `ast` references)

## [2.1.0] - 2026-02-14

### Added
- **Configurable embedding models** — 5 options from zero-dependency hash to Qodo-Embed-1-1.5B
  - `cg config set-embedding hash|minilm|bge-base|jina-code|qodo-1.5b`
- Embedding config commands: `show-embedding`, `unset-embedding`
- Gemini and OpenRouter LLM provider support (6 providers total)
- Chat session export (`cg chat export <id> --format markdown|json`)
- Auto-context injection in chat (project summary + recent files)
- CrewAI session continuity across restarts (`load_session_history`)
- Watch mode for automatic re-indexing on file changes (`cg watch`)
- Project health dashboard (`cg health dashboard`)
- Command aliases: `cg find`, `cg ask`, `cg gen`, `cg fix`
- Quickstart wizard (`cg quickstart` / `cg init`)
- Change history with undo/redo (`cg undo`, `cg redo`, `cg history`)

### Changed
- Neural embedding models use raw `transformers` + `torch` — no sentence-transformers dependency
- Models cached in `~/.codegraph/models/` with first-use download from HuggingFace

## [2.0.0] - 2026-02-13

### Added
- **CrewAI multi-agent system** (`cg chat start --crew`) with 4 specialized agents:
  - Project Coordinator, File System Engineer, Senior Developer, Code Intelligence Analyst
- **11 CrewAI tools** for autonomous file operations (read, write, patch, delete, rollback)
- **Tree-sitter parsing** for Python, JavaScript, and TypeScript (replaces `ast`-only parser)
- **LanceDB vector store** for scalable embedding search
- Code generation v2 commands (`cg v2 generate`, `cg v2 review`, `cg v2 refactor`, `cg v2 test`)
- Automatic file backups before every modification (`~/.codegraph/backups/`)
- `/rollback` and `/undo` in-chat commands for CrewAI mode
- Security scanner and performance analyzer modules
- Validation engine for code proposals
- Interactive chat REPL with session persistence
- Workflow commands (`cg review-and-fix`, `cg full-analysis`)
- Cheat sheet (`cg-cheatsheet.md`)

### Changed
- Architecture refactored from single-agent to multi-agent pipeline
- LLM adapter now routes through LiteLLM for CrewAI compatibility
- Storage split into SQLite (graph) + LanceDB (vectors)
- Config stored exclusively in `~/.codegraph/config.toml`

## [1.0.0] - 2026-02-12

### Added
- Python code graph indexing with AST parsing
- Semantic search using hash-based embeddings
- Multi-hop impact analysis with dependency tracking
- ASCII dependency graph visualization
- HTML and DOT graph export
- Multi-LLM provider support (Ollama, Groq, OpenAI, Anthropic)
- Persistent project memory in SQLite
- Project management (load, unload, delete, merge)
- Comprehensive documentation (INSTALL, QUICKSTART, EXAMPLES, FAQ)
- Better error messages with fuzzy matching suggestions
- `--version` flag

### Changed
- Package renamed from `pyhelpcli` to `codegraph-cli`
- Improved README with badges and better structure

### Fixed
- Error handling for missing symbols
- LLM provider connection issues

## [0.1.0] - 2026-01-15

### Added
- Initial release
- Basic Python parsing
- Simple graph storage
- Command-line interface

[2.1.1]: https://github.com/al1-nasir/codegraph-cli/releases/tag/v2.1.1
[2.1.0]: https://github.com/al1-nasir/codegraph-cli/releases/tag/v2.1.0
[2.0.0]: https://github.com/al1-nasir/codegraph-cli/releases/tag/v2.0.0
[1.0.0]: https://github.com/al1-nasir/codegraph-cli/releases/tag/v1.0.0
[0.1.0]: https://github.com/al1-nasir/codegraph-cli/releases/tag/v0.1.0
