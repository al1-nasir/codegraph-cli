# Changelog

All notable changes to CodeGraph CLI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[1.0.0]: https://github.com/yourusername/codegraph-cli/releases/tag/v1.0.0
[0.1.0]: https://github.com/yourusername/codegraph-cli/releases/tag/v0.1.0
