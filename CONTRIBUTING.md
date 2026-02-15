# Contributing to CodeGraph CLI

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/al1-nasir/codegraph-cli.git
cd codegraph-cli
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,crew,embeddings]"
```

## Running Tests

```bash
pytest                    # run all tests
pytest -x                 # stop on first failure
pytest --cov              # with coverage report
pytest tests/test_cli.py  # single file
```

## Code Style

- **Type hints** on all public functions
- **Docstrings** on all modules, classes, and public methods
- Use **Rich** for terminal output (not raw `print()` with ANSI codes)
- Keep modules focused — one responsibility per file

## Making Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make your changes with tests
4. Run the test suite: `pytest`
5. Commit with a descriptive message: `git commit -m "feat: add X"`
6. Push and open a Pull Request

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation only
- `refactor:` — code change that neither fixes a bug nor adds a feature
- `test:` — adding or updating tests
- `chore:` — maintenance (CI, deps, build)

## Project Structure

See [docs/architecture.md](docs/architecture.md) for a full overview.

Key directories:

| Path | Purpose |
|------|---------|
| `codegraph_cli/` | Main package |
| `codegraph_cli/cli*.py` | CLI commands (Typer) |
| `codegraph_cli/crew_*.py` | CrewAI multi-agent system |
| `tests/` | pytest test suite |
| `docs/` | Documentation |

## Reporting Issues

Open an issue at [github.com/al1-nasir/codegraph-cli/issues](https://github.com/al1-nasir/codegraph-cli/issues) with:

- What you expected vs what happened
- Steps to reproduce
- Python version and OS
- Relevant error output

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
