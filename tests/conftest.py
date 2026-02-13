"""Pytest configuration and fixtures for CodeGraph CLI tests."""

import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from codegraph_cli.storage import GraphStore, ProjectManager


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    tmp = Path(tempfile.mkdtemp())
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_project_path() -> Path:
    """Get path to sample test project."""
    return Path(__file__).parent / "fixtures" / "sample_project"


@pytest.fixture
def temp_project_manager(temp_dir: Path, monkeypatch) -> ProjectManager:
    """Create a ProjectManager with temporary storage."""
    memory_dir = temp_dir / "memory"
    state_file = temp_dir / "state.json"
    
    # Patch the config to use temp directories
    monkeypatch.setattr("codegraph_cli.config.MEMORY_DIR", memory_dir)
    monkeypatch.setattr("codegraph_cli.config.STATE_FILE", state_file)
    
    return ProjectManager()


@pytest.fixture
def temp_graph_store(temp_dir: Path) -> GraphStore:
    """Create a GraphStore with temporary storage."""
    project_dir = temp_dir / "test_project"
    project_dir.mkdir(parents=True, exist_ok=True)
    return GraphStore(project_dir)


@pytest.fixture
def indexed_sample_store(temp_dir: Path, sample_project_path: Path) -> GraphStore:
    """Create a GraphStore with indexed sample project."""
    from codegraph_cli.agents import GraphAgent
    from codegraph_cli.embeddings import HashEmbeddingModel
    
    project_dir = temp_dir / "sample_indexed"
    project_dir.mkdir(parents=True, exist_ok=True)
    
    store = GraphStore(project_dir)
    agent = GraphAgent(store, HashEmbeddingModel())
    agent.index_project(sample_project_path)
    
    return store


@pytest.fixture
def sample_python_code() -> str:
    """Sample Python code for testing parser."""
    return '''"""Sample module for testing."""

def hello(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"

class Calculator:
    """Simple calculator."""
    
    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    
    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        result = self.add(a, 0)  # Call to add
        for _ in range(b - 1):
            result = self.add(result, a)
        return result
'''


@pytest.fixture
def mock_llm_response() -> str:
    """Mock LLM response for testing."""
    return """Main risks:
- Changing this function may break downstream callers
- Data validation logic could affect database integrity

Most likely breakpoints:
- Function signature changes will require caller updates
- Return type changes will break type checking

Test recommendations:
- Add unit tests for edge cases
- Run integration tests with dependent modules
- Verify error handling paths
"""
