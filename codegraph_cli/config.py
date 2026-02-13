"""Configuration paths for local CodeGraph memory."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(os.environ.get("CODEGRAPH_HOME", str(Path.home() / ".codegraph"))).expanduser()
MEMORY_DIR = BASE_DIR / "memory"
STATE_FILE = BASE_DIR / "state.json"
DEFAULT_EMBEDDING_DIM = 256
SUPPORTED_EXTENSIONS = {".py"}

# Load configuration from TOML file (if available)
try:
    from .config_manager import load_config
    _toml_config = load_config()
except ImportError:
    _toml_config = {}

# LLM Provider Configuration — loaded from ~/.codegraph/config.toml (set via `cg setup` or `cg set-llm`)
LLM_PROVIDER = _toml_config.get("provider", "ollama")
LLM_API_KEY = _toml_config.get("api_key", "")
LLM_MODEL = _toml_config.get("model", "qwen2.5-coder:7b")
LLM_ENDPOINT = _toml_config.get("endpoint", "http://127.0.0.1:11434/api/generate")


def ensure_base_dirs() -> None:
    """Create base directories for local storage if needed."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    BASE_DIR.mkdir(parents=True, exist_ok=True)
