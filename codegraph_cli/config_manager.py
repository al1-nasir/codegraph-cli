"""Configuration manager for CodeGraph CLI using TOML files."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import toml
except ImportError:
    toml = None  # type: ignore

from .config import BASE_DIR


CONFIG_FILE = BASE_DIR / "config.toml"


# Default configurations for each provider
DEFAULT_CONFIGS = {
    "ollama": {
        "provider": "ollama",
        "model": "qwen2.5-coder:7b",
        "endpoint": "http://127.0.0.1:11434/api/generate",
    },
    "groq": {
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
        "api_key": "",
    },
    "openai": {
        "provider": "openai",
        "model": "gpt-4",
        "api_key": "",
    },
    "anthropic": {
        "provider": "anthropic",
        "model": "claude-3-5-sonnet-20241022",
        "api_key": "",
    },
    "gemini": {
        "provider": "gemini",
        "model": "gemini-2.0-flash",
        "api_key": "",
    },
    "openrouter": {
        "provider": "openrouter",
        "model": "google/gemini-2.0-flash-exp:free",
        "api_key": "",
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
    },
}


def load_config() -> Dict[str, Any]:
    """Load configuration from TOML file.
    
    Returns:
        Configuration dictionary with provider settings.
        Falls back to Ollama defaults if file doesn't exist.
    """
    if not CONFIG_FILE.exists():
        return DEFAULT_CONFIGS["ollama"].copy()
    
    if toml is None:
        # Fallback if toml not installed
        return DEFAULT_CONFIGS["ollama"].copy()
    
    try:
        with open(CONFIG_FILE, "r") as f:
            config = toml.load(f)
        return config.get("llm", DEFAULT_CONFIGS["ollama"].copy())
    except Exception:
        return DEFAULT_CONFIGS["ollama"].copy()


def load_full_config() -> Dict[str, Any]:
    """Load the entire TOML config (all sections)."""
    if not CONFIG_FILE.exists() or toml is None:
        return {}
    try:
        with open(CONFIG_FILE, "r") as f:
            return toml.load(f)
    except Exception:
        return {}


def _save_full_config(config: Dict[str, Any]) -> bool:
    """Write entire config dict to TOML file, preserving all sections."""
    if toml is None:
        return False
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(CONFIG_FILE, "w") as f:
            toml.dump(config, f)
        return True
    except Exception:
        return False


def save_config(provider: str, model: str, api_key: str = "", endpoint: str = "") -> bool:
    """Save LLM configuration to TOML file.
    
    Preserves other sections (e.g. ``[embeddings]``) in the file.
    
    Args:
        provider: Provider name (ollama, groq, openai, anthropic, gemini, openrouter)
        model: Model name
        api_key: API key for cloud providers
        endpoint: Custom endpoint (for Ollama)
    
    Returns:
        True if saved successfully, False otherwise
    """
    config = load_full_config()
    
    config["llm"] = {
        "provider": provider,
        "model": model,
    }
    if api_key:
        config["llm"]["api_key"] = api_key
    if endpoint:
        config["llm"]["endpoint"] = endpoint
    
    return _save_full_config(config)


# ------------------------------------------------------------------
# Embedding configuration
# ------------------------------------------------------------------

def load_embedding_config() -> Dict[str, Any]:
    """Load embedding configuration from ``[embeddings]`` section.
    
    Returns:
        Dict with at least ``model`` key, or empty dict.
    """
    full = load_full_config()
    return full.get("embeddings", {})


def save_embedding_config(model_key: str) -> bool:
    """Save embedding model choice to config TOML.
    
    Preserves ``[llm]`` and other sections.
    
    Args:
        model_key: One of the keys from ``EMBEDDING_MODELS``
                   (e.g. ``"minilm"``, ``"jina-code"``, ``"hash"``).
    
    Returns:
        True if saved successfully.
    """
    config = load_full_config()
    config["embeddings"] = {"model": model_key}
    return _save_full_config(config)


def clear_embedding_config() -> bool:
    """Remove ``[embeddings]`` section from config, resetting to default."""
    config = load_full_config()
    config.pop("embeddings", None)
    return _save_full_config(config)


def get_provider_config(provider: str) -> Dict[str, Any]:
    """Get default configuration for a specific provider.
    
    Args:
        provider: Provider name
    
    Returns:
        Default configuration dictionary
    """
    return DEFAULT_CONFIGS.get(provider, DEFAULT_CONFIGS["ollama"]).copy()


def validate_ollama_connection(endpoint: str = "http://127.0.0.1:11434") -> bool:
    """Check if Ollama is running and accessible.
    
    Args:
        endpoint: Ollama endpoint URL
    
    Returns:
        True if Ollama is accessible, False otherwise
    """
    try:
        req = urllib.request.Request(f"{endpoint}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def get_ollama_models(endpoint: str = "http://127.0.0.1:11434") -> list[str]:
    """Fetch available models from Ollama.
    
    Args:
        endpoint: Ollama endpoint URL
    
    Returns:
        List of available model names
    """
    try:
        req = urllib.request.Request(f"{endpoint}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("models", [])
            return [model["name"] for model in models]
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
        return []


def validate_api_key(provider: str, api_key: str, model: str) -> tuple[bool, str]:
    """Validate API key by making a test request.
    
    Args:
        provider: Provider name (groq, openai, anthropic)
        api_key: API key to validate
        model: Model name to test
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not api_key:
        return False, "API key is required"
    
    try:
        if provider == "groq":
            return _validate_groq(api_key, model)
        elif provider == "openai":
            return _validate_openai(api_key, model)
        elif provider == "anthropic":
            return _validate_anthropic(api_key, model)
        elif provider == "gemini":
            return _validate_gemini(api_key, model)
        elif provider == "openrouter":
            return _validate_openrouter(api_key, model)
        else:
            return False, f"Unknown provider: {provider}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def _validate_groq(api_key: str, model: str) -> tuple[bool, str]:
    """Validate Groq API key."""
    import subprocess
    
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 5,
    })
    
    try:
        result = subprocess.run(
            [
                "curl", "-s", "-X", "POST",
                "https://api.groq.com/openai/v1/chat/completions",
                "-H", "Content-Type: application/json",
                "-H", f"Authorization: Bearer {api_key}",
                "-d", payload,
                "--max-time", "10"
            ],
            capture_output=True,
            text=True,
            timeout=15
        )
        
        if result.returncode == 0 and result.stdout:
            response = json.loads(result.stdout)
            if "error" in response:
                return False, response["error"].get("message", "Invalid API key")
            return True, "Valid"
        return False, "Connection failed"
    except Exception as e:
        return False, str(e)


def _validate_openai(api_key: str, model: str) -> tuple[bool, str]:
    """Validate OpenAI API key."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 5,
    }).encode("utf-8")
    
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return True, "Valid"
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return False, "Invalid API key"
        return False, f"HTTP error: {e.code}"
    except Exception as e:
        return False, str(e)


def _validate_anthropic(api_key: str, model: str) -> tuple[bool, str]:
    """Validate Anthropic API key."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 5,
    }).encode("utf-8")
    
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return True, "Valid"
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return False, "Invalid API key"
        return False, f"HTTP error: {e.code}"
    except Exception as e:
        return False, str(e)


def _validate_gemini(api_key: str, model: str) -> tuple[bool, str]:
    """Validate Gemini API key."""
    # Use the list models endpoint for validation
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    
    req = urllib.request.Request(url, method="GET")
    
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return True, "Valid"
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return False, "Invalid API key"
        return False, f"HTTP error: {e.code}"
    except Exception as e:
        return False, str(e)


def _validate_openrouter(api_key: str, model: str) -> tuple[bool, str]:
    """Validate OpenRouter API key."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 5,
    }).encode("utf-8")
    
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return True, "Valid"
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return False, "Invalid API key"
        # 402 means valid key but no credits - still a valid key
        if e.code == 402:
            return True, "Valid (no credits)"
        return False, f"HTTP error: {e.code}"
    except Exception as e:
        return False, str(e)
