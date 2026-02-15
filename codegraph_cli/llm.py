"""Multi-provider LLM adapter supporting Ollama, Groq, OpenAI, and Anthropic."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from .config import LLM_API_KEY, LLM_ENDPOINT, LLM_MODEL, LLM_PROVIDER


class LLMProvider:
    """Base class for LLM providers."""
    
    def generate(self, prompt: str) -> Optional[str]:
        """Generate a response from the LLM."""
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""
    
    def __init__(self, model: str, endpoint: str):
        self.model = model
        self.endpoint = endpoint
    
    def generate(self, prompt: str) -> Optional[str]:
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }).encode("utf-8")
        
        req = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8")
                parsed = json.loads(body)
                return parsed.get("response")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return None


class GroqProvider(LLMProvider):
    """Groq cloud API provider."""
    
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
    
    def generate(self, prompt: str) -> Optional[str]:
        if not self.api_key:
            return None
        
        try:
            import requests
            
            response = requests.post(
                self.endpoint,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 1024,
                },
                timeout=20,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception:
            return None


class OpenAIProvider(LLMProvider):
    """OpenAI API provider (also works with OpenRouter and other OpenAI-compatible APIs)."""
    
    def __init__(self, model: str, api_key: str, endpoint: str = "https://api.openai.com/v1/chat/completions"):
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint
    
    def generate(self, prompt: str) -> Optional[str]:
        if not self.api_key:
            return None
        
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1024,
        }).encode("utf-8")
        
        req = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8")
                parsed = json.loads(body)
                return parsed["choices"][0]["message"]["content"]
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
            return None


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""
    
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self.endpoint = "https://api.anthropic.com/v1/messages"
    
    def generate(self, prompt: str) -> Optional[str]:
        if not self.api_key:
            return None
        
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.1,
        }).encode("utf-8")
        
        req = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8")
                parsed = json.loads(body)
                return parsed["content"][0]["text"]
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
            return None


class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""
    
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    
    def generate(self, prompt: str) -> Optional[str]:
        if not self.api_key:
            return None
        
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 1024,
            },
        }).encode("utf-8")
        
        url = f"{self.endpoint}?key={self.api_key}"
        
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8")
                parsed = json.loads(body)
                return parsed["candidates"][0]["content"]["parts"][0]["text"]
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError):
            return None


class OpenRouterProvider(LLMProvider):
    """OpenRouter API provider (OpenAI-compatible, multi-model gateway)."""
    
    def __init__(self, model: str, api_key: str, endpoint: str = "https://openrouter.ai/api/v1/chat/completions"):
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint
    
    def generate(self, prompt: str) -> Optional[str]:
        if not self.api_key:
            return None
        
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 4096,
        }).encode("utf-8")
        
        req = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode("utf-8")
                parsed = json.loads(body)
                return self._extract_response(parsed)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
            return None

    @staticmethod
    def _extract_response(parsed: dict) -> Optional[str]:
        """Extract response text, handling reasoning models that return empty content."""
        msg = parsed["choices"][0]["message"]
        content = msg.get("content") or ""
        if content.strip():
            return content
        # Reasoning models put output in 'reasoning' field
        reasoning = msg.get("reasoning") or ""
        if reasoning.strip():
            return reasoning
        # Check reasoning_details array
        for detail in msg.get("reasoning_details") or []:
            if isinstance(detail, dict) and detail.get("text", "").strip():
                return detail["text"]
        return content or None


class LocalLLM:
    """Multi-provider LLM manager with automatic fallback."""
    
    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
    ):
        """Initialize LLM with provider selection.
        
        Args:
            model: Model name (defaults to config or "qwen2.5-coder:7b")
            provider: Provider name: "ollama", "groq", "openai", "anthropic" (defaults to config)
            api_key: API key for cloud providers (defaults to config)
            endpoint: Custom endpoint for Ollama (defaults to config)
        """
        self.provider_name = provider or LLM_PROVIDER
        self.model = model or LLM_MODEL
        self.api_key = api_key or LLM_API_KEY
        self.endpoint = endpoint or LLM_ENDPOINT
        
        self.provider = self._create_provider()
    
    def _create_provider(self) -> LLMProvider:
        """Create the appropriate provider based on configuration."""
        provider_name = self.provider_name.lower()
        
        if provider_name == "groq":
            # Default Groq models: llama-3.3-70b-versatile, mixtral-8x7b-32768
            model = self.model if self.model != "qwen2.5-coder:7b" else "llama-3.3-70b-versatile"
            return GroqProvider(model, self.api_key)
        
        elif provider_name == "openai":
            # Default OpenAI models: gpt-4, gpt-3.5-turbo
            # Also supports OpenRouter and other OpenAI-compatible APIs via custom endpoint
            model = self.model if self.model != "qwen2.5-coder:7b" else "gpt-4"
            endpoint = self.endpoint if self.endpoint else "https://api.openai.com/v1/chat/completions"
            return OpenAIProvider(model, self.api_key, endpoint)
        
        elif provider_name == "anthropic":
            # Default Anthropic models: claude-3-5-sonnet-20241022, claude-3-opus-20240229
            model = self.model if self.model != "qwen2.5-coder:7b" else "claude-3-5-sonnet-20241022"
            return AnthropicProvider(model, self.api_key)
        
        elif provider_name == "gemini":
            # Google Gemini models: gemini-2.0-flash, gemini-1.5-pro, etc.
            model = self.model if self.model != "qwen2.5-coder:7b" else "gemini-2.0-flash"
            return GeminiProvider(model, self.api_key)
        
        elif provider_name == "openrouter":
            # OpenRouter: multi-model gateway with OpenAI-compatible API
            model = self.model if self.model != "qwen2.5-coder:7b" else "google/gemini-2.0-flash-exp:free"
            endpoint = self.endpoint if self.endpoint else "https://openrouter.ai/api/v1/chat/completions"
            return OpenRouterProvider(model, self.api_key, endpoint)
        
        else:  # Default to Ollama
            return OllamaProvider(self.model, self.endpoint)
    
    def explain(self, prompt: str) -> str:
        """Generate explanation using configured provider with fallback."""
        response = self.provider.generate(prompt)
        if response:
            return response
        return self._fallback(prompt)
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> Optional[str]:
        """Generate response for multi-turn chat conversation.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            Assistant response or None if failed
        """
        provider_name = self.provider_name.lower()
        
        try:
            if provider_name == "groq":
                return self._chat_groq(messages, max_tokens, temperature)
            elif provider_name == "openai":
                return self._chat_openai(messages, max_tokens, temperature)
            elif provider_name == "anthropic":
                return self._chat_anthropic(messages, max_tokens, temperature)
            elif provider_name == "gemini":
                return self._chat_gemini(messages, max_tokens, temperature)
            elif provider_name == "openrouter":
                return self._chat_openrouter(messages, max_tokens, temperature)
            else:  # Ollama
                # Ollama doesn't support chat format, convert to single prompt
                prompt = self._messages_to_prompt(messages)
                return self.provider.generate(prompt)
        except Exception:
            return None
    
    def _chat_groq(self, messages: List[Dict], max_tokens: int, temperature: float) -> Optional[str]:
        """Chat completion for Groq."""
        import requests
        
        response = requests.post(
            self.provider.endpoint,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=35,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def _chat_openai(self, messages: List[Dict], max_tokens: int, temperature: float) -> Optional[str]:
        """Chat completion for OpenAI."""
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")
        
        req = urllib.request.Request(
            self.provider.endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            parsed = json.loads(body)
            return parsed["choices"][0]["message"]["content"]
    
    def _chat_anthropic(self, messages: List[Dict], max_tokens: int, temperature: float) -> Optional[str]:
        """Chat completion for Anthropic."""
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode("utf-8")
        
        req = urllib.request.Request(
            self.provider.endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            parsed = json.loads(body)
            return parsed["content"][0]["text"]
    
    def _chat_gemini(self, messages: List[Dict], max_tokens: int, temperature: float) -> Optional[str]:
        """Chat completion for Gemini."""
        # Convert messages to Gemini format
        contents = []
        system_instruction = None
        for msg in messages:
            role = msg["role"]
            if role == "system":
                system_instruction = msg["content"]
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": msg["content"]}]})
        
        body: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        
        payload = json.dumps(body).encode("utf-8")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        
        with urllib.request.urlopen(req, timeout=30) as resp:
            body_resp = resp.read().decode("utf-8")
            parsed = json.loads(body_resp)
            return parsed["candidates"][0]["content"]["parts"][0]["text"]
    
    def _chat_openrouter(self, messages: List[Dict], max_tokens: int, temperature: float) -> Optional[str]:
        """Chat completion for OpenRouter (OpenAI-compatible)."""
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")
        
        req = urllib.request.Request(
            self.provider.endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        
        with urllib.request.urlopen(req, timeout=60) as resp:
            body_resp = resp.read().decode("utf-8")
            parsed = json.loads(body_resp)
            return OpenRouterProvider._extract_response(parsed)
    
    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert chat messages to single prompt for Ollama."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        return "\n\n".join(parts)
    
    def _fallback(self, prompt: str) -> str:
        """Deterministic fallback when LLM is unavailable."""
        head = prompt[:600].strip().replace("\n", " ")
        provider_msg = f"LLM provider '{self.provider_name}' was unavailable"
        
        return (
            f"{provider_msg}; returning a deterministic fallback summary.\n"
            "Context excerpt:\n"
            f"{head}\n\n"
            "Recommendation:\n"
            "- Inspect the listed call/dependency chain\n"
            "- Run unit tests around impacted functions\n"
            "- Validate side effects at integration boundaries"
        )


def create_crewai_llm(local_llm: LocalLLM):
    """Create CrewAI-compatible LLM from LocalLLM configuration.
    
    This factory function creates native CrewAI LLM instances based on the
    configured provider. CrewAI has its own LLM handling and doesn't work
    with custom adapter objects.
    
    Args:
        local_llm: LocalLLM instance with provider configuration
        
    Returns:
        CrewAI-compatible LLM instance
        
    Raises:
        ImportError: If crewai package is not installed
    """
    try:
        from crewai import LLM
    except ImportError:
        raise ImportError(
            "CrewAI is required for multi-agent chat. "
            "Install it with: pip install crewai"
        )
    
    provider = local_llm.provider_name.lower()
    
    if provider == "ollama":
        # CrewAI supports Ollama with custom base_url
        # Remove /api/generate suffix if present
        base_url = local_llm.endpoint
        if base_url.endswith("/api/generate"):
            base_url = base_url.replace("/api/generate", "")
        
        return LLM(
            model=f"ollama/{local_llm.model}",
            base_url=base_url,
            max_tokens=4096,
        )
    
    elif provider == "groq":
        # Groq uses OpenAI-compatible API
        return LLM(
            model=f"groq/{local_llm.model}",
            api_key=local_llm.api_key,
            max_tokens=4096,
        )
    
    elif provider == "openai":
        # OpenAI or OpenAI-compatible APIs (OpenRouter, etc.)
        # Check if using OpenRouter
        if local_llm.endpoint and "openrouter.ai" in local_llm.endpoint:
            # OpenRouter - use special prefix for LiteLLM routing
            return LLM(
                model=f"openrouter/{local_llm.model}",
                api_key=local_llm.api_key,
                max_tokens=4096,
            )
        elif local_llm.endpoint and local_llm.endpoint != "https://api.openai.com/v1/chat/completions":
            # Other custom endpoint - use base_url
            return LLM(
                model=local_llm.model,
                api_key=local_llm.api_key,
                base_url=local_llm.endpoint.replace("/chat/completions", ""),
                max_tokens=4096,
            )
        else:
            # Standard OpenAI
            return LLM(
                model=f"openai/{local_llm.model}",
                api_key=local_llm.api_key,
                max_tokens=4096,
            )
    
    elif provider == "anthropic":
        return LLM(
            model=f"anthropic/{local_llm.model}",
            api_key=local_llm.api_key,
            max_tokens=4096,
        )
    
    elif provider == "gemini":
        # CrewAI supports Gemini via LiteLLM
        return LLM(
            model=f"gemini/{local_llm.model}",
            api_key=local_llm.api_key,
            max_tokens=4096,
        )
    
    elif provider == "openrouter":
        # OpenRouter uses OpenAI-compatible API
        return LLM(
            model=f"openrouter/{local_llm.model}",
            api_key=local_llm.api_key,
            max_tokens=4096,
        )
    
    else:
        # Fallback to Ollama for unknown providers
        base_url = local_llm.endpoint
        if base_url.endswith("/api/generate"):
            base_url = base_url.replace("/api/generate", "")
        
        return LLM(
            model=f"ollama/{local_llm.model}",
            base_url=base_url,
        )
