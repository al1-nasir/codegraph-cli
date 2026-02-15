"""Configurable code embedding engine with multiple model support.

Supported models (configure via ``cg config set-embedding``):

========== ====================================== ========= ====== ======================
Key        HuggingFace Model                      Download  Dim    Notes
========== ====================================== ========= ====== ======================
qodo-1.5b  Qodo/Qodo-Embed-1-1.5B                ~6.2 GB   1536   Best quality, code-optimized
jina-code  jinaai/jina-embeddings-v2-base-code    ~550 MB    768   Good quality, code-aware
bge-base   BAAI/bge-base-en-v1.5                  ~440 MB    768   Solid general-purpose
minilm     sentence-transformers/all-MiniLM-L6-v2  ~80 MB    384   Tiny and fast
hash       (none)                                     0 B    256   No ML, keyword-level only
========== ====================================== ========= ====== ======================

Architecture:
- Models downloaded once from HuggingFace and cached in ``~/.codegraph/models``.
- All inference runs on-device (CPU or GPU). No data leaves the machine.
- Uses raw ``transformers`` library only — no sentence-transformers, no flash_attn.
- Falls back to hash embeddings when ``torch``/``transformers`` are not installed.
"""

from __future__ import annotations

import logging
import math
import re
from hashlib import blake2b
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from .config import BASE_DIR

logger = logging.getLogger(__name__)

# Default local model cache directory
MODEL_CACHE_DIR: Path = BASE_DIR / "models"

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


# ===================================================================
# Model Registry
# ===================================================================

EMBEDDING_MODELS: Dict[str, Dict[str, Any]] = {
    "qodo-1.5b": {
        "name": "Qodo Embed 1.5B",
        "hf_id": "Qodo/Qodo-Embed-1-1.5B",
        "dim": 1536,
        "max_tokens": 8192,
        "size": "~6.2 GB",
        "description": "Best quality, code-optimized (needs 8GB+ RAM)",
        "pooling": "last_token",
        "trust_remote_code": True,
    },
    "jina-code": {
        "name": "Jina Embeddings v2 Code",
        "hf_id": "jinaai/jina-embeddings-v2-base-code",
        "dim": 768,
        "max_tokens": 8192,
        "size": "~550 MB",
        "description": "Good quality, code-aware, lightweight",
        "pooling": "mean",
        "trust_remote_code": True,
    },
    "bge-base": {
        "name": "BGE Base EN v1.5",
        "hf_id": "BAAI/bge-base-en-v1.5",
        "dim": 768,
        "max_tokens": 512,
        "size": "~440 MB",
        "description": "Solid general-purpose, fast",
        "pooling": "cls",
        "trust_remote_code": False,
    },
    "minilm": {
        "name": "MiniLM L6 v2",
        "hf_id": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "max_tokens": 256,
        "size": "~80 MB",
        "description": "Tiny and fast, decent quality",
        "pooling": "mean",
        "trust_remote_code": False,
    },
    "hash": {
        "name": "Hash Embedding",
        "hf_id": None,
        "dim": 256,
        "max_tokens": None,
        "size": "0 bytes",
        "description": "Zero-dependency fallback, no semantics",
        "pooling": None,
        "trust_remote_code": False,
    },
}

DEFAULT_MODEL = "hash"


# ===================================================================
# TransformerEmbedder  (handles all HuggingFace models)
# ===================================================================

class TransformerEmbedder:
    """Generic HuggingFace embedding engine with configurable pooling.

    Supports multiple pooling strategies:

    - **last_token** — last non-padding token (Qodo models).
    - **mean** — mean over non-padding tokens (Jina, MiniLM).
    - **cls** — ``[CLS]`` first token (BGE models).

    Model weights are downloaded on first use and cached in
    ``~/.codegraph/models/`` for offline subsequent runs.
    """

    def __init__(
        self,
        model_key: str,
        cache_dir: Optional[Path] = None,
        device: str = "cpu",
    ) -> None:
        if model_key not in EMBEDDING_MODELS:
            raise ValueError(
                f"Unknown model: '{model_key}'. "
                f"Available: {', '.join(EMBEDDING_MODELS.keys())}"
            )

        spec = EMBEDDING_MODELS[model_key]
        if spec["hf_id"] is None:
            raise ValueError(
                f"'{model_key}' has no transformer backend. Use HashEmbeddingModel."
            )

        self.model_key = model_key
        self.hf_id: str = spec["hf_id"]
        self.dim: int = spec["dim"]
        self.max_length: int = spec["max_tokens"]
        self.pooling: str = spec["pooling"]
        self.trust_remote_code: bool = spec["trust_remote_code"]
        self.cache_dir = cache_dir or MODEL_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self._model: Any = None
        self._tokenizer: Any = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None:
            return

        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "torch and transformers are required for neural embeddings.\n"
                "Install with:  pip install codegraph-cli[embeddings]\n"
                "For CPU-only (skip NVIDIA packages):\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                "  pip install transformers"
            )

        logger.info(
            "Loading embedding model '%s' (%s) — first run downloads %s...",
            self.model_key, self.hf_id, EMBEDDING_MODELS[self.model_key]["size"],
        )

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.hf_id,
                cache_dir=str(self.cache_dir),
                trust_remote_code=self.trust_remote_code,
            )
            self._model = AutoModel.from_pretrained(
                self.hf_id,
                cache_dir=str(self.cache_dir),
                trust_remote_code=self.trust_remote_code,
            )
            self._model.eval()
            self._model.to(self.device)
            logger.info(
                "Loaded '%s' (dim=%d, pooling=%s) on %s",
                self.model_key, self.dim, self.pooling, self.device,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embedding model '{self.model_key}' "
                f"({self.hf_id}): {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Pooling strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _pool_last_token(last_hidden_states: Any, attention_mask: Any) -> Any:
        """Last non-padding token (Qodo style)."""
        import torch

        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]

    @staticmethod
    def _pool_mean(last_hidden_states: Any, attention_mask: Any) -> Any:
        """Mean over non-padding tokens (Jina, MiniLM)."""
        mask_expanded = attention_mask.unsqueeze(-1).expand(
            last_hidden_states.size()
        ).float()
        sum_embeddings = (last_hidden_states * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

    @staticmethod
    def _pool_cls(last_hidden_states: Any, attention_mask: Any) -> Any:
        """[CLS] first token (BGE)."""
        return last_hidden_states[:, 0]

    def _pool(self, last_hidden_states: Any, attention_mask: Any) -> Any:
        """Dispatch to the pooling strategy for this model."""
        if self.pooling == "last_token":
            return self._pool_last_token(last_hidden_states, attention_mask)
        if self.pooling == "mean":
            return self._pool_mean(last_hidden_states, attention_mask)
        if self.pooling == "cls":
            return self._pool_cls(last_hidden_states, attention_mask)
        raise ValueError(f"Unknown pooling strategy: {self.pooling}")

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def _encode(self, texts: List[str]) -> List[List[float]]:
        """Encode a batch of texts into L2-normalised embedding vectors."""
        import torch
        import torch.nn.functional as F

        self._load_model()

        batch_dict = self._tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

        with torch.no_grad():
            outputs = self._model(**batch_dict)

        embeddings = self._pool(
            outputs.last_hidden_state, batch_dict["attention_mask"],
        )
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().tolist()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string and return a unit-norm vector."""
        return self._encode([text])[0]

    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = 16,
    ) -> List[List[float]]:
        """Embed multiple documents with batching."""
        if not texts:
            return []
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            all_embeddings.extend(self._encode(texts[i : i + batch_size]))
        return all_embeddings

    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        """Alias for :meth:`embed_documents`."""
        return self.embed_documents(list(texts))


# ===================================================================
# HashEmbeddingModel  (Zero-dependency fallback)
# ===================================================================

class HashEmbeddingModel:
    """Deterministic token-hashing embedder — no ML dependencies.

    Provides basic keyword-level similarity. Used as the default when
    ``torch``/``transformers`` are not installed or when ``hash`` is
    selected via ``cg config set-embedding hash``.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed_text(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            return vec
        for token in tokens:
            digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vec[idx] += sign
        return _l2_normalize(vec)

    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        return [self.embed_text(text) for text in texts]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Alias matching the TransformerEmbedder interface."""
        return self.embed_many(texts)


# ===================================================================
# Factory
# ===================================================================

def get_embedder(
    model_key: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    device: str = "cpu",
) -> Union[TransformerEmbedder, HashEmbeddingModel]:
    """Return the configured embedder.

    Resolution order:

    1. Explicit ``model_key`` argument.
    2. ``[embeddings].model`` from ``~/.codegraph/config.toml``.
    3. ``"hash"`` (zero-dependency fallback).

    If a transformer model is configured but ``torch``/``transformers``
    are missing, falls back to hash with a warning.
    """
    if model_key is None:
        try:
            from .config_manager import load_embedding_config
            emb_cfg = load_embedding_config()
            model_key = emb_cfg.get("model", None)
        except Exception:
            model_key = None

    # Default to hash if nothing configured
    if model_key is None:
        model_key = DEFAULT_MODEL

    # Hash path — no ML needed
    if model_key == "hash":
        return HashEmbeddingModel()

    # Unknown model guard
    if model_key not in EMBEDDING_MODELS:
        logger.warning(
            "Unknown embedding model '%s' — falling back to hash.", model_key,
        )
        return HashEmbeddingModel()

    spec = EMBEDDING_MODELS[model_key]
    if spec["hf_id"] is None:
        return HashEmbeddingModel()

    # Transformer path — check dependencies
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return TransformerEmbedder(
            model_key=model_key, cache_dir=cache_dir, device=device,
        )
    except ImportError:
        logger.warning(
            "Embedding model '%s' requires torch + transformers. "
            "Falling back to hash embeddings.  Install with: "
            "pip install codegraph-cli[embeddings]",
            model_key,
        )
        return HashEmbeddingModel()


# ===================================================================
# Utility
# ===================================================================

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Cosine similarity between two vectors.

    If vectors are already L2-normalised the dot product *is* the
    cosine similarity.  For safety this function still divides by
    the product of norms so it works correctly even when vectors
    are *not* pre-normalised.

    Returns a value in ``[-1, 1]``.  Zero-length or mismatched
    vectors return ``0.0``.
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def _l2_normalize(vec: List[float]) -> List[float]:
    """L2-normalise *vec* in-place.  Returns zero vector unchanged."""
    norm = math.sqrt(sum(v * v for v in vec))
    if norm < 1e-12:
        return vec
    return [v / norm for v in vec]


def embedding_norm(vec: List[float]) -> float:
    """Return the L2 norm of a vector."""
    return math.sqrt(sum(v * v for v in vec))


# ===================================================================
# Debug / Validation Utilities
# ===================================================================

def validate_embedding(vec: List[float], label: str = "embedding") -> Dict[str, Any]:
    """Validate an embedding vector and return diagnostic info.

    Checks:
    - Vector is non-empty
    - Vector contains no NaN / Inf values
    - L2 norm is approximately 1.0 (unit-normalised)
    - Vector is not all-zeros

    Returns a dict with ``ok``, ``norm``, ``dim``, and any ``warnings``.
    """
    info: Dict[str, Any] = {
        "label": label,
        "dim": len(vec),
        "ok": True,
        "warnings": [],
    }

    if not vec:
        info["ok"] = False
        info["warnings"].append("empty vector")
        info["norm"] = 0.0
        return info

    norm = embedding_norm(vec)
    info["norm"] = norm

    if any(math.isnan(v) or math.isinf(v) for v in vec):
        info["ok"] = False
        info["warnings"].append("contains NaN or Inf")

    if norm < 1e-9:
        info["ok"] = False
        info["warnings"].append("zero vector")
    elif abs(norm - 1.0) > 0.01:
        info["warnings"].append(f"not unit-normalised (norm={norm:.4f})")

    return info


def debug_embed(text: str, embedder: Optional[Union["TransformerEmbedder", "HashEmbeddingModel"]] = None) -> Dict[str, Any]:
    """Embed *text* and return detailed diagnostics.

    If *embedder* is ``None`` the default embedder from config is used.

    Returns dict with ``text``, ``dim``, ``norm``, ``first_5``,
    ``self_similarity``, and ``validation``.
    """
    if embedder is None:
        embedder = get_embedder()

    vec = embedder.embed_text(text)
    val = validate_embedding(vec, label=text[:60])
    self_sim = cosine_similarity(vec, vec)

    return {
        "text": text[:120],
        "model": getattr(embedder, "model_key", "hash"),
        "dim": len(vec),
        "norm": val["norm"],
        "first_5": vec[:5],
        "self_similarity": self_sim,
        "validation": val,
    }
