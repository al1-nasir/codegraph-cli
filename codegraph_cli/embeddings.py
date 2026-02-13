"""Neural embedding engine using Sentence Transformers for semantic code understanding.

Local-first architecture:
- Models are downloaded once and cached in ``~/.codegraph/models``.
- All inference runs on-device (CPU or GPU).  No data is ever sent to
  external APIs.

Falls back to a lightweight deterministic hash-embedding when
``sentence-transformers`` is not installed.
"""

from __future__ import annotations

import logging
import math
import os
import re
from hashlib import blake2b
from pathlib import Path
from typing import Iterable, List, Optional, Union

from .config import BASE_DIR

logger = logging.getLogger(__name__)

# Default local model cache directory
MODEL_CACHE_DIR: Path = BASE_DIR / "models"

# Preferred models in priority order
PREFERRED_MODELS: List[str] = [
    "all-MiniLM-L6-v2",
    "nomic-ai/nomic-embed-text-v1.5",
]

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


# ===================================================================
# NeuralEmbedder  (Primary – Sentence Transformers)
# ===================================================================

class NeuralEmbedder:
    """Semantic embedding engine powered by Sentence Transformers.

    The model is downloaded on first use and cached in
    ``~/.codegraph/models`` so that subsequent runs are fully offline.
    All computation is local – **no data leaves the machine**.

    Example::

        embedder = NeuralEmbedder()
        vecs = embedder.embed_documents(["def hello(): ...", "class Foo: ..."])
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[Path] = None,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir or MODEL_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self._model: object = None  # lazy-loaded SentenceTransformer
        self._dim: Optional[int] = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install with:  pip install sentence-transformers"
            )

        # Tell sentence-transformers where to cache
        os.environ.setdefault(
            "SENTENCE_TRANSFORMERS_HOME", str(self.cache_dir),
        )

        try:
            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=str(self.cache_dir),
                device=self.device,
            )
            self._dim = self._model.get_sentence_embedding_dimension()  # type: ignore[union-attr]
            logger.info(
                "Loaded model '%s' (dim=%d) on %s",
                self.model_name, self._dim, self.device,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embedding model '{self.model_name}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        if self._dim is None:
            self._load_model()
        assert self._dim is not None
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string and return a unit-norm vector."""
        self._load_model()
        assert self._model is not None
        embedding = self._model.encode(  # type: ignore[union-attr]
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[List[float]]:
        """Embed multiple documents with batching for efficiency.

        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts per forward pass.

        Returns:
            List of embedding vectors (each normalised to unit length).
        """
        if not texts:
            return []
        self._load_model()
        assert self._model is not None
        embeddings = self._model.encode(  # type: ignore[union-attr]
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings.tolist()

    # Backward-compat alias used by legacy callers
    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        """Alias for :meth:`embed_documents`."""
        return self.embed_documents(list(texts))


# ===================================================================
# HashEmbeddingModel  (Lightweight Fallback)
# ===================================================================

class HashEmbeddingModel:
    """Deterministic token-hashing embedder – no ML dependencies.

    Provides basic keyword-level similarity.  Automatically used as a
    fallback when ``sentence-transformers`` is not available.
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
        """Alias matching the NeuralEmbedder interface."""
        return self.embed_many(texts)


# ===================================================================
# Factory
# ===================================================================

def get_embedder(
    model_name: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    device: str = "cpu",
) -> Union[NeuralEmbedder, HashEmbeddingModel]:
    """Return the best available embedder.

    * If ``sentence-transformers`` is installed → :class:`NeuralEmbedder`.
    * Otherwise → :class:`HashEmbeddingModel` (zero-dependency fallback).
    """
    try:
        import sentence_transformers  # noqa: F401
        return NeuralEmbedder(
            model_name=model_name or "all-MiniLM-L6-v2",
            cache_dir=cache_dir,
            device=device,
        )
    except ImportError:
        logger.warning(
            "sentence-transformers not installed – "
            "using hash-based embeddings (no semantic understanding). "
            "Install with: pip install sentence-transformers"
        )
        return HashEmbeddingModel()


# ===================================================================
# Utility
# ===================================================================

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Cosine similarity between two L2-normalised vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    return sum(a * b for a, b in zip(vec_a, vec_b))


def _l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]
