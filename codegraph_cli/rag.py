"""Retrieval-Augmented components for semantic code search.

Uses LanceDB hybrid search (vector + metadata filters) for fast,
accurate code retrieval.  Falls back to brute-force cosine similarity
when the vector store is unavailable.

Improvements over the original implementation:

- **Cosine metric** — LanceDB searches use ``metric="cosine"`` so
  ``_distance`` values are true cosine distances (``1 − cos_sim``).
- **Minimum score threshold** — results below a configurable quality
  floor are discarded before returning to callers.
- **Graph-neighbour augmentation** — after the initial semantic top-k,
  direct dependency neighbours of the best hits are fetched from the
  graph store and merged (de-duplicated) into the result set.
- **Result caching** — a small LRU dict avoids re-computing identical
  queries within the same session.
- **Context compression** — :meth:`retrieve_context` strips import
  lines, trims excessively long snippets, and formats structured
  metadata so the LLM receives clean, information-dense context.
"""

from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, Union

from .embeddings import HashEmbeddingModel, TransformerEmbedder, cosine_similarity
from .models import SearchResult
from .storage import GraphStore

logger = logging.getLogger(__name__)

# Minimum similarity score (0..1) below which results are dropped.
MIN_SCORE_THRESHOLD: float = 0.05

# Import line regex — used to strip bare imports from context snippets.
_IMPORT_RE = re.compile(r"^(?:from\s+\S+\s+)?import\s+.+\n?", re.MULTILINE)

# Max characters per snippet in formatted context output.
_MAX_SNIPPET_CHARS = 1000

# Max entries in the per-session query cache.
_CACHE_SIZE = 64


class RAGRetriever:
    """Retrieve relevant code nodes from graph memory via semantic similarity.

    Supports two modes:

    1. **Vector store mode** (fast, preferred) — delegates to a
       **model-specific** LanceDB table via ``GraphStore``.  Each
       embedding model gets its own table so dimension mismatches
       cannot occur.  If no table exists for the current model, the
       retriever automatically re-embeds all nodes from SQLite
       (one-time, transparent to the caller).
    2. **Brute-force mode** (fallback) — scans all SQLite rows and
       computes cosine similarity in Python.  Used only when LanceDB
       is not installed at all.

    The ``embedding_model`` argument accepts either a
    :class:`~codegraph_cli.embeddings.TransformerEmbedder` or the lightweight
    :class:`~codegraph_cli.embeddings.HashEmbeddingModel`.
    """

    def __init__(
        self,
        store: GraphStore,
        embedding_model: Union[TransformerEmbedder, HashEmbeddingModel, Any],
        min_score: float = MIN_SCORE_THRESHOLD,
        enable_graph_augment: bool = True,
    ) -> None:
        self.store = store
        self.embedding_model = embedding_model
        self.min_score = min_score
        self.enable_graph_augment = enable_graph_augment

        # Resolve the model-specific vector store
        self._model_key: str = getattr(embedding_model, "model_key", "hash")
        self._model_vs: Optional[Any] = None
        self.use_vector_store: bool = False
        self._init_model_vector_store()

        # Simple LRU cache: query_text → List[SearchResult]
        self._cache: OrderedDict[str, List[SearchResult]] = OrderedDict()

    # ------------------------------------------------------------------
    # Model-specific vector store initialisation
    # ------------------------------------------------------------------

    def _init_model_vector_store(self) -> None:
        """Obtain the LanceDB table for the current embedding model.

        If the table doesn't exist or is empty, trigger a one-time
        re-ingestion from SQLite so every model always has its own
        properly-dimensioned vector store.
        """
        self._model_vs = self.store.get_vector_store_for_model(self._model_key)
        if self._model_vs is None:
            # LanceDB not available — fall back to brute-force
            self.use_vector_store = False
            return

        table_ready = (
            getattr(self._model_vs, "_table", None) is not None
            and self._model_vs.count() > 0
        )

        if not table_ready:
            # Table is empty / missing — auto re-ingest from SQLite
            node_count = self.store.get_nodes()
            if node_count:
                logger.info(
                    "No vector table for model '%s' — re-ingesting %d nodes…",
                    self._model_key, len(node_count),
                )
                n = self.store.reingest_for_model(
                    self._model_key, self.embedding_model,
                )
                if n > 0:
                    # Refresh the reference after ingestion
                    self._model_vs = self.store.get_vector_store_for_model(
                        self._model_key,
                    )

        self.use_vector_store = (
            self._model_vs is not None
            and getattr(self._model_vs, "_table", None) is not None
        )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, query: str, top_k: int, node_type: Optional[str]) -> str:
        return f"{query}||{top_k}||{node_type or ''}"

    def _cache_get(self, key: str) -> Optional[List[SearchResult]]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(self, key: str, value: List[SearchResult]) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > _CACHE_SIZE:
            self._cache.popitem(last=False)

    def clear_cache(self) -> None:
        """Flush the query result cache."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Primary search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        node_type: Optional[str] = None,
        file_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """Semantic search for code nodes.

        Args:
            query:       Natural-language or code query.
            top_k:       Number of results.
            node_type:   Optional filter (``function``, ``class``, ``module``).
            file_filter: Optional file-path SQL pattern,
                         e.g. ``"src/%"`` to restrict to files under *src/*.

        Returns:
            List of :class:`SearchResult` sorted by relevance (highest first).
        """
        ck = self._cache_key(query, top_k, node_type)
        cached = self._cache_get(ck)
        if cached is not None:
            return cached

        query_emb: List[float] = self.embedding_model.embed_text(query)

        if self.use_vector_store:
            results = self._search_vector_store(
                query_emb, top_k, node_type, file_filter,
            )
            # Fall back to brute-force if the vector store returned nothing
            # (e.g. empty table, dimension mismatch, or LanceDB error).
            if not results:
                results = self._search_brute_force(query_emb, top_k, node_type)
        else:
            results = self._search_brute_force(query_emb, top_k, node_type)

        # ── Graph-neighbour augmentation ────────────────────────
        if self.enable_graph_augment and results:
            results = self._augment_with_graph_neighbours(
                results, query_emb, top_k,
            )

        # ── Minimum-score gate ──────────────────────────────────
        results = [r for r in results if r.score >= self.min_score]

        # ── Final sort & trim ───────────────────────────────────
        results.sort(key=lambda r: r.score, reverse=True)
        results = results[:top_k]

        self._cache_put(ck, results)
        return results

    # ------------------------------------------------------------------
    # LanceDB path (fast)
    # ------------------------------------------------------------------

    def _search_vector_store(
        self,
        query_emb: List[float],
        top_k: int,
        node_type: Optional[str],
        file_filter: Optional[str],
    ) -> List[SearchResult]:
        assert self._model_vs is not None

        # Build SQL WHERE clause for hybrid search
        clauses: List[str] = []
        if node_type:
            clauses.append(f'node_type = "{node_type}"')
        if file_filter:
            clauses.append(f'file_path LIKE "{file_filter}"')
        where_sql = " AND ".join(clauses) if clauses else None

        raw_results = self._model_vs.hybrid_search(
            query_embedding=query_emb,
            n_results=top_k,
            where_sql=where_sql,
        )

        results: List[SearchResult] = []
        for row in raw_results:
            distance = row.get("_distance", 0.0)
            # With cosine metric, _distance is cosine distance ∈ [0, 2].
            # Similarity = 1 − distance, clamped to [0, 1].
            score = max(0.0, min(1.0, 1.0 - distance))

            # Enrich from SQLite if full node data is needed
            node_row = self.store.get_node(row.get("id", ""))

            if node_row is not None:
                results.append(SearchResult(
                    node_id=node_row["node_id"],
                    score=score,
                    node_type=node_row["node_type"],
                    qualname=node_row["qualname"],
                    file_path=node_row["file_path"],
                    start_line=node_row["start_line"],
                    end_line=node_row["end_line"],
                    snippet=node_row["code"],
                ))
            else:
                # Use data straight from LanceDB
                results.append(SearchResult(
                    node_id=row.get("id", ""),
                    score=score,
                    node_type=row.get("node_type", ""),
                    qualname=row.get("qualname", ""),
                    file_path=row.get("file_path", ""),
                    start_line=0,
                    end_line=0,
                    snippet=row.get("document", ""),
                ))

        return results

    # ------------------------------------------------------------------
    # Brute-force fallback
    # ------------------------------------------------------------------

    def _search_brute_force(
        self,
        query_emb: List[float],
        top_k: int,
        node_type: Optional[str],
    ) -> List[SearchResult]:
        results: List[SearchResult] = []
        query_dim = len(query_emb)
        for row in self.store.get_nodes():
            if node_type and row["node_type"] != node_type:
                continue
            embedding = json.loads(row["embedding"] or "[]")
            if not embedding:
                continue
            # Skip rows whose stored embedding dimension doesn't match
            if len(embedding) != query_dim:
                continue
            score = cosine_similarity(query_emb, embedding)
            if score < self.min_score:
                continue
            results.append(SearchResult(
                node_id=row["node_id"],
                score=score,
                node_type=row["node_type"],
                qualname=row["qualname"],
                file_path=row["file_path"],
                start_line=row["start_line"],
                end_line=row["end_line"],
                snippet=row["code"],
            ))

        return sorted(results, key=lambda r: r.score, reverse=True)[:top_k]

    # ------------------------------------------------------------------
    # Graph-neighbour augmentation
    # ------------------------------------------------------------------

    def _augment_with_graph_neighbours(
        self,
        results: List[SearchResult],
        query_emb: List[float],
        max_total: int,
    ) -> List[SearchResult]:
        """Expand the result set with direct dependency neighbours.

        For the top-3 semantic hits, fetch their outgoing and incoming
        graph edges and score the neighbour nodes against the query.
        Merge into *results* (deduplicated by ``node_id``).
        """
        seen_ids: Set[str] = {r.node_id for r in results}
        extra: List[SearchResult] = []

        # Only augment from the best 3 hits to keep it fast
        for sr in results[:3]:
            for edge in self.store.neighbors(sr.node_id):
                dst_id = edge["dst"]
                if dst_id in seen_ids:
                    continue
                seen_ids.add(dst_id)
                node_row = self.store.get_node(dst_id)
                if node_row is None:
                    continue
                emb = json.loads(node_row["embedding"] or "[]")
                if emb:
                    score = cosine_similarity(query_emb, emb)
                else:
                    score = sr.score * 0.3
                extra.append(SearchResult(
                    node_id=node_row["node_id"],
                    score=score,
                    node_type=node_row["node_type"],
                    qualname=node_row["qualname"],
                    file_path=node_row["file_path"],
                    start_line=node_row["start_line"],
                    end_line=node_row["end_line"],
                    snippet=node_row["code"],
                ))

            for edge in self.store.reverse_neighbors(sr.node_id):
                src_id = edge["src"]
                if src_id in seen_ids:
                    continue
                seen_ids.add(src_id)
                node_row = self.store.get_node(src_id)
                if node_row is None:
                    continue
                emb = json.loads(node_row["embedding"] or "[]")
                if emb:
                    score = cosine_similarity(query_emb, emb)
                else:
                    score = sr.score * 0.3
                extra.append(SearchResult(
                    node_id=node_row["node_id"],
                    score=score,
                    node_type=node_row["node_type"],
                    qualname=node_row["qualname"],
                    file_path=node_row["file_path"],
                    start_line=node_row["start_line"],
                    end_line=node_row["end_line"],
                    snippet=node_row["code"],
                ))

        return results + extra

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        node_type: Optional[str] = None,
        file_filter: Optional[str] = None,
    ) -> str:
        """Return a formatted string of the top search results.

        Useful for injecting relevant code context into LLM prompts.
        Applies context compression: strips imports, trims long code,
        and formats structured metadata.
        """
        matches = self.search(
            query, top_k=top_k, node_type=node_type, file_filter=file_filter,
        )
        if not matches:
            return "No relevant nodes found."

        blocks: List[str] = []
        for item in matches:
            snippet = _compress_snippet(item.snippet)
            blocks.append(
                f"[{item.node_type}] {item.qualname}\n"
                f"file: {item.file_path}:{item.start_line}\n"
                f"score: {item.score:.3f}\n"
                f"```python\n{snippet}\n```"
            )
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    # Debug helper
    # ------------------------------------------------------------------

    def debug_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Diagnostic search — returns raw dicts with full scoring info.

        Intended for ``cg debug-rag`` CLI command.
        """
        results = self.search(query, top_k=top_k)
        out: List[Dict[str, Any]] = []
        for r in results:
            out.append({
                "node_id": r.node_id,
                "qualname": r.qualname,
                "node_type": r.node_type,
                "file_path": r.file_path,
                "score": round(r.score, 5),
                "lines": f"{r.start_line}-{r.end_line}",
                "snippet_len": len(r.snippet),
            })
        return out


# ===================================================================
# Context compression utilities
# ===================================================================

def _compress_snippet(code: str, max_chars: int = _MAX_SNIPPET_CHARS) -> str:
    """Clean and truncate a code snippet for LLM context.

    1. Strip bare import lines (the LLM rarely needs them).
    2. Collapse runs of blank lines.
    3. Truncate to *max_chars*.
    """
    code = _IMPORT_RE.sub("", code)
    code = re.sub(r"\n{3,}", "\n\n", code).strip()
    if len(code) > max_chars:
        code = code[:max_chars] + "\n# ... (truncated)"
    return code
