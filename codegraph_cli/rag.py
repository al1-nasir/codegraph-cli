"""Retrieval-Augmented components for semantic code search.

Uses LanceDB hybrid search (vector + metadata filters) for fast,
accurate code retrieval.  Falls back to brute-force cosine similarity
when the vector store is unavailable.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Union

from .embeddings import HashEmbeddingModel, TransformerEmbedder, cosine_similarity
from .models import SearchResult
from .storage import GraphStore

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Retrieve relevant code nodes from graph memory via semantic similarity.

    Supports two modes:

    1. **Vector store mode** (fast, preferred) – delegates to LanceDB via
       ``GraphStore.vector_store``.
    2. **Brute-force mode** (fallback) – scans all SQLite rows and computes
       cosine similarity in Python.

    The ``embedding_model`` argument accepts either a
    :class:`~codegraph_cli.embeddings.TransformerEmbedder` or the lightweight
    :class:`~codegraph_cli.embeddings.HashEmbeddingModel`.
    """

    def __init__(
        self,
        store: GraphStore,
        embedding_model: Union[TransformerEmbedder, HashEmbeddingModel, Any],
    ) -> None:
        self.store = store
        self.embedding_model = embedding_model
        self.use_vector_store: bool = store.vector_store is not None

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
        query_emb: List[float] = self.embedding_model.embed_text(query)

        if self.use_vector_store:
            return self._search_vector_store(
                query_emb, top_k, node_type, file_filter,
            )
        return self._search_brute_force(query_emb, top_k, node_type)

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
        assert self.store.vector_store is not None

        # Build SQL WHERE clause for hybrid search
        clauses: List[str] = []
        if node_type:
            clauses.append(f'node_type = "{node_type}"')
        if file_filter:
            clauses.append(f'file_path LIKE "{file_filter}"')
        where_sql = " AND ".join(clauses) if clauses else None

        raw_results = self.store.vector_store.hybrid_search(
            query_embedding=query_emb,
            n_results=top_k,
            where_sql=where_sql,
        )

        results: List[SearchResult] = []
        for row in raw_results:
            distance = row.get("_distance", 0.0)
            # LanceDB returns L2 distance by default; convert to a similarity
            # score in [0, 1].  For cosine distance the relationship is
            # score = 1 - distance  (since embeddings are unit-normalised).
            score = max(0.0, 1.0 - distance)

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
        for row in self.store.get_nodes():
            if node_type and row["node_type"] != node_type:
                continue
            embedding = json.loads(row["embedding"] or "[]")
            score = cosine_similarity(query_emb, embedding)
            if score <= 0:
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
        """
        matches = self.search(
            query, top_k=top_k, node_type=node_type, file_filter=file_filter,
        )
        if not matches:
            return "No relevant nodes found."

        blocks: List[str] = []
        for item in matches:
            blocks.append(
                f"[{item.node_type}] {item.qualname} "
                f"({item.file_path}:{item.start_line})\n"
                f"Score: {item.score:.3f}\n"
                f"```python\n{item.snippet[:1200]}\n```"
            )
        return "\n\n".join(blocks)
