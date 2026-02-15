"""Vector store backed by LanceDB – serverless, local-first vector database.

Replaces the Chroma-based vector store with LanceDB which offers:
- Zero-server architecture (embedded, like SQLite for vectors)
- Native hybrid search (vector + SQL predicate filtering)
- Lance columnar format for fast scans and efficient storage
- Full-text search index support

All data stays on disk under the project directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import lancedb  # type: ignore[import-untyped]
    import pyarrow as pa  # type: ignore[import-untyped]
    LANCE_AVAILABLE = True
except ImportError:
    LANCE_AVAILABLE = False

# Keep legacy name for backward compatibility
CHROMA_AVAILABLE = LANCE_AVAILABLE  # old code may check this


class VectorStore:
    """LanceDB-backed vector store for code embeddings.

    Drop-in replacement for the old Chroma-based ``VectorStore``.
    The public API is unchanged so existing callers continue to work.

    Schema per row:

    ======== ============ =====================================
    Column   Type         Description
    ======== ============ =====================================
    id       utf8         Unique node identifier
    vector   float32[dim] Embedding vector
    document utf8         Source code text
    node_type utf8        function / class / module
    file_path utf8        Relative file path
    qualname  utf8        Fully-qualified symbol name
    name      utf8        Short symbol name
    ======== ============ =====================================
    """

    def __init__(self, project_dir: Path, model_key: str = "") -> None:
        if not LANCE_AVAILABLE:
            raise ImportError(
                "lancedb is not installed. Install with: pip install lancedb pyarrow"
            )

        self.project_dir = project_dir
        self.model_key = model_key
        self._lance_dir = project_dir / "lancedb"
        self._lance_dir.mkdir(exist_ok=True, parents=True)

        # Each embedding model gets its own table to avoid dimension conflicts
        self._table_name = f"code_nodes_{model_key}" if model_key else "code_nodes"

        self._db: Any = lancedb.connect(str(self._lance_dir))
        self._table: Optional[Any] = None

        # Try to open existing table
        try:
            self._table = self._db.open_table(self._table_name)
        except Exception:
            self._table = None

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_nodes(
        self,
        node_ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, str]],
        documents: List[str],
    ) -> None:
        """Add or update node embeddings in the vector store.

        Args:
            node_ids:   Unique node identifiers.
            embeddings: Embedding vectors (all same dimensionality).
            metadatas:  Per-node metadata dicts.
            documents:  Source code text per node.
        """
        if not node_ids:
            return

        rows = [
            {
                "id": nid,
                "vector": emb,
                "document": doc,
                "node_type": meta.get("node_type", ""),
                "file_path": meta.get("file_path", ""),
                "qualname": meta.get("qualname", ""),
                "name": meta.get("name", ""),
            }
            for nid, emb, meta, doc in zip(node_ids, embeddings, metadatas, documents)
        ]

        if self._table is None:
            # First insert – create the table (schema inferred from data)
            self._table = self._db.create_table(
                self._table_name, data=rows, mode="overwrite",
            )
        else:
            # Subsequent inserts – upsert by deleting old IDs first
            try:
                existing_ids = set(node_ids)
                for nid in existing_ids:
                    try:
                        self._table.delete(f'id = "{nid}"')
                    except Exception:
                        pass
            except Exception:
                pass
            self._table.add(rows)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Vector similarity search with optional metadata filtering.

        Returns a dict with keys ``ids``, ``distances``, ``metadatas``,
        ``documents`` – matching the legacy Chroma return format for
        backward compatibility.

        Args:
            query_embedding: Query vector.
            n_results:       Max results to return.
            where:           Optional metadata filter, e.g.
                             ``{"node_type": "function"}``.
        """
        empty: Dict[str, Any] = {
            "ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]],
        }
        if self._table is None:
            return empty

        try:
            query = (
                self._table
                .search(query_embedding)
                .metric("cosine")
                .limit(n_results)
            )

            # Apply metadata filters as SQL WHERE clause
            if where:
                clauses = []
                for key, value in where.items():
                    clauses.append(f'{key} = "{value}"')
                if clauses:
                    query = query.where(" AND ".join(clauses))

            results = query.to_list()
        except Exception as exc:
            logger.warning("LanceDB search failed: %s", exc)
            return empty

        ids: List[str] = []
        distances: List[float] = []
        metas: List[Dict[str, str]] = []
        docs: List[str] = []

        for row in results:
            # With cosine metric, _distance is the *cosine distance*
            # (1 − cos_sim), so values are in [0, 2].
            dist = row.get("_distance", 0.0)
            ids.append(row.get("id", ""))
            distances.append(dist)
            metas.append({
                "node_type": row.get("node_type", ""),
                "file_path": row.get("file_path", ""),
                "qualname": row.get("qualname", ""),
                "name": row.get("name", ""),
            })
            docs.append(row.get("document", ""))

        return {
            "ids": [ids],
            "distances": [distances],
            "metadatas": [metas],
            "documents": [docs],
        }

    def hybrid_search(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where_sql: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid search: vector similarity + arbitrary SQL filter.

        Unlike :meth:`search`, this returns a plain list of dicts and
        accepts a raw SQL ``WHERE`` clause for maximum flexibility.

        Example::

            store.hybrid_search(
                vec, n_results=5, where_sql="file_path LIKE 'src/%'"
            )
        """
        if self._table is None:
            return []

        try:
            query = (
                self._table
                .search(query_embedding)
                .metric("cosine")
                .limit(n_results)
            )
            if where_sql:
                query = query.where(where_sql)
            return query.to_list()
        except Exception as exc:
            logger.warning("LanceDB hybrid search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Point lookups
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single node by its ID."""
        if self._table is None:
            return None
        try:
            import pandas as pd  # type: ignore[import-untyped]
            df: pd.DataFrame = self._table.to_pandas()
            match = df[df["id"] == node_id]
            if match.empty:
                return None
            row = match.iloc[0].to_dict()
            return {
                "id": row["id"],
                "embedding": row.get("vector"),
                "metadata": {
                    "node_type": row.get("node_type", ""),
                    "file_path": row.get("file_path", ""),
                    "qualname": row.get("qualname", ""),
                    "name": row.get("name", ""),
                },
                "document": row.get("document", ""),
            }
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Delete / clear
    # ------------------------------------------------------------------

    def delete_nodes(self, node_ids: List[str]) -> None:
        """Delete nodes by ID."""
        if not node_ids or self._table is None:
            return
        for nid in node_ids:
            try:
                self._table.delete(f'id = "{nid}"')
            except Exception:
                pass

    def delete_by_file_path(self, file_path: str) -> int:
        """Delete all nodes belonging to a specific file.

        Args:
            file_path: Relative file path (must match the ``file_path``
                       column stored during indexing).

        Returns:
            Number of rows deleted (0 if table is empty / missing).
        """
        if self._table is None:
            return 0
        try:
            before = self._table.count_rows()
            # Escape single quotes in the path to avoid SQL injection
            safe_path = file_path.replace("'", "''")
            self._table.delete(f"file_path = '{safe_path}'")
            after = self._table.count_rows()
            return max(0, before - after)
        except Exception as exc:
            logger.warning(
                "delete_by_file_path('%s') failed: %s", file_path, exc,
            )
            return 0

    def clear(self) -> None:
        """Drop all data and recreate an empty table."""
        try:
            self._db.drop_table(self._table_name)
        except Exception:
            pass
        self._table = None

    def list_model_tables(self) -> List[str]:
        """Return model keys for which a LanceDB table exists.

        Tables are named ``code_nodes_{model_key}``; this method strips
        the prefix and returns just the model keys.
        """
        try:
            all_tables = self._db.table_names()
        except Exception:
            return []
        models: List[str] = []
        prefix = "code_nodes_"
        for name in all_tables:
            if name == "code_nodes":
                models.append("")  # legacy table
            elif name.startswith(prefix):
                models.append(name[len(prefix):])
        return models

    # ------------------------------------------------------------------
    # Informational
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Number of rows in the vector store."""
        if self._table is None:
            return 0
        try:
            return self._table.count_rows()
        except Exception:
            return 0

    def peek(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return a sample of rows for debugging."""
        if self._table is None:
            return []
        try:
            import pandas as pd  # type: ignore[import-untyped]
            df: pd.DataFrame = self._table.to_pandas()
            return df.head(limit).to_dict(orient="records")
        except Exception:
            return []

    def debug_search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """Diagnostic search returning raw scores and distance details.

        Unlike :meth:`search`, this returns a flat list of dicts that
        includes the raw ``_distance`` value, the derived similarity
        score, and key metadata — useful for inspecting retrieval
        quality from the CLI.
        """
        if self._table is None:
            return []
        try:
            results = (
                self._table
                .search(query_embedding)
                .metric("cosine")
                .limit(n_results)
                .to_list()
            )
        except Exception as exc:
            logger.warning("debug_search failed: %s", exc)
            return []

        out: List[Dict[str, Any]] = []
        for row in results:
            dist = row.get("_distance", 0.0)
            out.append({
                "id": row.get("id", ""),
                "name": row.get("name", ""),
                "qualname": row.get("qualname", ""),
                "node_type": row.get("node_type", ""),
                "file_path": row.get("file_path", ""),
                "cosine_distance": round(dist, 5),
                "similarity_score": round(max(0.0, 1.0 - dist), 5),
                "document_preview": (row.get("document", "") or "")[:120],
            })
        return out
