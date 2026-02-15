"""Persistence layer for project-specific code graph memory.

Architecture:
- **SQLite** for structured data (nodes, edges) and graph traversal queries.
- **LanceDB** (via :class:`~codegraph_cli.vector_store.VectorStore`) for
  vector similarity search.

This hybrid approach gives the best of both worlds: fast relational queries
for graph traversal and fast ANN search for semantic retrieval.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import MEMORY_DIR, STATE_FILE, ensure_base_dirs
from .models import Edge, Node

logger = logging.getLogger(__name__)

try:
    from .vector_store import VectorStore
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False


# ===================================================================
# ProjectManager  (unchanged – manages directories / active project)
# ===================================================================

class ProjectManager:
    """Manage project memory directories and active project state."""

    def __init__(self) -> None:
        ensure_base_dirs()

    def list_projects(self) -> List[str]:
        if not MEMORY_DIR.exists():
            return []
        return sorted([p.name for p in MEMORY_DIR.iterdir() if p.is_dir()])

    def project_dir(self, project_name: str) -> Path:
        return MEMORY_DIR / project_name

    def create_or_get_project(self, project_name: str) -> Path:
        path = self.project_dir(project_name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def set_current_project(self, project_name: str) -> None:
        ensure_base_dirs()
        STATE_FILE.write_text(
            json.dumps({"current_project": project_name}, indent=2),
            encoding="utf-8",
        )

    def get_current_project(self) -> Optional[str]:
        if not STATE_FILE.exists():
            return None
        try:
            payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return payload.get("current_project")

    def unload_project(self) -> None:
        ensure_base_dirs()
        STATE_FILE.write_text(
            json.dumps({"current_project": None}, indent=2),
            encoding="utf-8",
        )

    def delete_project(self, project_name: str) -> bool:
        path = self.project_dir(project_name)
        if not path.exists():
            return False
        for child in sorted(path.glob("**/*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        path.rmdir()
        return True


# ===================================================================
# GraphStore  (SQLite + LanceDB hybrid)
# ===================================================================

class GraphStore:
    """Hybrid store: SQLite for structure, LanceDB for vectors.

    Public API is backward-compatible with the legacy SQLite-only store.
    The ``vector_store`` attribute exposes the underlying
    :class:`~codegraph_cli.vector_store.VectorStore` for direct vector
    search when needed.
    """

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir
        self.db_path = project_dir / "graph.db"
        self.meta_path = project_dir / "project.json"
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

        # Initialise LanceDB vector store (default / legacy table)
        self.vector_store: Optional[VectorStore] = None
        if VECTOR_STORE_AVAILABLE:
            try:
                self.vector_store = VectorStore(project_dir)
            except Exception as exc:
                logger.warning("LanceDB vector store unavailable: %s", exc)

        # Per-model vector store cache: model_key → VectorStore
        self._model_vector_stores: Dict[str, "VectorStore"] = {}

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                node_id   TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                name      TEXT NOT NULL,
                qualname  TEXT NOT NULL,
                file_path TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line   INTEGER NOT NULL,
                code      TEXT NOT NULL,
                docstring TEXT,
                embedding TEXT,
                metadata  TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                src       TEXT NOT NULL,
                dst       TEXT NOT NULL,
                edge_type TEXT NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_qualname ON nodes(qualname)")
        self.conn.commit()

    # ------------------------------------------------------------------
    # Clear / metadata
    # ------------------------------------------------------------------

    def clear(self) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM edges")
        cur.execute("DELETE FROM nodes")
        self.conn.commit()
        if self.vector_store is not None:
            try:
                self.vector_store.clear()
            except Exception:
                pass

    def set_metadata(self, payload: Dict[str, Any]) -> None:
        self.meta_path.write_text(
            json.dumps(payload, indent=2), encoding="utf-8",
        )

    def get_metadata(self) -> Dict[str, Any]:
        if not self.meta_path.exists():
            return {}
        try:
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert_nodes(
        self,
        rows: Iterable[Tuple[Node, List[float]]],
        model_key: Optional[str] = None,
    ) -> None:
        """Insert nodes with their embedding vectors.

        Each element of *rows* is a ``(Node, embedding)`` tuple.  Data is
        written to both SQLite (for structured queries) and LanceDB (for
        vector search).

        When *model_key* is provided the embeddings are also written to
        the model-specific LanceDB table (``code_nodes_{model_key}``).
        """
        rows_list = list(rows)
        if not rows_list:
            return

        # ---- SQLite -----------------------------------------------------
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO nodes (
                node_id, node_type, name, qualname, file_path,
                start_line, end_line, code, docstring, embedding, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    node.node_id,
                    node.node_type,
                    node.name,
                    node.qualname,
                    node.file_path,
                    node.start_line,
                    node.end_line,
                    node.code,
                    node.docstring,
                    json.dumps(embedding),
                    json.dumps(node.metadata) if node.metadata else None,
                )
                for node, embedding in rows_list
            ],
        )
        self.conn.commit()

        # ---- LanceDB (vector store) ------------------------------------
        node_ids = [node.node_id for node, _ in rows_list]
        embeddings = [emb for _, emb in rows_list]
        metadatas = [
            {
                "node_type": node.node_type,
                "file_path": node.file_path,
                "qualname": node.qualname,
                "name": node.name,
            }
            for node, _ in rows_list
        ]
        documents = [node.code for node, _ in rows_list]

        # Write to legacy table (backward compat)
        if self.vector_store is not None:
            try:
                self.vector_store.add_nodes(
                    node_ids, embeddings, metadatas, documents,
                )
            except Exception as exc:
                logger.warning("Failed to sync nodes to LanceDB: %s", exc)

        # Write to model-specific table
        if model_key:
            model_vs = self.get_vector_store_for_model(model_key)
            if model_vs is not None:
                try:
                    model_vs.add_nodes(
                        node_ids, embeddings, metadatas, documents,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to sync nodes to model table '%s': %s",
                        model_key, exc,
                    )

    def insert_edges(self, edges: Iterable[Edge]) -> None:
        cur = self.conn.cursor()
        cur.executemany(
            "INSERT INTO edges (src, dst, edge_type) VALUES (?, ?, ?)",
            [(e.src, e.dst, e.edge_type) for e in edges],
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Incremental index (single-file add / remove)
    # ------------------------------------------------------------------

    def remove_nodes_for_file(self, rel_path: str) -> int:
        """Remove all nodes and related edges for a specific file.

        Clears data from SQLite **and** every known LanceDB table
        (legacy + per-model).

        Args:
            rel_path: Relative file path as stored in the ``file_path``
                      column (e.g. ``"src/utils.py"``).

        Returns:
            Number of SQLite node rows deleted.
        """
        # 1. Collect node IDs that belong to this file
        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT node_id FROM nodes WHERE file_path = ?", (rel_path,),
        ).fetchall()
        node_ids = [r[0] for r in rows]

        if not node_ids:
            return 0

        # 2. Delete edges referencing these nodes (src OR dst)
        placeholders = ",".join("?" * len(node_ids))
        cur.execute(
            f"DELETE FROM edges WHERE src IN ({placeholders}) OR dst IN ({placeholders})",
            node_ids + node_ids,
        )
        # 3. Delete nodes themselves
        cur.execute(
            f"DELETE FROM nodes WHERE node_id IN ({placeholders})",
            node_ids,
        )
        self.conn.commit()

        # 4. Remove from legacy LanceDB table
        if self.vector_store is not None:
            try:
                self.vector_store.delete_by_file_path(rel_path)
            except Exception as exc:
                logger.debug("Legacy vector delete for '%s': %s", rel_path, exc)

        # 5. Remove from all per-model LanceDB tables
        for _key, vs in self._model_vector_stores.items():
            try:
                vs.delete_by_file_path(rel_path)
            except Exception:
                pass

        # Also try tables that haven't been opened yet
        if VECTOR_STORE_AVAILABLE:
            try:
                probe = VectorStore(self.project_dir, model_key="")
                for mk in probe.list_model_tables():
                    if mk and mk not in self._model_vector_stores:
                        try:
                            vs = VectorStore(self.project_dir, model_key=mk)
                            vs.delete_by_file_path(rel_path)
                        except Exception:
                            pass
            except Exception:
                pass

        return len(node_ids)

    def index_single_file(
        self,
        file_path: Path,
        project_root: Path,
        embedder: Any,
        model_key: str = "",
    ) -> int:
        """Parse and index a single file incrementally.

        Removes old nodes/edges for the file, parses it fresh,
        embeds the new nodes, and inserts them.

        Args:
            file_path:    Absolute path to the source file.
            project_root: Project root (for computing relative paths).
            embedder:     Object with ``embed_text(str) -> List[float]``.
            model_key:    Embedding model identifier.

        Returns:
            Number of nodes indexed for this file.
        """
        from .parser import PythonGraphParser
        from .agents import _build_chunk_text

        rel_path = str(file_path.relative_to(project_root))

        # Remove stale data for this file
        self.remove_nodes_for_file(rel_path)

        # Parse the single file
        parser = PythonGraphParser(project_root)
        try:
            nodes, edges = parser.parse_file(file_path)
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", file_path, exc)
            return 0

        if not nodes:
            return 0

        # Embed and insert
        node_payload = []
        for node in nodes:
            text = _build_chunk_text(node)
            emb = embedder.embed_text(text)
            node_payload.append((node, emb))

        self.insert_nodes(node_payload, model_key=model_key)
        self.insert_edges(edges)

        logger.info(
            "Incremental index: %d nodes, %d edges for %s",
            len(nodes), len(edges), rel_path,
        )
        return len(nodes)

    # ------------------------------------------------------------------
    # Read (structured)
    # ------------------------------------------------------------------

    def get_nodes(self) -> List[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM nodes").fetchall()

    def get_node(self, node_id_or_name: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM nodes WHERE node_id = ? OR qualname = ? OR name = ? LIMIT 1",
            (node_id_or_name, node_id_or_name, node_id_or_name),
        ).fetchone()

    def get_edges(self) -> List[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM edges").fetchall()

    def neighbors(self, src_node_id: str) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM edges WHERE src = ?", (src_node_id,),
        ).fetchall()

    def reverse_neighbors(self, dst_node_id: str) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM edges WHERE dst = ?", (dst_node_id,),
        ).fetchall()

    def all_by_file(self) -> Dict[str, List[Dict[str, Any]]]:
        by_file: Dict[str, List[Dict[str, Any]]] = {}
        for row in self.get_nodes():
            payload = dict(row)
            by_file.setdefault(payload["file_path"], []).append(payload)
        return by_file

    # ------------------------------------------------------------------
    # Vector search (convenience wrappers)
    # ------------------------------------------------------------------

    def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 10,
        where: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search via LanceDB.

        Args:
            query_vector: Embedding of the search query.
            top_k:        Max results.
            where:        Metadata filter dict, e.g. ``{"node_type": "function"}``.

        Returns:
            List of result dicts with ``id``, ``_distance``, ``document``, etc.
        """
        if self.vector_store is None:
            return []
        result = self.vector_store.search(query_vector, n_results=top_k, where=where)
        # Flatten the nested Chroma-compat format into a plain list
        out: List[Dict[str, Any]] = []
        if result["ids"] and result["ids"][0]:
            for i, nid in enumerate(result["ids"][0]):
                out.append({
                    "id": nid,
                    "_distance": result["distances"][0][i] if result["distances"][0] else 0.0,
                    "metadata": result["metadatas"][0][i] if result["metadatas"][0] else {},
                    "document": result["documents"][0][i] if result["documents"][0] else "",
                })
        return out

    def hybrid_search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        where_sql: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid search: vector + SQL filter (e.g. ``file_path LIKE 'src/%'``).

        Falls back to :meth:`search_vectors` when where_sql is ``None``.
        """
        if self.vector_store is None:
            return []
        return self.vector_store.hybrid_search(
            query_vector, n_results=top_k, where_sql=where_sql,
        )

    # ------------------------------------------------------------------
    # Merge (cross-project)
    # ------------------------------------------------------------------

    def merge_from(self, other: "GraphStore", source_project: str) -> None:
        current_nodes = self.conn.execute("SELECT node_id FROM nodes").fetchall()
        existing = {r[0] for r in current_nodes}

        node_rows: List[Dict[str, Any]] = []
        id_map: Dict[str, str] = {}
        for row in other.get_nodes():
            row_dict = dict(row)
            original_id = row_dict["node_id"]
            new_id = original_id if original_id not in existing else f"{source_project}:{original_id}"
            existing.add(new_id)
            id_map[original_id] = new_id
            metadata = json.loads(row_dict.get("metadata") or "{}")
            metadata["merged_from"] = source_project
            row_dict["node_id"] = new_id
            row_dict["metadata"] = json.dumps(metadata)
            node_rows.append(row_dict)

        cur = self.conn.cursor()
        cur.executemany(
            """
            INSERT OR REPLACE INTO nodes (
                node_id, node_type, name, qualname, file_path,
                start_line, end_line, code, docstring, embedding, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    n["node_id"], n["node_type"], n["name"], n["qualname"],
                    n["file_path"], n["start_line"], n["end_line"], n["code"],
                    n["docstring"], n["embedding"], n["metadata"],
                )
                for n in node_rows
            ],
        )

        edge_rows: List[Tuple[str, str, str]] = []
        for edge_row in other.get_edges():
            e = dict(edge_row)
            src = id_map.get(e["src"], e["src"])
            dst = id_map.get(e["dst"], e["dst"])
            edge_rows.append((src, dst, e["edge_type"]))
        cur.executemany(
            "INSERT INTO edges (src, dst, edge_type) VALUES (?, ?, ?)",
            edge_rows,
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Per-model vector stores  (auto re-ingestion)
    # ------------------------------------------------------------------

    def get_vector_store_for_model(self, model_key: str) -> Optional["VectorStore"]:
        """Get (or create) a LanceDB vector store for a specific embedding model.

        Each embedding model gets its own LanceDB table so that
        different dimensionalities never collide.  The table is named
        ``code_nodes_{model_key}``.

        Returns ``None`` when LanceDB is not available.
        """
        if not VECTOR_STORE_AVAILABLE:
            return None
        if model_key in self._model_vector_stores:
            return self._model_vector_stores[model_key]
        try:
            vs = VectorStore(self.project_dir, model_key=model_key)
            self._model_vector_stores[model_key] = vs
            return vs
        except Exception as exc:
            logger.warning(
                "Cannot create vector store for model '%s': %s", model_key, exc,
            )
            return None

    def reingest_for_model(
        self,
        model_key: str,
        embedder: Any,
        chunk_builder: Any = None,
    ) -> int:
        """Re-embed all SQLite nodes into a model-specific LanceDB table.

        Reads raw code/metadata from the SQLite ``nodes`` table,
        computes embeddings with *embedder*, and writes them into the
        LanceDB table for *model_key*.

        Args:
            model_key:     Embedding model identifier (e.g. ``"minilm"``).
            embedder:      Object with an ``embed_text(str) -> List[float]``
                           method (and optionally ``embed_documents``).
            chunk_builder: Optional callable ``(dict) -> str`` that builds
                           the text chunk from a node row dict.  Falls back
                           to an internal default.

        Returns:
            Number of nodes ingested.
        """
        vs = self.get_vector_store_for_model(model_key)
        if vs is None:
            return 0

        rows = self.get_nodes()
        if not rows:
            return 0

        if chunk_builder is None:
            chunk_builder = _default_chunk_builder

        # Clear old data for this model's table and re-open
        vs.clear()
        self._model_vector_stores.pop(model_key, None)
        vs = self.get_vector_store_for_model(model_key)
        if vs is None:
            return 0

        node_ids: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict[str, str]] = []
        documents: List[str] = []
        texts: List[str] = []

        for row in rows:
            row_dict = dict(row)
            text = chunk_builder(row_dict)
            texts.append(text)
            node_ids.append(row_dict["node_id"])
            metadatas.append({
                "node_type": row_dict["node_type"],
                "file_path": row_dict["file_path"],
                "qualname": row_dict["qualname"],
                "name": row_dict["name"],
            })
            documents.append(row_dict["code"])

        # Batch-embed when possible, single-embed otherwise
        if hasattr(embedder, "embed_documents"):
            embeddings = embedder.embed_documents(texts)
        else:
            embeddings = [embedder.embed_text(t) for t in texts]

        try:
            vs.add_nodes(node_ids, embeddings, metadatas, documents)
            logger.info(
                "Re-ingested %d nodes into table for embedding model '%s'.",
                len(node_ids), model_key,
            )
        except Exception as exc:
            logger.warning("Re-ingestion for model '%s' failed: %s", model_key, exc)
            return 0

        return len(node_ids)


# ===================================================================
# Helpers
# ===================================================================

# Regex to strip bare import lines from chunk text (mirrors agents._IMPORT_RE)
_CHUNK_IMPORT_RE = re.compile(r"^(?:from\s+\S+\s+)?import\s+.+$", re.MULTILINE)
_MAX_CHUNK_CODE = 1500


def _default_chunk_builder(row: Dict[str, Any]) -> str:
    """Build embedding text from a SQLite node row dict.

    Mirrors :func:`codegraph_cli.agents._build_chunk_text` but works
    with plain dicts instead of :class:`Node` objects.
    """
    parts: List[str] = [
        f"file: {row['file_path']}",
        f"symbol: {row['qualname']}",
        f"type: {row['node_type']}",
    ]
    docstring = row.get("docstring") or ""
    if docstring.strip():
        parts.append(f"doc: {docstring.strip()}")

    code: str = row.get("code", "")
    if row["node_type"] != "module":
        code = _CHUNK_IMPORT_RE.sub("", code).strip()
    else:
        code = code[:_MAX_CHUNK_CODE]

    if len(code) > _MAX_CHUNK_CODE:
        code = code[:_MAX_CHUNK_CODE] + "\n# ... (truncated)"
    if code:
        parts.append(code)

    return "\n".join(parts)
