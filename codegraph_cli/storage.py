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

        # Initialise LanceDB vector store
        self.vector_store: Optional[VectorStore] = None
        if VECTOR_STORE_AVAILABLE:
            try:
                self.vector_store = VectorStore(project_dir)
            except Exception as exc:
                logger.warning("LanceDB vector store unavailable: %s", exc)

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

    def insert_nodes(self, rows: Iterable[Tuple[Node, List[float]]]) -> None:
        """Insert nodes with their embedding vectors.

        Each element of *rows* is a ``(Node, embedding)`` tuple.  Data is
        written to both SQLite (for structured queries) and LanceDB (for
        vector search).
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
        if self.vector_store is not None:
            try:
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
                self.vector_store.add_nodes(
                    node_ids, embeddings, metadatas, documents,
                )
            except Exception as exc:
                logger.warning("Failed to sync nodes to LanceDB: %s", exc)

    def insert_edges(self, edges: Iterable[Edge]) -> None:
        cur = self.conn.cursor()
        cur.executemany(
            "INSERT INTO edges (src, dst, edge_type) VALUES (?, ?, ?)",
            [(e.src, e.dst, e.edge_type) for e in edges],
        )
        self.conn.commit()

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
