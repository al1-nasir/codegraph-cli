"""Multi-agent components: graph indexing, retrieval, and summarization."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, List, Set

from .embeddings import HashEmbeddingModel
from .llm import LocalLLM
from .models import ImpactReport
from .parser import PythonGraphParser
from .rag import RAGRetriever
from .storage import GraphStore


class GraphAgent:
    """Responsible for parsing projects and maintaining graph memory."""

    def __init__(self, store: GraphStore, embedding_model: HashEmbeddingModel):
        self.store = store
        self.embedding_model = embedding_model

    def index_project(self, project_root: Path, show_progress: bool = True) -> Dict[str, int]:
        parser = PythonGraphParser(project_root)
        nodes, edges = parser.parse_project()

        self.store.clear()

        node_payload = []
        total_nodes = len(nodes)
        
        for idx, node in enumerate(nodes, 1):
            text = "\n".join([node.qualname, node.docstring, node.code])
            emb = self.embedding_model.embed_text(text)
            node_payload.append((node, emb))
            
            # Show progress
            if show_progress and idx % max(1, total_nodes // 20) == 0:
                progress = (idx / total_nodes) * 100
                print(f"\r📊 Indexing: {idx}/{total_nodes} nodes ({progress:.0f}%)", end="", flush=True)
        
        if show_progress:
            print(f"\r📊 Indexing: {total_nodes}/{total_nodes} nodes (100%)  ")

        self.store.insert_nodes(node_payload)
        self.store.insert_edges(edges)
        self.store.set_metadata(
            {
                "project_root": str(project_root),
                "node_count": len(nodes),
                "edge_count": len(edges),
            }
        )
        return {"nodes": len(nodes), "edges": len(edges)}

    def ascii_neighbors(self, symbol: str, depth: int = 1) -> str:
        node = self.store.get_node(symbol)
        if not node:
            return f"Symbol '{symbol}' not found in current project."

        start = node["node_id"]
        lines = [f"{node['qualname']} ({node['node_type']})"]

        frontier = [(start, 0)]
        seen = {start}
        while frontier:
            current, level = frontier.pop(0)
            if level >= depth:
                continue
            for edge in self.store.neighbors(current):
                dst = edge["dst"]
                dst_node = self.store.get_node(dst)
                label = dst_node["qualname"] if dst_node else dst
                lines.append(f"{'  ' * (level + 1)}|-{edge['edge_type']}-> {label}")
                if dst not in seen:
                    seen.add(dst)
                    frontier.append((dst, level + 1))
        return "\n".join(lines)


class RAGAgent:
    """Runs semantic retrieval against project memory."""

    def __init__(self, retriever: RAGRetriever):
        self.retriever = retriever

    def semantic_search(self, query: str, top_k: int = 5, node_type: str = None):
        """Perform semantic search with optional node type filtering.
        
        Args:
            query: Search query
            top_k: Number of results
            node_type: Optional filter (function, class, module)
        """
        return self.retriever.search(query, top_k=top_k, node_type=node_type)

    def context_for_query(self, query: str, top_k: int = 6) -> str:
        return self.retriever.retrieve_context(query, top_k=top_k)


class SummarizationAgent:
    """Uses retrieved graph context + local LLM for reasoning/explanations."""

    def __init__(self, store: GraphStore, llm: LocalLLM):
        self.store = store
        self.llm = llm

    def impact_analysis(self, symbol: str, hops: int = 2) -> ImpactReport:
        root = self.store.get_node(symbol)
        if not root:
            message = f"Symbol '{symbol}' not found in current project."
            return ImpactReport(root=symbol, impacted=[], explanation=message, ascii_graph=message)

        root_id = root["node_id"]
        impacted_ids = self._multi_hop(root_id, hops)
        impacted_rows = [self.store.get_node(node_id) for node_id in impacted_ids]
        impacted_rows = [row for row in impacted_rows if row is not None and row["node_id"] != root_id]

        impacted_names = [row["qualname"] for row in impacted_rows]
        ascii_graph = self._impact_ascii(root_id, hops)

        prompt = self._build_impact_prompt(root, impacted_rows, ascii_graph)
        explanation = self.llm.explain(prompt)

        return ImpactReport(
            root=root["qualname"],
            impacted=impacted_names,
            explanation=explanation,
            ascii_graph=ascii_graph,
        )

    def _multi_hop(self, start_node_id: str, hops: int) -> Set[str]:
        seen = {start_node_id}
        queue = deque([(start_node_id, 0)])

        while queue:
            current, depth = queue.popleft()
            if depth >= hops:
                continue
            for edge in self.store.neighbors(current):
                nxt = edge["dst"]
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, depth + 1))
        return seen

    def _impact_ascii(self, start_node_id: str, hops: int) -> str:
        lines: List[str] = []
        queue = deque([(start_node_id, 0)])
        seen = {start_node_id}

        while queue:
            current, depth = queue.popleft()
            node = self.store.get_node(current)
            if not node:
                continue
            prefix = "  " * depth
            lines.append(f"{prefix}{node['qualname']}")
            if depth >= hops:
                continue
            for edge in self.store.neighbors(current):
                dst = edge["dst"]
                dst_node = self.store.get_node(dst)
                label = dst_node["qualname"] if dst_node else dst
                lines.append(f"{prefix}  |- {edge['edge_type']} -> {label}")
                if dst not in seen:
                    seen.add(dst)
                    queue.append((dst, depth + 1))
        return "\n".join(lines)

    def _build_impact_prompt(self, root_row, impacted_rows, ascii_graph: str) -> str:
        impacted_block = "\n".join(
            [
                f"- {row['qualname']} ({row['file_path']}:{row['start_line']})"
                for row in impacted_rows[:20]
            ]
        )
        return (
            "You are a local code reasoning assistant. "
            "Explain the likely downstream impact of changing a symbol.\n\n"
            f"Root symbol: {root_row['qualname']}\n"
            "Potentially impacted symbols:\n"
            f"{impacted_block or '- None detected'}\n\n"
            "Dependency sketch:\n"
            f"{ascii_graph}\n\n"
            "Output:\n"
            "1) Main risks\n"
            "2) Most likely breakpoints\n"
            "3) Test recommendations"
        )
