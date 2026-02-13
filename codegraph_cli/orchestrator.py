"""MCP-style orchestrator coordinating specialized agents."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .agents import GraphAgent, RAGAgent, SummarizationAgent
from .embeddings import HashEmbeddingModel
from .llm import LocalLLM
from .models import ImpactReport, SearchResult
from .rag import RAGRetriever
from .storage import GraphStore


class MCPOrchestrator:
    """Coordinates graph, retrieval, and summarization agents."""

    def __init__(
        self,
        store: GraphStore,
        llm_model: str = "qwen2.5-coder:7b",
        llm_provider: str = "ollama",
        llm_api_key: str | None = None,
        llm_endpoint: str | None = None,
    ):
        self.store = store
        self.embedding_model = HashEmbeddingModel()
        self.graph_agent = GraphAgent(store, self.embedding_model)
        self.rag_agent = RAGAgent(RAGRetriever(store, self.embedding_model))
        self.summarization_agent = SummarizationAgent(
            store,
            LocalLLM(model=llm_model, provider=llm_provider, api_key=llm_api_key, endpoint=llm_endpoint),
        )

    def index(self, project_root: Path) -> Dict[str, int]:
        return self.graph_agent.index_project(project_root)

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        return self.rag_agent.semantic_search(query, top_k=top_k)

    def impact(self, symbol: str, hops: int = 2) -> ImpactReport:
        return self.summarization_agent.impact_analysis(symbol=symbol, hops=hops)

    def graph(self, symbol: str, depth: int = 2) -> str:
        return self.graph_agent.ascii_neighbors(symbol=symbol, depth=depth)

    def rag_context(self, query: str, top_k: int = 6) -> str:
        return self.rag_agent.context_for_query(query, top_k=top_k)
