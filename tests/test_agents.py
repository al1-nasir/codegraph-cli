"""Tests for multi-agent components."""

from pathlib import Path

import pytest

from codegraph_cli.agents import GraphAgent, RAGAgent, SummarizationAgent
from codegraph_cli.embeddings import HashEmbeddingModel
from codegraph_cli.llm import LocalLLM
from codegraph_cli.rag import RAGRetriever
from codegraph_cli.storage import GraphStore


class TestGraphAgent:
    """Tests for GraphAgent."""
    
    def test_index_project(self, temp_graph_store: GraphStore, sample_project_path: Path):
        """Test indexing a project."""
        agent = GraphAgent(temp_graph_store, HashEmbeddingModel())
        stats = agent.index_project(sample_project_path)
        
        assert stats["nodes"] > 15
        assert stats["edges"] > 20
        
        # Verify nodes were stored
        nodes = temp_graph_store.get_nodes()
        assert len(nodes) == stats["nodes"]
    
    def test_ascii_neighbors(self, indexed_sample_store: GraphStore):
        """Test ASCII neighbor visualization."""
        agent = GraphAgent(indexed_sample_store, HashEmbeddingModel())
        
        result = agent.ascii_neighbors("processor.UserProcessor", depth=1)
        
        assert "UserProcessor" in result
        assert "contains" in result or "calls" in result
    
    def test_ascii_neighbors_not_found(self, temp_graph_store: GraphStore):
        """Test ASCII neighbors for non-existent symbol."""
        agent = GraphAgent(temp_graph_store, HashEmbeddingModel())
        
        result = agent.ascii_neighbors("NonExistent", depth=1)
        
        assert "not found" in result.lower()


class TestRAGAgent:
    """Tests for RAGAgent."""
    
    def test_semantic_search(self, indexed_sample_store: GraphStore):
        """Test semantic search."""
        retriever = RAGRetriever(indexed_sample_store, HashEmbeddingModel())
        agent = RAGAgent(retriever)
        
        results = agent.semantic_search("validate email", top_k=5)
        
        assert len(results) > 0
        assert len(results) <= 5
        
        # Should find validate_email function
        qualnames = [r.qualname for r in results]
        assert any("validate_email" in qn for qn in qualnames)
    
    def test_semantic_search_with_limit(self, indexed_sample_store: GraphStore):
        """Test semantic search respects top_k limit."""
        retriever = RAGRetriever(indexed_sample_store, HashEmbeddingModel())
        agent = RAGAgent(retriever)
        
        results = agent.semantic_search("function", top_k=3)
        
        assert len(results) <= 3
    
    def test_context_for_query(self, indexed_sample_store: GraphStore):
        """Test retrieving context for a query."""
        retriever = RAGRetriever(indexed_sample_store, HashEmbeddingModel())
        agent = RAGAgent(retriever)
        
        context = agent.context_for_query("user creation", top_k=3)
        
        assert isinstance(context, str)
        assert len(context) > 0
        # Should contain code snippets
        assert "```" in context or "def " in context


class TestSummarizationAgent:
    """Tests for SummarizationAgent."""
    
    def test_impact_analysis(self, indexed_sample_store: GraphStore):
        """Test impact analysis."""
        llm = LocalLLM()
        agent = SummarizationAgent(indexed_sample_store, llm)
        
        report = agent.impact_analysis("processor.OrderProcessor.create_order", hops=2)
        
        assert report.root == "processor.OrderProcessor.create_order"
        assert len(report.impacted) > 0
        assert len(report.explanation) > 0
        assert len(report.ascii_graph) > 0
    
    def test_impact_analysis_not_found(self, temp_graph_store: GraphStore):
        """Test impact analysis for non-existent symbol."""
        llm = LocalLLM()
        agent = SummarizationAgent(temp_graph_store, llm)
        
        report = agent.impact_analysis("NonExistent", hops=2)
        
        assert "not found" in report.explanation.lower()
        assert len(report.impacted) == 0
    
    def test_impact_analysis_multi_hop(self, indexed_sample_store: GraphStore):
        """Test multi-hop impact analysis."""
        llm = LocalLLM()
        agent = SummarizationAgent(indexed_sample_store, llm)
        
        # Test with different hop counts
        report1 = agent.impact_analysis("utils.validate_email", hops=1)
        report2 = agent.impact_analysis("utils.validate_email", hops=3)
        
        # More hops should potentially find more impacted symbols
        assert len(report2.impacted) >= len(report1.impacted)


class TestAgentIntegration:
    """Integration tests for agents working together."""
    
    def test_full_workflow(self, temp_graph_store: GraphStore, sample_project_path: Path):
        """Test complete workflow: index, search, impact."""
        embedding_model = HashEmbeddingModel()
        
        # 1. Index project
        graph_agent = GraphAgent(temp_graph_store, embedding_model)
        stats = graph_agent.index_project(sample_project_path)
        assert stats["nodes"] > 0
        
        # 2. Search for something
        retriever = RAGRetriever(temp_graph_store, embedding_model)
        rag_agent = RAGAgent(retriever)
        results = rag_agent.semantic_search("order processing", top_k=5)
        assert len(results) > 0
        
        # 3. Analyze impact
        llm = LocalLLM()
        summ_agent = SummarizationAgent(temp_graph_store, llm)
        report = summ_agent.impact_analysis("processor.OrderProcessor", hops=2)
        assert len(report.impacted) > 0
