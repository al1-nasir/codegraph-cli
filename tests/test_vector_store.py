"""Tests for VectorStore class."""

import tempfile
from pathlib import Path

import pytest

from codegraph_cli.vector_store import VectorStore, CHROMA_AVAILABLE


@pytest.mark.skipif(not CHROMA_AVAILABLE, reason="chromadb not installed")
class TestVectorStore:
    """Test VectorStore functionality."""
    
    def test_init(self, temp_dir: Path):
        """Test vector store initialization."""
        store = VectorStore(temp_dir)
        assert store.project_dir == temp_dir
        assert (temp_dir / ".chroma").exists()
        assert store.count() == 0
    
    def test_add_nodes(self, temp_dir: Path):
        """Test adding nodes to vector store."""
        store = VectorStore(temp_dir)
        
        node_ids = ["node1", "node2", "node3"]
        embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ]
        metadatas = [
            {"node_type": "function", "file_path": "test.py"},
            {"node_type": "class", "file_path": "test.py"},
            {"node_type": "function", "file_path": "utils.py"}
        ]
        documents = [
            "def foo(): pass",
            "class Bar: pass",
            "def baz(): pass"
        ]
        
        store.add_nodes(node_ids, embeddings, metadatas, documents)
        
        assert store.count() == 3
    
    def test_search(self, temp_dir: Path):
        """Test similarity search."""
        store = VectorStore(temp_dir)
        
        # Add some nodes
        node_ids = ["func1", "func2", "class1"]
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],  # Similar to func1
            [0.0, 1.0, 0.0]   # Different
        ]
        metadatas = [
            {"node_type": "function"},
            {"node_type": "function"},
            {"node_type": "class"}
        ]
        documents = ["func1 code", "func2 code", "class1 code"]
        
        store.add_nodes(node_ids, embeddings, metadatas, documents)
        
        # Search for similar to func1
        results = store.search(
            query_embedding=[1.0, 0.0, 0.0],
            n_results=2
        )
        
        assert len(results["ids"][0]) == 2
        # func1 should be most similar
        assert results["ids"][0][0] == "func1"
    
    def test_search_with_filter(self, temp_dir: Path):
        """Test search with metadata filtering."""
        store = VectorStore(temp_dir)
        
        node_ids = ["func1", "func2", "class1"]
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [1.0, 0.0, 0.0]  # Same as func1 but different type
        ]
        metadatas = [
            {"node_type": "function"},
            {"node_type": "function"},
            {"node_type": "class"}
        ]
        documents = ["func1", "func2", "class1"]
        
        store.add_nodes(node_ids, embeddings, metadatas, documents)
        
        # Search only for functions
        results = store.search(
            query_embedding=[1.0, 0.0, 0.0],
            n_results=10,
            where={"node_type": "function"}
        )
        
        # Should only return functions, not class
        assert len(results["ids"][0]) == 2
        assert all("func" in id for id in results["ids"][0])
    
    def test_get_node(self, temp_dir: Path):
        """Test getting a specific node."""
        store = VectorStore(temp_dir)
        
        store.add_nodes(
            ["test_node"],
            [[0.1, 0.2, 0.3]],
            [{"node_type": "function"}],
            ["def test(): pass"]
        )
        
        node = store.get_node("test_node")
        assert node is not None
        assert node["id"] == "test_node"
        assert node["metadata"]["node_type"] == "function"
        assert node["document"] == "def test(): pass"
        
        # Non-existent node
        assert store.get_node("nonexistent") is None
    
    def test_delete_nodes(self, temp_dir: Path):
        """Test deleting nodes."""
        store = VectorStore(temp_dir)
        
        store.add_nodes(
            ["node1", "node2"],
            [[0.1, 0.2], [0.3, 0.4]],
            [{"node_type": "function"}, {"node_type": "class"}],  # Non-empty metadata
            ["doc1", "doc2"]
        )
        
        assert store.count() == 2
        
        store.delete_nodes(["node1"])
        assert store.count() == 1
        assert store.get_node("node1") is None
        assert store.get_node("node2") is not None
    
    def test_clear(self, temp_dir: Path):
        """Test clearing all nodes."""
        store = VectorStore(temp_dir)
        
        store.add_nodes(
            ["node1", "node2"],
            [[0.1, 0.2], [0.3, 0.4]],
            [{"node_type": "function"}, {"node_type": "class"}],  # Non-empty metadata
            ["doc1", "doc2"]
        )
        
        assert store.count() == 2
        
        store.clear()
        assert store.count() == 0
    
    def test_persistence(self, temp_dir: Path):
        """Test that data persists across store instances."""
        # Create store and add data
        store1 = VectorStore(temp_dir)
        store1.add_nodes(
            ["persistent_node"],
            [[0.5, 0.5]],
            [{"test": "data"}],
            ["persistent doc"]
        )
        
        # Create new store instance with same directory
        store2 = VectorStore(temp_dir)
        assert store2.count() == 1
        node = store2.get_node("persistent_node")
        assert node is not None
        assert node["metadata"]["test"] == "data"
    
    def test_upsert(self, temp_dir: Path):
        """Test that adding same node ID updates it."""
        store = VectorStore(temp_dir)
        
        # Add initial node
        store.add_nodes(
            ["node1"],
            [[0.1, 0.2]],
            [{"version": "1"}],
            ["original doc"]
        )
        
        # Update same node
        store.add_nodes(
            ["node1"],
            [[0.3, 0.4]],
            [{"version": "2"}],
            ["updated doc"]
        )
        
        # Should still have only 1 node
        assert store.count() == 1
        
        # Should have updated data
        node = store.get_node("node1")
        assert node["metadata"]["version"] == "2"
        assert node["document"] == "updated doc"
