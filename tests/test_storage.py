"""Tests for storage layer (ProjectManager and GraphStore)."""

import json
from pathlib import Path

import pytest

from codegraph_cli.models import Edge, Node
from codegraph_cli.storage import GraphStore, ProjectManager


class TestProjectManager:
    """Tests for ProjectManager."""
    
    def test_create_project(self, temp_project_manager: ProjectManager):
        """Test creating a new project."""
        pm = temp_project_manager
        project_dir = pm.create_or_get_project("TestProject")
        
        assert project_dir.exists()
        assert project_dir.is_dir()
        assert "TestProject" in pm.list_projects()
    
    def test_list_projects(self, temp_project_manager: ProjectManager):
        """Test listing projects."""
        pm = temp_project_manager
        
        # Initially empty
        assert pm.list_projects() == []
        
        # Create projects
        pm.create_or_get_project("Project1")
        pm.create_or_get_project("Project2")
        
        projects = pm.list_projects()
        assert len(projects) == 2
        assert "Project1" in projects
        assert "Project2" in projects
    
    def test_set_and_get_current_project(self, temp_project_manager: ProjectManager):
        """Test setting and getting current project."""
        pm = temp_project_manager
        pm.create_or_get_project("MyProject")
        
        # Set current
        pm.set_current_project("MyProject")
        
        # Get current
        assert pm.get_current_project() == "MyProject"
    
    def test_unload_project(self, temp_project_manager: ProjectManager):
        """Test unloading current project."""
        pm = temp_project_manager
        pm.create_or_get_project("MyProject")
        pm.set_current_project("MyProject")
        
        assert pm.get_current_project() == "MyProject"
        
        pm.unload_project()
        assert pm.get_current_project() is None
    
    def test_delete_project(self, temp_project_manager: ProjectManager):
        """Test deleting a project."""
        pm = temp_project_manager
        pm.create_or_get_project("ToDelete")
        
        assert "ToDelete" in pm.list_projects()
        
        result = pm.delete_project("ToDelete")
        assert result is True
        assert "ToDelete" not in pm.list_projects()
    
    def test_delete_nonexistent_project(self, temp_project_manager: ProjectManager):
        """Test deleting a project that doesn't exist."""
        pm = temp_project_manager
        result = pm.delete_project("DoesNotExist")
        assert result is False


class TestGraphStore:
    """Tests for GraphStore."""
    
    def test_store_initialization(self, temp_graph_store: GraphStore):
        """Test GraphStore initialization."""
        store = temp_graph_store
        assert store.db_path.exists()
        assert store.meta_path.parent.exists()
    
    def test_insert_and_get_nodes(self, temp_graph_store: GraphStore):
        """Test inserting and retrieving nodes."""
        store = temp_graph_store
        
        node = Node(
            node_id="test:func1",
            node_type="function",
            name="func1",
            qualname="test.func1",
            file_path="test.py",
            start_line=1,
            end_line=5,
            code="def func1(): pass",
            docstring="Test function"
        )
        
        embedding = [0.1] * 256
        store.insert_nodes([(node, embedding)])
        
        # Retrieve by ID
        retrieved = store.get_node("test:func1")
        assert retrieved is not None
        assert retrieved["qualname"] == "test.func1"
        assert retrieved["node_type"] == "function"
    
    def test_insert_and_get_edges(self, temp_graph_store: GraphStore):
        """Test inserting and retrieving edges."""
        store = temp_graph_store
        
        # Insert nodes first
        node1 = Node(
            node_id="test:func1",
            node_type="function",
            name="func1",
            qualname="test.func1",
            file_path="test.py",
            start_line=1,
            end_line=5,
            code="def func1(): pass",
            docstring=""
        )
        node2 = Node(
            node_id="test:func2",
            node_type="function",
            name="func2",
            qualname="test.func2",
            file_path="test.py",
            start_line=7,
            end_line=10,
            code="def func2(): func1()",
            docstring=""
        )
        
        embedding = [0.1] * 256
        store.insert_nodes([(node1, embedding), (node2, embedding)])
        
        # Insert edge
        edge = Edge(src="test:func2", dst="test:func1", edge_type="calls")
        store.insert_edges([edge])
        
        # Retrieve edges
        edges = store.get_edges()
        assert len(edges) == 1
        assert edges[0]["src"] == "test:func2"
        assert edges[0]["dst"] == "test:func1"
        assert edges[0]["edge_type"] == "calls"
    
    def test_neighbors(self, temp_graph_store: GraphStore):
        """Test getting neighbors of a node."""
        store = temp_graph_store
        
        # Create a simple graph: A -> B -> C
        nodes = [
            Node("A", "function", "A", "test.A", "test.py", 1, 2, "def A(): pass", ""),
            Node("B", "function", "B", "test.B", "test.py", 4, 5, "def B(): A()", ""),
            Node("C", "function", "C", "test.C", "test.py", 7, 8, "def C(): B()", ""),
        ]
        embedding = [0.1] * 256
        store.insert_nodes([(n, embedding) for n in nodes])
        
        edges = [
            Edge("B", "A", "calls"),
            Edge("C", "B", "calls"),
        ]
        store.insert_edges(edges)
        
        # Get neighbors of B
        neighbors = store.neighbors("B")
        assert len(neighbors) == 1
        assert neighbors[0]["dst"] == "A"
    
    def test_metadata(self, temp_graph_store: GraphStore):
        """Test setting and getting metadata."""
        store = temp_graph_store
        
        metadata = {
            "project_name": "TestProject",
            "node_count": 42,
            "indexed_at": "2024-01-01"
        }
        
        store.set_metadata(metadata)
        retrieved = store.get_metadata()
        
        assert retrieved["project_name"] == "TestProject"
        assert retrieved["node_count"] == 42
    
    def test_clear(self, temp_graph_store: GraphStore):
        """Test clearing all data from store."""
        store = temp_graph_store
        
        # Insert some data
        node = Node("test", "function", "test", "test", "test.py", 1, 2, "code", "")
        embedding = [0.1] * 256
        store.insert_nodes([(node, embedding)])
        
        assert len(store.get_nodes()) == 1
        
        # Clear
        store.clear()
        
        assert len(store.get_nodes()) == 0
    
    def test_get_node_by_qualname(self, temp_graph_store: GraphStore):
        """Test retrieving node by qualname."""
        store = temp_graph_store
        
        node = Node(
            "test:MyClass",
            "class",
            "MyClass",
            "module.MyClass",
            "module.py",
            1,
            10,
            "class MyClass: pass",
            "A class"
        )
        embedding = [0.1] * 256
        store.insert_nodes([(node, embedding)])
        
        # Retrieve by qualname
        retrieved = store.get_node("module.MyClass")
        assert retrieved is not None
        assert retrieved["node_id"] == "test:MyClass"
    
    def test_merge_projects(self, temp_dir: Path):
        """Test merging two project stores."""
        # Create two stores
        project1_dir = temp_dir / "project1"
        project2_dir = temp_dir / "project2"
        project1_dir.mkdir()
        project2_dir.mkdir()
        
        store1 = GraphStore(project1_dir)
        store2 = GraphStore(project2_dir)
        
        # Add data to store1
        node1 = Node("p1:func", "function", "func", "p1.func", "p1.py", 1, 2, "code", "")
        embedding = [0.1] * 256
        store1.insert_nodes([(node1, embedding)])
        
        # Add data to store2
        node2 = Node("p2:func", "function", "func", "p2.func", "p2.py", 1, 2, "code", "")
        store2.insert_nodes([(node2, embedding)])
        
        # Merge store1 into store2
        store2.merge_from(store1, "Project1")
        
        # store2 should now have both nodes
        nodes = store2.get_nodes()
        assert len(nodes) == 2
        
        store1.close()
        store2.close()


class TestIndexedSampleStore:
    """Tests using the indexed sample project."""
    
    def test_indexed_sample_has_nodes(self, indexed_sample_store: GraphStore):
        """Test that indexed sample has expected nodes."""
        nodes = indexed_sample_store.get_nodes()
        assert len(nodes) > 15
        
        qualnames = {n["qualname"] for n in nodes}
        assert "utils.validate_email" in qualnames
        assert "models.User" in qualnames
    
    def test_indexed_sample_has_edges(self, indexed_sample_store: GraphStore):
        """Test that indexed sample has edges."""
        edges = indexed_sample_store.get_edges()
        assert len(edges) > 20
        
        edge_types = {e["edge_type"] for e in edges}
        assert "contains" in edge_types
        assert "calls" in edge_types
