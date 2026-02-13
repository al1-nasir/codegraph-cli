"""Integration tests for CLI commands."""

from pathlib import Path
from typer.testing import CliRunner

import pytest

from codegraph_cli.cli import app


runner = CliRunner()


class TestIndexCommand:
    """Tests for 'cg index' command."""
    
    def test_index_project(self, sample_project_path: Path, temp_project_manager):
        """Test indexing a project."""
        result = runner.invoke(app, ["index", str(sample_project_path), "--name", "TestProj"])
        
        assert result.exit_code == 0
        assert "Indexed" in result.stdout
        assert "TestProj" in result.stdout
        assert "Nodes:" in result.stdout
        assert "Edges:" in result.stdout
    
    def test_index_nonexistent_path(self):
        """Test indexing a non-existent path."""
        result = runner.invoke(app, ["index", "/nonexistent/path"])
        
        assert result.exit_code != 0


class TestListProjectsCommand:
    """Tests for 'cg list-projects' command."""
    
    def test_list_empty(self, temp_project_manager):
        """Test listing when no projects exist."""
        result = runner.invoke(app, ["list-projects"])
        
        assert result.exit_code == 0
        assert "No projects" in result.stdout
    
    def test_list_with_projects(self, sample_project_path: Path, temp_project_manager):
        """Test listing projects."""
        # Index a project first
        runner.invoke(app, ["index", str(sample_project_path), "--name", "Proj1"])
        
        result = runner.invoke(app, ["list-projects"])
        
        assert result.exit_code == 0
        assert "Proj1" in result.stdout


class TestLoadProjectCommand:
    """Tests for 'cg load-project' command."""
    
    def test_load_existing_project(self, sample_project_path: Path, temp_project_manager):
        """Test loading an existing project."""
        # Index first
        runner.invoke(app, ["index", str(sample_project_path), "--name", "MyProj"])
        
        # Load
        result = runner.invoke(app, ["load-project", "MyProj"])
        
        assert result.exit_code == 0
        assert "Loaded" in result.stdout
        assert "MyProj" in result.stdout
    
    def test_load_nonexistent_project(self, temp_project_manager):
        """Test loading a non-existent project."""
        result = runner.invoke(app, ["load-project", "DoesNotExist"])
        
        assert result.exit_code != 0


class TestSearchCommand:
    """Tests for 'cg search' command."""
    
    def test_search_with_results(self, sample_project_path: Path, temp_project_manager):
        """Test search that returns results."""
        # Index and load
        runner.invoke(app, ["index", str(sample_project_path), "--name", "SearchTest"])
        
        result = runner.invoke(app, ["search", "validate email"])
        
        assert result.exit_code == 0
        assert "validate_email" in result.stdout or "score=" in result.stdout
    
    def test_search_no_project_loaded(self, temp_project_manager):
        """Test search without a loaded project."""
        result = runner.invoke(app, ["search", "test"])
        
        assert result.exit_code != 0
        assert "No project loaded" in result.stdout


class TestImpactCommand:
    """Tests for 'cg impact' command."""
    
    def test_impact_analysis(self, sample_project_path: Path, temp_project_manager):
        """Test impact analysis."""
        # Index and load
        runner.invoke(app, ["index", str(sample_project_path), "--name", "ImpactTest"])
        
        result = runner.invoke(app, ["impact", "create_order"])
        
        assert result.exit_code == 0
        assert "Root:" in result.stdout
        assert "Impacted" in result.stdout or "Explanation" in result.stdout
    
    def test_impact_with_hops(self, sample_project_path: Path, temp_project_manager):
        """Test impact analysis with custom hops."""
        runner.invoke(app, ["index", str(sample_project_path), "--name", "ImpactTest2"])
        
        result = runner.invoke(app, ["impact", "create_order", "--hops", "3"])
        
        assert result.exit_code == 0


class TestGraphCommand:
    """Tests for 'cg graph' command."""
    
    def test_graph_visualization(self, sample_project_path: Path, temp_project_manager):
        """Test graph visualization."""
        runner.invoke(app, ["index", str(sample_project_path), "--name", "GraphTest"])
        
        result = runner.invoke(app, ["graph", "UserProcessor"])
        
        assert result.exit_code == 0
        assert "UserProcessor" in result.stdout
    
    def test_graph_with_depth(self, sample_project_path: Path, temp_project_manager):
        """Test graph with custom depth."""
        runner.invoke(app, ["index", str(sample_project_path), "--name", "GraphTest2"])
        
        result = runner.invoke(app, ["graph", "UserProcessor", "--depth", "3"])
        
        assert result.exit_code == 0


class TestExportGraphCommand:
    """Tests for 'cg export-graph' command."""
    
    def test_export_html(self, sample_project_path: Path, temp_project_manager, temp_dir: Path):
        """Test exporting graph as HTML."""
        runner.invoke(app, ["index", str(sample_project_path), "--name", "ExportTest"])
        
        output_file = temp_dir / "graph.html"
        result = runner.invoke(app, ["export-graph", "--format", "html", "--output", str(output_file)])
        
        assert result.exit_code == 0
        assert output_file.exists()
    
    def test_export_dot(self, sample_project_path: Path, temp_project_manager, temp_dir: Path):
        """Test exporting graph as DOT."""
        runner.invoke(app, ["index", str(sample_project_path), "--name", "ExportTest2"])
        
        output_file = temp_dir / "graph.dot"
        result = runner.invoke(app, ["export-graph", "--format", "dot", "--output", str(output_file)])
        
        assert result.exit_code == 0
        assert output_file.exists()


class TestDeleteProjectCommand:
    """Tests for 'cg delete-project' command."""
    
    def test_delete_existing_project(self, sample_project_path: Path, temp_project_manager):
        """Test deleting an existing project."""
        # Index first
        runner.invoke(app, ["index", str(sample_project_path), "--name", "ToDelete"])
        
        # Delete
        result = runner.invoke(app, ["delete-project", "ToDelete"])
        
        assert result.exit_code == 0
        assert "Deleted" in result.stdout
    
    def test_delete_nonexistent_project(self, temp_project_manager):
        """Test deleting a non-existent project."""
        result = runner.invoke(app, ["delete-project", "DoesNotExist"])
        
        assert result.exit_code != 0


class TestMergeProjectsCommand:
    """Tests for 'cg merge-projects' command."""
    
    def test_merge_projects(self, sample_project_path: Path, temp_project_manager):
        """Test merging two projects."""
        # Index two projects
        runner.invoke(app, ["index", str(sample_project_path), "--name", "Source"])
        runner.invoke(app, ["index", str(sample_project_path), "--name", "Target"])
        
        # Merge
        result = runner.invoke(app, ["merge-projects", "Source", "Target"])
        
        assert result.exit_code == 0
        assert "Merged" in result.stdout


class TestCurrentProjectCommand:
    """Tests for 'cg current-project' command."""
    
    def test_current_project_loaded(self, sample_project_path: Path, temp_project_manager):
        """Test showing current project when one is loaded."""
        runner.invoke(app, ["index", str(sample_project_path), "--name", "Current"])
        
        result = runner.invoke(app, ["current-project"])
        
        assert result.exit_code == 0
        assert "Current" in result.stdout
    
    def test_current_project_none(self, temp_project_manager):
        """Test showing current project when none is loaded."""
        # Unload any project
        runner.invoke(app, ["unload-project"])
        
        result = runner.invoke(app, ["current-project"])
        
        assert result.exit_code == 0
        assert "No project" in result.stdout


class TestRAGContextCommand:
    """Tests for 'cg rag-context' command."""
    
    def test_rag_context(self, sample_project_path: Path, temp_project_manager):
        """Test RAG context retrieval."""
        runner.invoke(app, ["index", str(sample_project_path), "--name", "RAGTest"])
        
        result = runner.invoke(app, ["rag-context", "user management"])
        
        assert result.exit_code == 0
        # Should contain code snippets or context
        assert len(result.stdout) > 0
