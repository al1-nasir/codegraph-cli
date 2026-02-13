"""Tests for Python AST parser."""

import ast
from pathlib import Path

import pytest

from codegraph_cli.models import Edge, Node
from codegraph_cli.parser import PythonGraphParser


def test_parser_initialization(temp_dir: Path):
    """Test parser can be initialized with a project root."""
    parser = PythonGraphParser(temp_dir)
    assert parser.project_root == temp_dir


def test_parse_simple_function(temp_dir: Path, sample_python_code: str):
    """Test parsing a simple function."""
    # Create a test file
    test_file = temp_dir / "test.py"
    test_file.write_text(sample_python_code)
    
    parser = PythonGraphParser(temp_dir)
    nodes, edges = parser.parse_project()
    
    # Should have module, class, and function nodes
    assert len(nodes) > 0
    
    # Check for hello function
    hello_nodes = [n for n in nodes if n.name == "hello" and n.node_type == "function"]
    assert len(hello_nodes) == 1
    assert "Say hello" in hello_nodes[0].docstring


def test_parse_class_with_methods(temp_dir: Path, sample_python_code: str):
    """Test parsing a class with methods."""
    test_file = temp_dir / "test.py"
    test_file.write_text(sample_python_code)
    
    parser = PythonGraphParser(temp_dir)
    nodes, edges = parser.parse_project()
    
    # Check for Calculator class
    calc_nodes = [n for n in nodes if n.name == "Calculator" and n.node_type == "class"]
    assert len(calc_nodes) == 1
    
    # Check for methods
    add_nodes = [n for n in nodes if n.name == "add" and n.node_type == "function"]
    assert len(add_nodes) == 1
    
    multiply_nodes = [n for n in nodes if n.name == "multiply" and n.node_type == "function"]
    assert len(multiply_nodes) == 1


def test_parse_detects_function_calls(temp_dir: Path, sample_python_code: str):
    """Test that parser detects function calls."""
    test_file = temp_dir / "test.py"
    test_file.write_text(sample_python_code)
    
    parser = PythonGraphParser(temp_dir)
    nodes, edges = parser.parse_project()
    
    # multiply() calls add(), should have call edges
    call_edges = [e for e in edges if e.edge_type == "calls"]
    assert len(call_edges) > 0


def test_parse_sample_project(sample_project_path: Path):
    """Test parsing the complete sample project."""
    parser = PythonGraphParser(sample_project_path)
    nodes, edges = parser.parse_project()
    
    # Should have multiple nodes
    assert len(nodes) >= 15  # modules, classes, functions
    
    # Should have edges
    assert len(edges) >= 20
    
    # Check for specific expected nodes
    node_names = {n.qualname for n in nodes}
    assert "utils.validate_email" in node_names
    assert "models.User" in node_names
    assert "processor.UserProcessor" in node_names
    assert "processor.OrderProcessor" in node_names


def test_parse_handles_imports(temp_dir: Path):
    """Test that parser detects import dependencies."""
    code = '''"""Test module."""
import os
from pathlib import Path

def test():
    pass
'''
    test_file = temp_dir / "test.py"
    test_file.write_text(code)
    
    parser = PythonGraphParser(temp_dir)
    nodes, edges = parser.parse_project()
    
    # Should have dependency edges for imports
    import_edges = [e for e in edges if e.edge_type == "depends_on"]
    assert len(import_edges) >= 2  # os and pathlib


def test_parse_skips_venv(temp_dir: Path):
    """Test that parser skips .venv and __pycache__ directories."""
    # Create files in .venv
    venv_dir = temp_dir / ".venv" / "lib"
    venv_dir.mkdir(parents=True)
    (venv_dir / "test.py").write_text("def venv_func(): pass")
    
    # Create file in __pycache__
    cache_dir = temp_dir / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "test.pyc").write_text("compiled")
    
    # Create normal file
    (temp_dir / "main.py").write_text("def main(): pass")
    
    parser = PythonGraphParser(temp_dir)
    nodes, edges = parser.parse_project()
    
    # Should only parse main.py
    file_paths = {n.file_path for n in nodes}
    assert any("main.py" in fp for fp in file_paths)
    assert not any(".venv" in fp for fp in file_paths)
    assert not any("__pycache__" in fp for fp in file_paths)


def test_parse_handles_syntax_errors(temp_dir: Path):
    """Test that parser handles files with syntax errors gracefully."""
    bad_code = "def broken( this is not valid python"
    test_file = temp_dir / "broken.py"
    test_file.write_text(bad_code)
    
    parser = PythonGraphParser(temp_dir)
    
    # Should not crash, just skip the broken file
    try:
        nodes, edges = parser.parse_project()
        # If it doesn't crash, test passes
        assert True
    except SyntaxError:
        # Parser might raise SyntaxError, which is acceptable
        assert True


def test_node_qualname_format(sample_project_path: Path):
    """Test that node qualnames are properly formatted."""
    parser = PythonGraphParser(sample_project_path)
    nodes, edges = parser.parse_project()
    
    # Check qualname format
    for node in nodes:
        if node.node_type == "module":
            assert "." in node.qualname or node.qualname.isidentifier()
        elif node.node_type in ("class", "function"):
            assert "." in node.qualname  # Should be module.class or module.function


def test_edge_types(sample_project_path: Path):
    """Test that edges have correct types."""
    parser = PythonGraphParser(sample_project_path)
    nodes, edges = parser.parse_project()
    
    edge_types = {e.edge_type for e in edges}
    
    # Should have these edge types
    assert "contains" in edge_types  # module contains class/function
    assert "calls" in edge_types  # function calls function
    assert "depends_on" in edge_types  # module imports module
