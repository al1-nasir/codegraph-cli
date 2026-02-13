"""Tests for bug detection module."""

import tempfile
from pathlib import Path

import pytest

from codegraph_cli.bug_detector import BugDetector
from codegraph_cli.models import Node
from codegraph_cli.storage import GraphStore


class TestBugDetector:
    """Test BugDetector functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def store(self, temp_dir):
        """Create GraphStore for testing."""
        return GraphStore(temp_dir)

    def test_detect_null_pointer_risk(self, store, temp_dir):
        """Test detection of null pointer risks."""
        code = '''
def get_user(user_id):
    user = database.get(user_id)
    return user.name
'''
        node = Node(
            node_id="function:test.get_user",
            node_type="function",
            name="get_user",
            qualname="test.get_user",
            file_path="test.py",
            start_line=1,
            end_line=4,
            code=code,
            docstring=""
        )
        
        store.insert_nodes([(node, [0.1, 0.2, 0.3])])
        
        detector = BugDetector(store)
        issues = detector.analyze_file("test.py")
        
        # Should detect potential None access
        null_risks = [i for i in issues if i["type"] == "null_pointer_risk"]
        assert len(null_risks) > 0
        assert "None" in null_risks[0]["message"]

    def test_detect_unreachable_code(self, store, temp_dir):
        """Test detection of unreachable code."""
        code = '''
def process():
    return 42
    print("unreachable")
'''
        node = Node(
            node_id="function:test.process",
            node_type="function",
            name="process",
            qualname="test.process",
            file_path="test.py",
            start_line=1,
            end_line=4,
            code=code,
            docstring=""
        )
        
        store.insert_nodes([(node, [0.1, 0.2, 0.3])])
        
        detector = BugDetector(store)
        issues = detector.analyze_file("test.py")
        
        # Should detect unreachable code
        unreachable = [i for i in issues if i["type"] == "unreachable_code"]
        assert len(unreachable) > 0
        assert "unreachable" in unreachable[0]["message"].lower()

    def test_detect_infinite_loop(self, store, temp_dir):
        """Test detection of infinite loops."""
        code = '''
def loop_forever():
    while True:
        pass
'''
        node = Node(
            node_id="function:test.loop_forever",
            node_type="function",
            name="loop_forever",
            qualname="test.loop_forever",
            file_path="test.py",
            start_line=1,
            end_line=4,
            code=code,
            docstring=""
        )
        
        store.insert_nodes([(node, [0.1, 0.2, 0.3])])
        
        detector = BugDetector(store)
        issues = detector.analyze_file("test.py")
        
        # Should detect infinite loop
        infinite_loops = [i for i in issues if i["type"] == "infinite_loop"]
        assert len(infinite_loops) > 0
        assert "infinite" in infinite_loops[0]["message"].lower()

    def test_detect_resource_leak(self, store, temp_dir):
        """Test detection of resource leaks."""
        code = '''
def read_file():
    f = open("data.txt")
    return f.read()
'''
        node = Node(
            node_id="function:test.read_file",
            node_type="function",
            name="read_file",
            qualname="test.read_file",
            file_path="test.py",
            start_line=1,
            end_line=4,
            code=code,
            docstring=""
        )
        
        store.insert_nodes([(node, [0.1, 0.2, 0.3])])
        
        detector = BugDetector(store)
        issues = detector.analyze_file("test.py")
        
        # Should detect file not closed
        resource_leaks = [i for i in issues if i["type"] == "resource_leak"]
        assert len(resource_leaks) > 0
        assert "with" in resource_leaks[0]["suggestion"].lower()

    def test_no_issues_for_good_code(self, store, temp_dir):
        """Test that good code doesn't trigger false positives."""
        code = '''
def safe_get_user(user_id):
    user = database.get(user_id)
    if user is not None:
        return user.name
    return None
'''
        node = Node(
            node_id="function:test.safe_get_user",
            node_type="function",
            name="safe_get_user",
            qualname="test.safe_get_user",
            file_path="test.py",
            start_line=1,
            end_line=6,
            code=code,
            docstring=""
        )
        
        store.insert_nodes([(node, [0.1, 0.2, 0.3])])
        
        detector = BugDetector(store)
        issues = detector.analyze_file("test.py")
        
        # Should not detect null pointer risk (has None check)
        null_risks = [i for i in issues if i["type"] == "null_pointer_risk"]
        assert len(null_risks) == 0
