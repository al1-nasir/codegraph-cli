"""Tests for security scanner module."""

import tempfile
from pathlib import Path

import pytest

from codegraph_cli.models import Node
from codegraph_cli.security_scanner import SecurityScanner
from codegraph_cli.storage import GraphStore


class TestSecurityScanner:
    """Test SecurityScanner functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def store(self, temp_dir):
        """Create GraphStore for testing."""
        return GraphStore(temp_dir)

    def test_detect_sql_injection(self, store, temp_dir):
        """Test detection of SQL injection vulnerabilities."""
        code = '''def get_user(username):
    query = "SELECT * FROM users WHERE name = '" + username + "'"
    cursor.execute(query)
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
        
        scanner = SecurityScanner(store)
        issues = scanner.scan_file("test.py")
        
        # Should detect SQL injection
        sql_issues = [i for i in issues if i["type"] == "sql_injection"]
        assert len(sql_issues) > 0
        assert "injection" in sql_issues[0]["message"].lower()

    def test_detect_command_injection(self, store, temp_dir):
        """Test detection of command injection."""
        code = '''
def run_command(cmd):
    os.system(cmd)
'''
        node = Node(
            node_id="function:test.run_command",
            node_type="function",
            name="run_command",
            qualname="test.run_command",
            file_path="test.py",
            start_line=1,
            end_line=3,
            code=code,
            docstring=""
        )
        
        store.insert_nodes([(node, [0.1, 0.2, 0.3])])
        
        scanner = SecurityScanner(store)
        issues = scanner.scan_file("test.py")
        
        # Should detect command injection
        cmd_issues = [i for i in issues if "injection" in i["type"]]
        assert len(cmd_issues) > 0

    def test_detect_hardcoded_secrets(self, store, temp_dir):
        """Test detection of hardcoded secrets."""
        code = '''
API_KEY = "sk-1234567890abcdefghijklmnop"
PASSWORD = "super_secret_password_123"
'''
        node = Node(
            node_id="module:test",
            node_type="module",
            name="test",
            qualname="test",
            file_path="test.py",
            start_line=1,
            end_line=3,
            code=code,
            docstring=""
        )
        
        store.insert_nodes([(node, [0.1, 0.2, 0.3])])
        
        scanner = SecurityScanner(store)
        issues = scanner.scan_file("test.py")
        
        # Should detect hardcoded secrets
        secret_issues = [i for i in issues if i["type"] == "hardcoded_secret"]
        assert len(secret_issues) > 0

    def test_detect_unsafe_deserialization(self, store, temp_dir):
        """Test detection of unsafe deserialization."""
        code = '''
import pickle

def load_data(data):
    return pickle.loads(data)
'''
        node = Node(
            node_id="function:test.load_data",
            node_type="function",
            name="load_data",
            qualname="test.load_data",
            file_path="test.py",
            start_line=1,
            end_line=5,
            code=code,
            docstring=""
        )
        
        store.insert_nodes([(node, [0.1, 0.2, 0.3])])
        
        scanner = SecurityScanner(store)
        issues = scanner.scan_file("test.py")
        
        # Should detect unsafe deserialization
        deser_issues = [i for i in issues if i["type"] == "unsafe_deserialization"]
        assert len(deser_issues) > 0

    def test_no_false_positives_for_safe_code(self, store, temp_dir):
        """Test that safe code doesn't trigger false positives."""
        code = '''
def safe_query(user_id):
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
'''
        node = Node(
            node_id="function:test.safe_query",
            node_type="function",
            name="safe_query",
            qualname="test.safe_query",
            file_path="test.py",
            start_line=1,
            end_line=3,
            code=code,
            docstring=""
        )
        
        store.insert_nodes([(node, [0.1, 0.2, 0.3])])
        
        scanner = SecurityScanner(store)
        issues = scanner.scan_file("test.py")
        
        # Should not detect SQL injection (uses parameterized query)
        sql_issues = [i for i in issues if i["type"] == "sql_injection"]
        assert len(sql_issues) == 0
