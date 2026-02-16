"""Security vulnerability scanner."""

from __future__ import annotations

import ast
import re
from typing import Dict, List, Optional

from .storage import GraphStore


class SecurityScanner:
    """Scan code for security vulnerabilities."""

    def __init__(self, store: GraphStore):
        self.store = store

        # Patterns for detecting hardcoded secrets
        self.secret_patterns = [
            (r'api[_-]?key\s*=\s*["\']([^"\']{10,})["\']', "API Key"),
            (r'password\s*=\s*["\']([^"\']+)["\']', "Password"),
            (r'secret\s*=\s*["\']([^"\']{10,})["\']', "Secret"),
            (r'token\s*=\s*["\']([^"\']{10,})["\']', "Token"),
            (r'aws_access_key_id\s*=\s*["\']([^"\']+)["\']', "AWS Access Key"),
            (r'private_key\s*=\s*["\']([^"\']+)["\']', "Private Key"),
        ]

        # Dangerous functions that could lead to injection
        self.dangerous_functions = {
            "eval": "code_injection",
            "exec": "code_injection",
            "compile": "code_injection",
            "__import__": "code_injection",
            "system": "command_injection",
            "popen": "command_injection",
            "spawn": "command_injection",
        }

    def scan_file(self, file_path: str, generate_fixes: bool = False) -> List[Dict]:
        """Scan file for security issues.
        
        Args:
            file_path: Path to file to scan
            generate_fixes: Whether to generate auto-fix suggestions
            
        Returns:
            List of security issue dictionaries
        """
        issues = []

        nodes = [n for n in self.store.get_nodes() if n["file_path"] == file_path]
        # Skip module-level node when function/class nodes exist to avoid dupes
        has_children = any(n["node_type"] != "module" for n in nodes)
        if has_children:
            nodes = [n for n in nodes if n["node_type"] != "module"]

        seen: set = set()  # (line, type) dedup

        for node in nodes:
            try:
                tree = ast.parse(node["code"])
            except SyntaxError:
                continue

            for issue in (
                self._detect_sql_injection(tree, node)
                + self._detect_command_injection(tree, node)
                + self._detect_hardcoded_secrets(node)
                + self._detect_path_traversal(tree, node)
                + self._detect_unsafe_deserialization(tree, node)
            ):
                key = (issue["line"], issue["type"])
                if key not in seen:
                    seen.add(key)
                    issues.append(issue)

        # Add auto-fixes if requested
        if generate_fixes:
            issues = [self._add_auto_fix(issue) for issue in issues]

        return issues
    
    def _add_auto_fix(self, issue: Dict) -> Dict:
        """Add auto-fix suggestion to issue."""
        issue_type = issue["type"]
        
        if issue_type == "sql_injection":
            issue["auto_fix"] = """# Use parameterized queries
cursor.execute(
    "SELECT * FROM users WHERE name = ?",
    (username,)
)

# Or for multiple parameters
cursor.execute(
    "SELECT * FROM users WHERE name = ? AND age = ?",
    (username, age)
)"""
        
        elif issue_type == "command_injection":
            issue["auto_fix"] = """# Use subprocess with shell=False
import subprocess

result = subprocess.run(
    ["command", "arg1", "arg2"],  # Pass as list
    shell=False,  # Never use shell=True with user input
    capture_output=True,
    text=True
)"""
        
        elif issue_type == "hardcoded_secret":
            issue["auto_fix"] = """# Use environment variables
import os

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable not set")

# Or use python-dotenv
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("API_KEY")"""
        
        elif issue_type == "path_traversal":
            issue["auto_fix"] = """# Validate and sanitize paths
from pathlib import Path

def safe_open_file(user_path: str, base_dir: str):
    # Resolve to absolute path
    full_path = Path(base_dir) / user_path
    full_path = full_path.resolve()
    
    # Ensure it's within base_dir
    if not str(full_path).startswith(str(Path(base_dir).resolve())):
        raise ValueError("Path traversal detected")
    
    return open(full_path)"""
        
        elif issue_type == "unsafe_deserialization":
            issue["auto_fix"] = """# Use safe alternatives

# For YAML: use SafeLoader
import yaml
data = yaml.safe_load(file_content)
# or
data = yaml.load(file_content, Loader=yaml.SafeLoader)

# For pickle: validate source or use JSON instead
import json
data = json.loads(file_content)  # Safer alternative"""
        
        return issue

    def _detect_sql_injection(self, tree: ast.AST, node: Dict) -> List[Dict]:
        """Detect SQL injection vulnerabilities.

        Catches both direct concatenation in execute() args **and** indirect
        patterns where a variable is assigned a concatenated/formatted string
        and then passed to execute().
        """
        issues = []

        # ── Phase 1: collect tainted variable names ───────────────────
        # A variable is "tainted" when its value comes from string
        # concatenation, f-string interpolation, or .format().
        tainted_vars: set[str] = set()
        for ast_node in ast.walk(tree):
            if isinstance(ast_node, ast.Assign):
                val = ast_node.value
                is_tainted = (
                    (isinstance(val, ast.BinOp) and isinstance(val.op, ast.Add))
                    or isinstance(val, ast.JoinedStr)
                    or (isinstance(val, ast.Call)
                        and isinstance(val.func, ast.Attribute)
                        and val.func.attr == "format")
                )
                if is_tainted:
                    for target in ast_node.targets:
                        if isinstance(target, ast.Name):
                            tainted_vars.add(target.id)

        # ── Phase 2: inspect .execute() / .executemany() / .raw() ────
        for ast_node in ast.walk(tree):
            # Look for string formatting in SQL-like strings
            if isinstance(ast_node, ast.Call):
                if isinstance(ast_node.func, ast.Attribute):
                    # Check for .execute() with string concatenation
                    if ast_node.func.attr in ["execute", "executemany", "raw"]:
                        if ast_node.args:
                            arg = ast_node.args[0]
                            
                            # Check if it's string concatenation (BinOp with Add)
                            if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Add):
                                issues.append({
                                    "type": "sql_injection",
                                    "severity": "critical",
                                    "line": node["start_line"] + ast_node.lineno - 1,
                                    "message": "Potential SQL injection via string concatenation",
                                    "suggestion": "Use parameterized queries with placeholders (?)",
                                    "code_snippet": ast.unparse(ast_node)[:100]
                                })
                            
                            # Check for f-strings
                            elif isinstance(arg, ast.JoinedStr):
                                issues.append({
                                    "type": "sql_injection",
                                    "severity": "critical",
                                    "line": node["start_line"] + ast_node.lineno - 1,
                                    "message": "Potential SQL injection via f-string",
                                    "suggestion": "Use parameterized queries instead of f-strings",
                                    "code_snippet": ast.unparse(ast_node)[:100]
                                })
                            
                            # Check for .format()
                            elif isinstance(arg, ast.Call):
                                if isinstance(arg.func, ast.Attribute) and arg.func.attr == "format":
                                    issues.append({
                                        "type": "sql_injection",
                                        "severity": "critical",
                                        "line": node["start_line"] + ast_node.lineno - 1,
                                        "message": "Potential SQL injection via .format()",
                                        "suggestion": "Use parameterized queries with placeholders",
                                        "code_snippet": ast.unparse(ast_node)[:100]
                                    })

                            # Check for indirect: variable previously assigned
                            # from concatenation / f-string / .format()
                            elif (isinstance(arg, ast.Name)
                                  and arg.id in tainted_vars):
                                issues.append({
                                    "type": "sql_injection",
                                    "severity": "critical",
                                    "line": node["start_line"] + ast_node.lineno - 1,
                                    "message": "Potential SQL injection via tainted variable",
                                    "suggestion": "Use parameterized queries with placeholders (?)",
                                    "code_snippet": ast.unparse(ast_node)[:100]
                                })

        return issues

    def _detect_command_injection(self, tree: ast.AST, node: Dict) -> List[Dict]:
        """Detect command injection risks."""
        issues = []

        for ast_node in ast.walk(tree):
            if isinstance(ast_node, ast.Call):
                func_name = None

                if isinstance(ast_node.func, ast.Name):
                    func_name = ast_node.func.id
                elif isinstance(ast_node.func, ast.Attribute):
                    func_name = ast_node.func.attr

                if func_name in self.dangerous_functions:
                    issue_type = self.dangerous_functions[func_name]
                    
                    issues.append({
                        "type": issue_type,
                        "severity": "critical",
                        "line": node["start_line"] + ast_node.lineno - 1,
                        "message": f"Unsafe use of '{func_name}()' with potential user input",
                        "suggestion": "Use subprocess.run() with shell=False and validate inputs",
                        "code_snippet": ast.unparse(ast_node)[:100]
                    })

                # Check for subprocess with shell=True
                if func_name in ["run", "call", "Popen"]:
                    for keyword in ast_node.keywords:
                        if keyword.arg == "shell":
                            if isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                                issues.append({
                                    "type": "command_injection",
                                    "severity": "high",
                                    "line": node["start_line"] + ast_node.lineno - 1,
                                    "message": "subprocess called with shell=True",
                                    "suggestion": "Use shell=False and pass command as list",
                                    "code_snippet": ast.unparse(ast_node)[:100]
                                })

        return issues

    def _detect_hardcoded_secrets(self, node: Dict) -> List[Dict]:
        """Detect hardcoded secrets in code."""
        issues = []
        code = node["code"]

        for pattern, secret_type in self.secret_patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                # Skip if it looks like a placeholder
                value = match.group(1)
                placeholders = ["your_key_here", "xxx", "***", "placeholder", "example", "test", "dummy"]
                
                if any(p in value.lower() for p in placeholders):
                    continue
                
                # Skip very short values (likely not real secrets)
                if len(value) < 8:
                    continue

                issues.append({
                    "type": "hardcoded_secret",
                    "severity": "high",
                    "line": node["start_line"] + code[:match.start()].count('\n'),
                    "message": f"Hardcoded {secret_type} found",
                    "suggestion": "Use environment variables or secret management (e.g., os.getenv())",
                    "code_snippet": f"{secret_type.lower()}=***"
                })

        return issues

    def _detect_path_traversal(self, tree: ast.AST, node: Dict) -> List[Dict]:
        """Detect path traversal vulnerabilities."""
        issues = []

        for ast_node in ast.walk(tree):
            # Look for file operations
            if isinstance(ast_node, ast.Call):
                func_name = None
                
                if isinstance(ast_node.func, ast.Name):
                    func_name = ast_node.func.id
                elif isinstance(ast_node.func, ast.Attribute):
                    func_name = ast_node.func.attr

                # File operations that could be vulnerable
                if func_name in ["open", "read", "write", "remove", "unlink", "rmdir"]:
                    # Check if path comes from string concatenation (potential user input)
                    if ast_node.args:
                        path_arg = ast_node.args[0]
                        
                        if isinstance(path_arg, (ast.BinOp, ast.JoinedStr)):
                            issues.append({
                                "type": "path_traversal",
                                "severity": "medium",
                                "line": node["start_line"] + ast_node.lineno - 1,
                                "message": "Potential path traversal if path comes from user input",
                                "suggestion": "Validate and sanitize file paths, use Path.resolve() and check against allowed directories",
                                "code_snippet": ast.unparse(ast_node)[:100]
                            })

        return issues

    def _detect_unsafe_deserialization(self, tree: ast.AST, node: Dict) -> List[Dict]:
        """Detect unsafe deserialization (pickle, yaml)."""
        issues = []

        for ast_node in ast.walk(tree):
            if isinstance(ast_node, ast.Call):
                # Check for pickle.loads, pickle.load
                if isinstance(ast_node.func, ast.Attribute):
                    if isinstance(ast_node.func.value, ast.Name):
                        if ast_node.func.value.id == "pickle" and ast_node.func.attr in ["loads", "load"]:
                            issues.append({
                                "type": "unsafe_deserialization",
                                "severity": "high",
                                "line": node["start_line"] + ast_node.lineno - 1,
                                "message": "Unsafe deserialization with pickle on untrusted data",
                                "suggestion": "Use JSON or validate data source before unpickling",
                                "code_snippet": ast.unparse(ast_node)[:100]
                            })

                # Check for yaml.load without safe loader
                if isinstance(ast_node.func, ast.Attribute):
                    if isinstance(ast_node.func.value, ast.Name):
                        if ast_node.func.value.id == "yaml" and ast_node.func.attr == "load":
                            # Check if Loader is specified
                            has_safe_loader = False
                            for keyword in ast_node.keywords:
                                if keyword.arg == "Loader":
                                    if isinstance(keyword.value, ast.Attribute):
                                        if keyword.value.attr in ["SafeLoader", "BaseLoader"]:
                                            has_safe_loader = True
                            
                            if not has_safe_loader:
                                issues.append({
                                    "type": "unsafe_deserialization",
                                    "severity": "high",
                                    "line": node["start_line"] + ast_node.lineno - 1,
                                    "message": "yaml.load() without SafeLoader",
                                    "suggestion": "Use yaml.safe_load() or yaml.load(data, Loader=yaml.SafeLoader)",
                                    "code_snippet": ast.unparse(ast_node)[:100]
                                })

        return issues

    def scan_project(self) -> Dict[str, List[Dict]]:
        """Scan entire project for security issues.
        
        Returns:
            Dictionary mapping file paths to lists of security issues
        """
        results = {}

        # Get all unique file paths
        all_nodes = self.store.get_nodes()
        file_paths = set(node["file_path"] for node in all_nodes)

        for file_path in file_paths:
            issues = self.scan_file(file_path)
            if issues:
                results[file_path] = issues

        return results
