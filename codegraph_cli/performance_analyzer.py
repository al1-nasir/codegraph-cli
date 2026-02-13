"""Performance issue analyzer."""

from __future__ import annotations

import ast
from typing import Dict, List, Set

from .storage import GraphStore


class PerformanceAnalyzer:
    """Analyze code for performance issues."""

    def __init__(self, store: GraphStore):
        self.store = store

        # Database-related method names that indicate queries
        self.query_methods = {
            "execute", "executemany", "query", "get", "filter",
            "all", "first", "one", "fetch", "fetchone", "fetchall",
            "select", "insert", "update", "delete"
        }

    def analyze_file(self, file_path: str) -> List[Dict]:
        """Analyze file for performance issues.
        
        Args:
            file_path: Path to file to analyze
            
        Returns:
            List of performance issue dictionaries
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
                self._detect_n_plus_one(tree, node)
                + self._detect_inefficient_algorithms(tree, node)
                + self._detect_memory_issues(tree, node)
            ):
                key = (issue["line"], issue["type"])
                if key not in seen:
                    seen.add(key)
                    issues.append(issue)

        return issues

    def _detect_n_plus_one(self, tree: ast.AST, node: Dict) -> List[Dict]:
        """Detect N+1 query patterns."""
        issues = []

        for ast_node in ast.walk(tree):
            # Look for loops with database queries inside
            if isinstance(ast_node, (ast.For, ast.While)):
                # Check for query calls inside the loop
                for inner_node in ast.walk(ast_node):
                    if isinstance(inner_node, ast.Call):
                        # Check for query-like method names
                        if isinstance(inner_node.func, ast.Attribute):
                            if inner_node.func.attr in self.query_methods:
                                issues.append({
                                    "type": "n_plus_one_query",
                                    "severity": "high",
                                    "line": node["start_line"] + inner_node.lineno - 1,
                                    "message": "Potential N+1 query pattern: database query inside loop",
                                    "suggestion": "Use bulk queries, eager loading, or prefetch_related()",
                                    "code_snippet": ast.unparse(inner_node)[:100]
                                })
                                break  # Only report once per loop

        return issues

    def _detect_inefficient_algorithms(self, tree: ast.AST, node: Dict) -> List[Dict]:
        """Detect inefficient algorithm patterns."""
        issues = []

        # Track nested loops for O(n²) detection
        loop_depth = {}
        
        for ast_node in ast.walk(tree):
            if isinstance(ast_node, (ast.For, ast.While)):
                # Check for nested loops
                nested_loops = []
                for inner in ast.walk(ast_node):
                    if inner != ast_node and isinstance(inner, (ast.For, ast.While)):
                        nested_loops.append(inner)
                
                if nested_loops:
                    # Report the innermost nested loop
                    innermost = nested_loops[0]
                    issues.append({
                        "type": "nested_loop",
                        "severity": "medium",
                        "line": node["start_line"] + innermost.lineno - 1,
                        "message": "Nested loop detected (O(n²) complexity)",
                        "suggestion": "Consider using hash maps, sets, or optimizing algorithm",
                        "code_snippet": f"for ... in ...: for ... in ..."
                    })

            # Detect list operations in loops (inefficient)
            if isinstance(ast_node, ast.For):
                for inner in ast.walk(ast_node):
                    # Check for list.append in loop with large iterations
                    if isinstance(inner, ast.Call):
                        if isinstance(inner.func, ast.Attribute):
                            # Detect repeated string concatenation
                            if inner.func.attr == "append":
                                # Check if it's appending to a list that's later joined
                                # This is actually efficient, so skip
                                pass
                            
                            # Detect list.insert(0, ...) which is O(n)
                            elif inner.func.attr == "insert":
                                if inner.args and isinstance(inner.args[0], ast.Constant):
                                    if inner.args[0].value == 0:
                                        issues.append({
                                            "type": "inefficient_operation",
                                            "severity": "medium",
                                            "line": node["start_line"] + inner.lineno - 1,
                                            "message": "list.insert(0, ...) in loop is O(n²)",
                                            "suggestion": "Use collections.deque or append and reverse",
                                            "code_snippet": ast.unparse(inner)[:100]
                                        })

        # Detect string concatenation in loops
        for ast_node in ast.walk(tree):
            if isinstance(ast_node, (ast.For, ast.While)):
                for inner in ast.walk(ast_node):
                    # Look for += on strings
                    if isinstance(inner, ast.AugAssign):
                        if isinstance(inner.op, ast.Add):
                            # Check if target is likely a string
                            if isinstance(inner.target, ast.Name):
                                issues.append({
                                    "type": "string_concatenation_loop",
                                    "severity": "low",
                                    "line": node["start_line"] + inner.lineno - 1,
                                    "message": "String concatenation in loop (inefficient)",
                                    "suggestion": "Use list.append() and ''.join() instead",
                                    "code_snippet": ast.unparse(inner)[:100]
                                })

        return issues

    def _detect_memory_issues(self, tree: ast.AST, node: Dict) -> List[Dict]:
        """Detect memory inefficiencies."""
        issues = []
        seen_lines: set = set()

        for ast_node in ast.walk(tree):
            report_line = node["start_line"] + ast_node.lineno - 1 if hasattr(ast_node, 'lineno') else None

            # Detect large list comprehensions that could be generators
            if isinstance(ast_node, ast.ListComp):
                if report_line and report_line not in seen_lines:
                    seen_lines.add(report_line)
                    issues.append({
                        "type": "memory_inefficiency",
                        "severity": "low",
                        "line": report_line,
                        "message": "List comprehension could be a generator expression",
                        "suggestion": "Use (...) instead of [...] if you only iterate once",
                        "code_snippet": ast.unparse(ast_node)[:100]
                    })

            # Detect reading entire file into memory
            if isinstance(ast_node, ast.Call):
                if isinstance(ast_node.func, ast.Attribute):
                    # file.read() without size limit
                    if ast_node.func.attr == "read" and not ast_node.args:
                        issues.append({
                            "type": "memory_inefficiency",
                            "severity": "medium",
                            "line": node["start_line"] + ast_node.lineno - 1,
                            "message": "Reading entire file into memory",
                            "suggestion": "Read in chunks or iterate line by line",
                            "code_snippet": ast.unparse(ast_node)[:100]
                        })
                    
                    # .readlines() also loads entire file
                    elif ast_node.func.attr == "readlines":
                        issues.append({
                            "type": "memory_inefficiency",
                            "severity": "medium",
                            "line": node["start_line"] + ast_node.lineno - 1,
                            "message": "readlines() loads entire file into memory",
                            "suggestion": "Iterate over file object directly: for line in file:",
                            "code_snippet": ast.unparse(ast_node)[:100]
                        })

        return issues

    def analyze_project(self) -> Dict[str, List[Dict]]:
        """Analyze entire project for performance issues.
        
        Returns:
            Dictionary mapping file paths to lists of performance issues
        """
        results = {}

        # Get all unique file paths
        all_nodes = self.store.get_nodes()
        file_paths = set(node["file_path"] for node in all_nodes)

        for file_path in file_paths:
            issues = self.analyze_file(file_path)
            if issues:
                results[file_path] = issues

        return results
