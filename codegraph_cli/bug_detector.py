"""Bug detection using AST analysis and pattern matching."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Optional, Set

from .llm import LocalLLM
from .storage import GraphStore


class BugDetector:
    """Detect potential bugs using AST + LLM analysis."""

    def __init__(self, store: GraphStore, llm: Optional[LocalLLM] = None):
        self.store = store
        self.llm = llm

    def analyze_file(self, file_path: str, use_llm: bool = False) -> List[Dict]:
        """Analyze a file for potential bugs.
        
        Args:
            file_path: Path to file to analyze
            use_llm: Whether to use LLM for deeper analysis
            
        Returns:
            List of issue dictionaries with type, severity, line, message, suggestion
        """
        issues = []

        # Get all nodes in this file, but skip the module-level node when
        # individual function/class nodes exist to avoid duplicate analysis.
        nodes = [n for n in self.store.get_nodes() if n["file_path"] == file_path]
        has_children = any(n["node_type"] != "module" for n in nodes)
        if has_children:
            nodes = [n for n in nodes if n["node_type"] != "module"]

        seen: set = set()  # (line, type) dedup across all nodes

        for node in nodes:
            # Parse code to AST
            try:
                tree = ast.parse(node["code"])
            except SyntaxError:
                continue

            # Run detectors
            for issue in (
                self._detect_null_risks(tree, node)
                + self._detect_logic_errors(tree, node)
                + self._detect_resource_leaks(tree, node)
            ):
                key = (issue["line"], issue["type"])
                if key not in seen:
                    seen.add(key)
                    issues.append(issue)

        # Enhance with LLM analysis if available
        if use_llm and self.llm and issues:
            issues = self._enhance_with_llm(issues, file_path)

        return issues
    
    def _enhance_with_llm(self, issues: List[Dict], file_path: str) -> List[Dict]:
        """Enhance issues with LLM-powered explanations and fixes."""
        enhanced_issues = []
        
        for issue in issues:
            # Generate auto-fix if possible
            auto_fix = self._generate_auto_fix(issue)
            if auto_fix:
                issue["auto_fix"] = auto_fix
            
            # Get LLM explanation for complex issues
            if issue["severity"] in ["high", "critical"]:
                explanation = self._get_llm_explanation(issue)
                if explanation:
                    issue["llm_explanation"] = explanation
            
            enhanced_issues.append(issue)
        
        return enhanced_issues
    
    def _generate_auto_fix(self, issue: Dict) -> Optional[str]:
        """Generate automatic fix for the issue."""
        issue_type = issue["type"]
        
        if issue_type == "null_pointer_risk":
            # Extract variable name from message
            # Example: "Potential None access on 'user.name'"
            msg = issue["message"]
            if "'" in msg:
                parts = msg.split("'")
                if len(parts) >= 2:
                    var_access = parts[1]  # e.g., "user.name"
                    if "." in var_access:
                        var_name = var_access.split(".")[0]
                        return f"""if {var_name} is not None:
    # Your code here
    result = {var_access}
else:
    # Handle None case
    result = None"""
        
        elif issue_type == "resource_leak":
            # Suggest using 'with' statement
            if "code_snippet" in issue:
                snippet = issue["code_snippet"]
                if "open(" in snippet:
                    return f"""with {snippet} as f:
    # Your code here
    data = f.read()"""
        
        elif issue_type == "infinite_loop":
            return """# Add a break condition
max_iterations = 1000
iteration = 0
while True:
    iteration += 1
    if iteration >= max_iterations:
        break
    # Your loop code here"""
        
        return None
    
    def _get_llm_explanation(self, issue: Dict) -> Optional[str]:
        """Get LLM explanation for the issue."""
        if not self.llm:
            return None
        
        prompt = f"""Explain this code issue in 2-3 sentences:

Issue Type: {issue['type']}
Severity: {issue['severity']}
Message: {issue['message']}
Code: {issue.get('code_snippet', 'N/A')}

Provide a clear explanation of:
1. Why this is a problem
2. What could go wrong
3. Best practice to fix it"""
        
        try:
            explanation = self.llm.explain(prompt)
            return explanation.strip()
        except Exception:
            return None

    def _detect_null_risks(self, tree: ast.AST, node: Dict) -> List[Dict]:
        """Detect potential None access without checks.
        
        Looks for:
        - Attribute access on variables that could be None
        - Dictionary access without .get()
        - List access without bounds checking
        """
        issues = []
        seen_lines: set = set()  # Deduplicate by line number

        # Collect decorator lines so we skip attribute access on decorators
        decorator_lines: set = set()
        for ast_node in ast.walk(tree):
            if isinstance(ast_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for dec in ast_node.decorator_list:
                    # Mark all lines spanned by decorator expressions
                    for sub in ast.walk(dec):
                        if hasattr(sub, 'lineno'):
                            decorator_lines.add(sub.lineno)

        for ast_node in ast.walk(tree):
            # Look for attribute access that might fail
            if isinstance(ast_node, ast.Attribute):
                # Skip decorator expressions (e.g. @app.route)
                if ast_node.lineno in decorator_lines:
                    continue

                # Get the variable being accessed
                if isinstance(ast_node.value, ast.Name):
                    var_name = ast_node.value.id

                    # Skip well-known safe names
                    if var_name in ('self', 'cls', 'super', 'os', 'sys', 'math',
                                    'json', 're', 'logging', 'typing', 'pathlib'):
                        continue

                    # Deduplicate: only report once per source line
                    report_line = node["start_line"] + ast_node.lineno - 1
                    if report_line in seen_lines:
                        continue

                    # Check if this variable is assigned from a function that could return None
                    if self._could_be_none(tree, var_name, ast_node.lineno):
                        seen_lines.add(report_line)
                        issues.append({
                            "type": "null_pointer_risk",
                            "severity": "medium",
                            "line": report_line,
                            "message": f"Potential None access on '{var_name}.{ast_node.attr}'",
                            "suggestion": f"Add None check: if {var_name} is not None:",
                            "code_snippet": ast.unparse(ast_node)
                        })

        return issues

    def _could_be_none(self, tree: ast.AST, var_name: str, access_line: int) -> bool:
        """Check if a variable could be None at the point of access.
        
        This is a simplified heuristic that checks:
        - If variable is assigned from a function call (could return None)
        - If there's no None check between assignment and access
        - Excludes constructors (ClassName(...)) which never return None
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == var_name:
                        # Check if assigned from function call
                        if isinstance(node.value, ast.Call):
                            # --- Skip class constructors / capitalized calls ---
                            # e.g.  app = Flask(__name__)  or  data = dict()
                            call_name = self._get_call_name(node.value)
                            if call_name and call_name[0].isupper():
                                return False  # Constructor – never None
                            # Also skip well-known stdlib constructors
                            if call_name in ('dict', 'list', 'set', 'tuple',
                                             'frozenset', 'bytearray', 'bytes',
                                             'str', 'int', 'float', 'bool',
                                             'object', 'type', 'super',
                                             'defaultdict', 'OrderedDict',
                                             'Counter', 'deque', 'namedtuple'):
                                return False
                            # Check if there's a None check before access
                            if not self._has_none_check_between(tree, var_name, node.lineno, access_line):
                                return True
        
        return False

    @staticmethod
    def _get_call_name(call_node: ast.Call) -> Optional[str]:
        """Extract the simple function/class name from a Call node."""
        func = call_node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return None

    def _has_none_check_between(self, tree: ast.AST, var_name: str, start_line: int, end_line: int) -> bool:
        """Check if there's a None check for var_name between start and end lines."""
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Check if this is a None check
                if hasattr(node, 'lineno') and start_line < node.lineno < end_line:
                    # Look for patterns like: if var is not None, if var, etc.
                    test = node.test
                    if isinstance(test, ast.Compare):
                        if isinstance(test.left, ast.Name) and test.left.id == var_name:
                            # Check for "is not None" pattern
                            if any(isinstance(op, ast.IsNot) for op in test.ops):
                                return True
                    elif isinstance(test, ast.Name) and test.id == var_name:
                        # Simple truthiness check
                        return True
        
        return False

    def _detect_logic_errors(self, tree: ast.AST, node: Dict) -> List[Dict]:
        """Detect logic errors like unreachable code and infinite loops."""
        issues = []

        for ast_node in ast.walk(tree):
            # Detect unreachable code after return
            if isinstance(ast_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for i, stmt in enumerate(ast_node.body):
                    if isinstance(stmt, ast.Return) and i < len(ast_node.body) - 1:
                        # Check if next statement is not just a comment or pass
                        next_stmt = ast_node.body[i + 1]
                        if not isinstance(next_stmt, ast.Pass):
                            issues.append({
                                "type": "unreachable_code",
                                "severity": "low",
                                "line": node["start_line"] + next_stmt.lineno - 1,
                                "message": "Unreachable code after return statement",
                                "suggestion": "Remove unreachable code or move return to end",
                                "code_snippet": ast.unparse(next_stmt)[:100]
                            })

            # Detect infinite loops
            if isinstance(ast_node, ast.While):
                if isinstance(ast_node.test, ast.Constant) and ast_node.test.value is True:
                    # Check if there's a break statement
                    has_break = any(isinstance(n, ast.Break) for n in ast.walk(ast_node))
                    if not has_break:
                        issues.append({
                            "type": "infinite_loop",
                            "severity": "high",
                            "line": node["start_line"] + ast_node.lineno - 1,
                            "message": "Potential infinite loop without break statement",
                            "suggestion": "Add break condition or change loop condition",
                            "code_snippet": f"while True: ..."
                        })

            # Detect comparison with True/False (code smell)
            if isinstance(ast_node, ast.Compare):
                for op, comparator in zip(ast_node.ops, ast_node.comparators):
                    if isinstance(comparator, ast.Constant) and isinstance(comparator.value, bool):
                        if isinstance(op, ast.Eq):
                            issues.append({
                                "type": "boolean_comparison",
                                "severity": "low",
                                "line": node["start_line"] + ast_node.lineno - 1,
                                "message": f"Unnecessary comparison with {comparator.value}",
                                "suggestion": "Use variable directly in condition",
                                "code_snippet": ast.unparse(ast_node)
                            })

        return issues

    def _detect_resource_leaks(self, tree: ast.AST, node: Dict) -> List[Dict]:
        """Detect unclosed resources (files, connections)."""
        issues = []

        # Track 'with' statement contexts
        with_contexts: Set[int] = set()
        for ast_node in ast.walk(tree):
            if isinstance(ast_node, ast.With):
                with_contexts.add(id(ast_node))

        # Look for resource-opening calls
        for ast_node in ast.walk(tree):
            if isinstance(ast_node, ast.Call):
                func_name = None
                
                if isinstance(ast_node.func, ast.Name):
                    func_name = ast_node.func.id
                elif isinstance(ast_node.func, ast.Attribute):
                    func_name = ast_node.func.attr

                # Check for file operations
                if func_name == "open":
                    # Check if this call is inside a 'with' statement
                    is_in_with = self._is_inside_with(tree, ast_node)
                    
                    if not is_in_with:
                        issues.append({
                            "type": "resource_leak",
                            "severity": "medium",
                            "line": node["start_line"] + ast_node.lineno - 1,
                            "message": "File opened without 'with' statement",
                            "suggestion": "Use 'with open(...) as f:' to ensure file is closed",
                            "code_snippet": ast.unparse(ast_node)
                        })

        return issues

    def _is_inside_with(self, tree: ast.AST, target: ast.AST) -> bool:
        """Check if target node is inside a 'with' statement."""
        # This is a simplified check - proper implementation would need
        # to track the AST hierarchy
        for node in ast.walk(tree):
            if isinstance(node, ast.With):
                # Check if target is in the body
                for item in node.items:
                    if item.context_expr == target:
                        return True
        return False

    def analyze_project(self) -> Dict[str, List[Dict]]:
        """Analyze entire project for bugs.
        
        Returns:
            Dictionary mapping file paths to lists of issues
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
