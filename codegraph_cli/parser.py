"""Semantic code parser using Tree-sitter for multi-language AST extraction.

Replaces the legacy ast-based parser with Tree-sitter for:
- Error-tolerant parsing (handles broken / incomplete syntax gracefully)
- Multi-language support (Python now; JS/TS/Go extensible)
- Semantic chunking by function / class definition (not line-count windows)

Falls back to Python's built-in ``ast`` module when tree-sitter is unavailable.
"""

from __future__ import annotations

import ast
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import Edge, Node

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language <-> file-extension mapping (extensible)
# ---------------------------------------------------------------------------
LANGUAGE_MAP: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".cpp": "cpp",
    ".c": "c",
    ".cs": "c_sharp",
}

SKIP_DIRS: Set[str] = {
    ".venv", "venv", "__pycache__", "node_modules", ".git",
    "site-packages", ".tox", ".pytest_cache", "build", "dist",
    ".mypy_cache", ".ruff_cache", "htmlcov", ".eggs",
    "egg-info", ".codegraph", "lancedb",
}


# ===================================================================
# Abstract Parser Interface
# ===================================================================

class Parser(ABC):
    """Abstract base class for all code parsers."""

    @abstractmethod
    def parse_file(
        self,
        file_path: Path,
        source: Optional[str] = None,
    ) -> Tuple[List[Node], List[Edge]]:
        """Parse a single file into nodes and edges."""
        ...

    @abstractmethod
    def parse_project(self) -> Tuple[List[Node], List[Edge]]:
        """Parse the entire project rooted at *project_root*."""
        ...

    @abstractmethod
    def supports_language(self, language: str) -> bool:
        """Return True if this parser can handle *language*."""
        ...


# ===================================================================
# Tree-sitter Parser (Primary)
# ===================================================================

class TreeSitterParser(Parser):
    """Error-tolerant, multi-language parser built on Tree-sitter.

    Uses ``tree-sitter-languages`` for pre-built grammars so setup is
    zero-config for the end-user.  Tree-sitter produces a *concrete
    syntax tree* (CST) that preserves every token, allowing reliable
    extraction even when the source has minor syntax errors.
    """

    def __init__(
        self,
        project_root: Path,
        languages: Optional[List[str]] = None,
    ) -> None:
        self.project_root = project_root
        self._parsers: Dict[str, Any] = {}
        self._requested_languages = languages or ["python"]
        self._init_parsers()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    # Map language name -> module that provides the tree-sitter Language
    _GRAMMAR_MODULES: Dict[str, str] = {
        "python": "tree_sitter_python",
        "javascript": "tree_sitter_javascript",
        "typescript": "tree_sitter_typescript",
    }

    def _init_parsers(self) -> None:
        try:
            import tree_sitter  # type: ignore[import-untyped]  # noqa: F401
        except ImportError:
            logger.warning(
                "tree-sitter is not installed -- "
                "Tree-sitter parsing unavailable. "
                "Install with: pip install tree-sitter tree-sitter-python"
            )
            return

        from tree_sitter import Language, Parser as TSParser  # type: ignore[import-untyped]

        for lang in self._requested_languages:
            mod_name = self._GRAMMAR_MODULES.get(lang)
            if mod_name is None:
                logger.warning("No grammar module mapped for language '%s'", lang)
                continue
            try:
                import importlib
                mod = importlib.import_module(mod_name)
                # tree-sitter >=0.22 per-language packages expose a
                # language() function that returns the Language capsule.
                ts_lang = Language(mod.language())
                parser = TSParser(ts_lang)
                self._parsers[lang] = parser
                logger.debug("Loaded tree-sitter parser for %s", lang)
            except ImportError:
                logger.warning(
                    "Grammar package '%s' not installed for language '%s'. "
                    "Install with: pip install %s",
                    mod_name, lang, mod_name.replace('_', '-'),
                )
            except Exception as exc:
                logger.warning("Could not load tree-sitter grammar for %s: %s", lang, exc)

    def supports_language(self, language: str) -> bool:
        return language in self._parsers

    # ------------------------------------------------------------------
    # Project-level parsing
    # ------------------------------------------------------------------

    def parse_project(self) -> Tuple[List[Node], List[Edge]]:
        all_nodes: List[Node] = []
        all_edges: List[Edge] = []

        for ext, lang in LANGUAGE_MAP.items():
            if lang not in self._parsers:
                continue
            for file_path in sorted(self.project_root.rglob(f"*{ext}")):
                if any(part in SKIP_DIRS for part in file_path.parts):
                    continue
                try:
                    nodes, edges = self.parse_file(file_path)
                    all_nodes.extend(nodes)
                    all_edges.extend(edges)
                except Exception as exc:
                    logger.warning("Failed to parse %s: %s", file_path, exc)

        all_edges = _resolve_call_edges(all_nodes, all_edges)
        return all_nodes, all_edges

    # ------------------------------------------------------------------
    # File-level parsing
    # ------------------------------------------------------------------

    def parse_file(
        self,
        file_path: Path,
        source: Optional[str] = None,
    ) -> Tuple[List[Node], List[Edge]]:
        if source is None:
            source = file_path.read_text(encoding="utf-8", errors="ignore")

        ext = file_path.suffix
        lang = LANGUAGE_MAP.get(ext)
        if not lang or lang not in self._parsers:
            return [], []

        parser = self._parsers[lang]
        source_bytes = source.encode("utf-8")
        tree = parser.parse(source_bytes)

        rel_path = str(file_path.relative_to(self.project_root))
        lines = source.splitlines()

        # -- Module node --------------------------------------------------
        module_name = rel_path.replace("/", ".").removesuffix(".py")
        module_id = f"module:{module_name}"
        module_node = Node(
            node_id=module_id,
            node_type="module",
            name=module_name.split(".")[-1],
            qualname=module_name,
            file_path=rel_path,
            start_line=1,
            end_line=max(len(lines), 1),
            code=source,
            docstring=self._extract_module_docstring(tree.root_node),
        )

        nodes: List[Node] = [module_node]
        edges: List[Edge] = []

        # -- Language-specific extraction ---------------------------------
        if lang == "python":
            self._walk_python(
                tree.root_node,
                scope_stack=[module_name],
                scope_id_stack=[module_id],
                rel_path=rel_path,
                lines=lines,
                nodes=nodes,
                edges=edges,
            )
            self._extract_python_imports(tree.root_node, module_id, edges)
        # Future: elif lang in ("javascript", "typescript"): ...

        return nodes, edges

    # ------------------------------------------------------------------
    # Python: recursive definition walker
    # ------------------------------------------------------------------

    def _walk_python(
        self,
        ts_node: Any,
        scope_stack: List[str],
        scope_id_stack: List[str],
        rel_path: str,
        lines: List[str],
        nodes: List[Node],
        edges: List[Edge],
    ) -> None:
        """Recursively extract class / function definitions from *ts_node*."""
        for child in ts_node.children:
            outer_node = child
            actual_def = child

            # Unwrap @decorated_definition -> inner function/class
            if child.type == "decorated_definition":
                inner = child.child_by_field_name("definition")
                if inner is None:
                    continue
                actual_def = inner

            if actual_def.type == "function_definition":
                self._process_python_function(
                    outer_node, actual_def, scope_stack, scope_id_stack,
                    rel_path, lines, nodes, edges,
                )
            elif actual_def.type == "class_definition":
                self._process_python_class(
                    outer_node, actual_def, scope_stack, scope_id_stack,
                    rel_path, lines, nodes, edges,
                )

    def _process_python_function(
        self,
        outer_node: Any,
        func_node: Any,
        scope_stack: List[str],
        scope_id_stack: List[str],
        rel_path: str,
        lines: List[str],
        nodes: List[Node],
        edges: List[Edge],
    ) -> None:
        name_node = func_node.child_by_field_name("name")
        if name_node is None:
            return
        name: str = name_node.text.decode("utf-8")
        qualname = ".".join(scope_stack + [name])
        node_id = f"function:{qualname}"

        start_line = outer_node.start_point[0] + 1
        end_line = outer_node.end_point[0] + 1
        code = "\n".join(lines[start_line - 1: end_line])

        nodes.append(Node(
            node_id=node_id,
            node_type="function",
            name=name,
            qualname=qualname,
            file_path=rel_path,
            start_line=start_line,
            end_line=end_line,
            code=code,
            docstring=self._extract_docstring(func_node),
        ))
        edges.append(Edge(src=scope_id_stack[-1], dst=node_id, edge_type="contains"))

        # Call edges from function body
        for call_name in self._collect_calls(func_node):
            edges.append(Edge(src=node_id, dst=call_name, edge_type="calls"))

        # Recurse into body for nested definitions
        body = func_node.child_by_field_name("body")
        if body is not None:
            self._walk_python(
                body,
                scope_stack + [name],
                scope_id_stack + [node_id],
                rel_path, lines, nodes, edges,
            )

    def _process_python_class(
        self,
        outer_node: Any,
        class_node: Any,
        scope_stack: List[str],
        scope_id_stack: List[str],
        rel_path: str,
        lines: List[str],
        nodes: List[Node],
        edges: List[Edge],
    ) -> None:
        name_node = class_node.child_by_field_name("name")
        if name_node is None:
            return
        name: str = name_node.text.decode("utf-8")
        qualname = ".".join(scope_stack + [name])
        node_id = f"class:{qualname}"

        start_line = outer_node.start_point[0] + 1
        end_line = outer_node.end_point[0] + 1
        code = "\n".join(lines[start_line - 1: end_line])

        nodes.append(Node(
            node_id=node_id,
            node_type="class",
            name=name,
            qualname=qualname,
            file_path=rel_path,
            start_line=start_line,
            end_line=end_line,
            code=code,
            docstring=self._extract_docstring(class_node),
        ))
        edges.append(Edge(src=scope_id_stack[-1], dst=node_id, edge_type="contains"))

        # Walk class body for methods / nested classes
        body = class_node.child_by_field_name("body")
        if body is not None:
            self._walk_python(
                body,
                scope_stack + [name],
                scope_id_stack + [node_id],
                rel_path, lines, nodes, edges,
            )

    # ------------------------------------------------------------------
    # Python: imports
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_python_imports(
        root: Any,
        module_id: str,
        edges: List[Edge],
    ) -> None:
        for child in root.children:
            if child.type == "import_statement":
                for sub in child.children:
                    if sub.type == "dotted_name":
                        mod = sub.text.decode("utf-8")
                        edges.append(Edge(
                            src=module_id, dst=f"module:{mod}", edge_type="depends_on",
                        ))
                    elif sub.type == "aliased_import":
                        name_n = sub.child_by_field_name("name")
                        if name_n is not None:
                            mod = name_n.text.decode("utf-8")
                            edges.append(Edge(
                                src=module_id, dst=f"module:{mod}", edge_type="depends_on",
                            ))

            elif child.type == "import_from_statement":
                mod_node = child.child_by_field_name("module_name")
                if mod_node is None:
                    continue
                if mod_node.type == "dotted_name":
                    mod = mod_node.text.decode("utf-8")
                elif mod_node.type == "relative_import":
                    dotted: Optional[str] = None
                    for sub in mod_node.children:
                        if sub.type == "dotted_name":
                            dotted = sub.text.decode("utf-8")
                    mod = dotted or ""
                else:
                    mod = mod_node.text.decode("utf-8")
                if mod:
                    edges.append(Edge(
                        src=module_id, dst=f"module:{mod}", edge_type="depends_on",
                    ))

    # ------------------------------------------------------------------
    # Call extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_calls(func_node: Any) -> List[str]:
        """Return every function/method name called inside *func_node*."""
        calls: List[str] = []

        def _find(node: Any) -> None:
            if node.type == "call":
                func = node.child_by_field_name("function")
                if func is not None:
                    name = _resolve_ts_call_name(func)
                    if name:
                        calls.append(name)
            for ch in node.children:
                if ch.type in (
                    "function_definition",
                    "class_definition",
                    "decorated_definition",
                ):
                    continue
                _find(ch)

        body = func_node.child_by_field_name("body")
        if body is not None:
            _find(body)
        return calls

    # ------------------------------------------------------------------
    # Docstring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_docstring(def_node: Any) -> str:
        """Extract the docstring from a function / class definition node."""
        body = def_node.child_by_field_name("body")
        if body is None:
            return ""
        for child in body.children:
            if child.type == "expression_statement":
                for expr in child.children:
                    if expr.type == "string":
                        raw = expr.text.decode("utf-8")
                        for q in ('"""', "'''"):
                            if raw.startswith(q) and raw.endswith(q):
                                return raw[3:-3].strip()
                        for q in ('"', "'"):
                            if raw.startswith(q) and raw.endswith(q):
                                return raw[1:-1].strip()
                        return raw.strip()
                break
            elif child.type != "comment":
                break
        return ""

    @staticmethod
    def _extract_module_docstring(root: Any) -> str:
        """Extract the module-level docstring."""
        for child in root.children:
            if child.type == "expression_statement":
                for expr in child.children:
                    if expr.type == "string":
                        raw = expr.text.decode("utf-8")
                        for q in ('"""', "'''"):
                            if raw.startswith(q) and raw.endswith(q):
                                return raw[3:-3].strip()
                        return raw.strip()
                break
            elif child.type == "comment":
                continue
            else:
                break
        return ""


# ===================================================================
# AST Fallback Parser (when tree-sitter is not installed)
# ===================================================================

class ASTFallbackParser(Parser):
    """Pure-Python fallback using the built-in ``ast`` module.

    Only supports Python.  Used automatically when tree-sitter is missing.
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

    def supports_language(self, language: str) -> bool:
        return language == "python"

    def parse_project(self) -> Tuple[List[Node], List[Edge]]:
        nodes: List[Node] = []
        edges: List[Edge] = []
        for fp in sorted(self.project_root.rglob("*.py")):
            if any(part in SKIP_DIRS for part in fp.parts):
                continue
            try:
                n, e = self.parse_file(fp)
                nodes.extend(n)
                edges.extend(e)
            except Exception as exc:
                logger.warning("AST parse failed for %s: %s", fp, exc)
        edges = _resolve_call_edges(nodes, edges)
        return nodes, edges

    def parse_file(
        self,
        file_path: Path,
        source: Optional[str] = None,
    ) -> Tuple[List[Node], List[Edge]]:
        if source is None:
            source = file_path.read_text(encoding="utf-8", errors="ignore")

        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            logger.warning("SyntaxError in %s: %s", file_path, exc)
            return [], []

        rel_path = str(file_path.relative_to(self.project_root))
        lines = source.splitlines()
        module_name = rel_path.replace("/", ".").removesuffix(".py")
        module_id = f"module:{module_name}"

        module_node = Node(
            node_id=module_id,
            node_type="module",
            name=module_name.split(".")[-1],
            qualname=module_name,
            file_path=rel_path,
            start_line=1,
            end_line=max(len(lines), 1),
            code=source,
            docstring=ast.get_docstring(tree) or "",
        )

        visitor = _ASTVisitor(module_id, module_name, rel_path, lines)
        visitor.visit(tree)

        nodes = [module_node] + visitor.nodes
        edges = list(visitor.edges)

        for stmt in tree.body:
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    edges.append(Edge(
                        src=module_id, dst=f"module:{alias.name}", edge_type="depends_on",
                    ))
            elif isinstance(stmt, ast.ImportFrom) and stmt.module:
                edges.append(Edge(
                    src=module_id, dst=f"module:{stmt.module}", edge_type="depends_on",
                ))

        return nodes, edges


# ===================================================================
# Backward-Compatible Alias
# ===================================================================

class PythonGraphParser(Parser):
    """Drop-in replacement for the legacy ``PythonGraphParser``.

    Automatically selects **TreeSitterParser** when tree-sitter is
    available, otherwise falls back to the built-in AST parser.
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        ts = TreeSitterParser(project_root, languages=["python"])
        if ts.supports_language("python"):
            self._delegate: Parser = ts
            logger.info("Using Tree-sitter parser (error-tolerant, semantic chunking)")
        else:
            self._delegate = ASTFallbackParser(project_root)
            logger.info("Using AST fallback parser (Python only)")

    def parse_file(
        self,
        file_path: Path,
        source: Optional[str] = None,
    ) -> Tuple[List[Node], List[Edge]]:
        return self._delegate.parse_file(file_path, source)

    def parse_project(self) -> Tuple[List[Node], List[Edge]]:
        return self._delegate.parse_project()

    def supports_language(self, language: str) -> bool:
        return self._delegate.supports_language(language)


# ===================================================================
# Shared Helpers
# ===================================================================

def _resolve_ts_call_name(func_node: Any) -> Optional[str]:
    """Resolve a Tree-sitter call-function node to a dotted name string."""
    if func_node.type == "identifier":
        return func_node.text.decode("utf-8")
    if func_node.type == "attribute":
        parts: List[str] = []
        current = func_node
        while current is not None and current.type == "attribute":
            attr = current.child_by_field_name("attribute")
            if attr is not None:
                parts.append(attr.text.decode("utf-8"))
            current = current.child_by_field_name("object")
        if current is not None and current.type == "identifier":
            parts.append(current.text.decode("utf-8"))
        return ".".join(reversed(parts)) if parts else None
    if func_node.type == "call":
        inner = func_node.child_by_field_name("function")
        if inner is not None:
            return _resolve_ts_call_name(inner)
    return None


def _resolve_call_edges(nodes: List[Node], edges: List[Edge]) -> List[Edge]:
    """Resolve symbolic call destinations to concrete node IDs.

    Language-agnostic post-processing shared by every parser backend.
    """
    qual_by_name: Dict[str, List[str]] = {}
    qual_by_qualname: Dict[str, str] = {}
    for n in nodes:
        qual_by_name.setdefault(n.name, []).append(n.node_id)
        qual_by_qualname[n.qualname] = n.node_id
    node_ids = {n.node_id for n in nodes}

    resolved: List[Edge] = []
    for edge in edges:
        if edge.edge_type != "calls":
            resolved.append(edge)
            continue
        if edge.dst in node_ids:
            resolved.append(edge)
            continue

        resolved_dst: Optional[str] = None

        # --- dotted calls (self.method, obj.method) ----------------------
        if "." in edge.dst:
            parts = edge.dst.split(".")
            method_name = parts[-1]

            # self.method -> resolve inside same class
            if parts[0] == "self" and edge.src.startswith("function:"):
                src_qualname = edge.src.removeprefix("function:")
                if "." in src_qualname:
                    class_qualname = ".".join(src_qualname.split(".")[:-1])
                    target_qualname = f"{class_qualname}.{method_name}"
                    if target_qualname in qual_by_qualname:
                        resolved_dst = qual_by_qualname[target_qualname]

            if resolved_dst is None and method_name in qual_by_name:
                candidates = qual_by_name[method_name]
                src_parts = edge.src.split(":")[1].split(".") if ":" in edge.src else []
                for cand in candidates:
                    cand_parts = cand.split(":")[1].split(".") if ":" in cand else []
                    if src_parts and cand_parts and src_parts[:-1] == cand_parts[:-1]:
                        resolved_dst = cand
                        break
                if resolved_dst is None:
                    resolved_dst = candidates[0]

        # --- simple name lookups -----------------------------------------
        elif edge.dst in qual_by_name:
            resolved_dst = qual_by_name[edge.dst][0]
        elif edge.dst in qual_by_qualname:
            resolved_dst = qual_by_qualname[edge.dst]

        if resolved_dst:
            resolved.append(Edge(src=edge.src, dst=resolved_dst, edge_type="calls"))
        else:
            resolved.append(edge)

    return resolved


# ===================================================================
# Legacy AST visitor (used by ASTFallbackParser)
# ===================================================================

class _ASTVisitor(ast.NodeVisitor):
    """Walks a Python AST and collects Node / Edge objects."""

    def __init__(
        self,
        module_id: str,
        module_name: str,
        rel_path: str,
        lines: List[str],
    ) -> None:
        self.module_id = module_id
        self.module_name = module_name
        self.rel_path = rel_path
        self.lines = lines
        self.scope_stack: List[str] = [module_name]
        self.scope_id_stack: List[str] = [module_id]
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualname = self._mk_qualname(node.name)
        node_id = f"class:{qualname}"
        self.nodes.append(Node(
            node_id=node_id, node_type="class", name=node.name,
            qualname=qualname, file_path=self.rel_path,
            start_line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
            code=self._snippet(node),
            docstring=ast.get_docstring(node) or "",
        ))
        self.edges.append(Edge(
            src=self.scope_id_stack[-1], dst=node_id, edge_type="contains",
        ))
        self.scope_stack.append(node.name)
        self.scope_id_stack.append(node_id)
        self.generic_visit(node)
        self.scope_stack.pop()
        self.scope_id_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.AST) -> None:
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        qualname = self._mk_qualname(node.name)
        node_id = f"function:{qualname}"
        self.nodes.append(Node(
            node_id=node_id, node_type="function", name=node.name,
            qualname=qualname, file_path=self.rel_path,
            start_line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
            code=self._snippet(node),
            docstring=ast.get_docstring(node) or "",
        ))
        self.edges.append(Edge(
            src=self.scope_id_stack[-1], dst=node_id, edge_type="contains",
        ))
        for call_name in _ast_collect_calls(node):
            self.edges.append(Edge(src=node_id, dst=call_name, edge_type="calls"))

        self.scope_stack.append(node.name)
        self.scope_id_stack.append(node_id)
        self.generic_visit(node)
        self.scope_stack.pop()
        self.scope_id_stack.pop()

    def _snippet(self, node: ast.AST) -> str:
        start = max(getattr(node, "lineno", 1) - 1, 0)
        end = getattr(node, "end_lineno", start + 1)
        return "\n".join(self.lines[start:end])

    def _mk_qualname(self, name: str) -> str:
        return ".".join(self.scope_stack + [name])


def _ast_collect_calls(node: ast.AST) -> List[str]:
    names: List[str] = []

    class _CV(ast.NodeVisitor):
        def visit_Call(self, call_node: ast.Call) -> None:
            n = _ast_name_from_expr(call_node.func)
            if n:
                names.append(n)
            self.generic_visit(call_node)

    _CV().visit(node)
    return names


def _ast_name_from_expr(expr: ast.AST) -> Optional[str]:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        parts: List[str] = []
        current: ast.AST = expr
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts)) if parts else None
    if isinstance(expr, ast.Call):
        return _ast_name_from_expr(expr.func)
    return None

