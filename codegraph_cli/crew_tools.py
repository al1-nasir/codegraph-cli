"""CrewAI tool wrappers for CodeGraph project context — full capabilities."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

try:
    from crewai.tools import BaseTool
    CREWAI_AVAILABLE = True
except ImportError:
    # Provide a dummy base class so the module can still be imported
    class BaseTool:  # type: ignore
        def __init_subclass__(cls, **kwargs): pass
        def __init__(self, **kwargs): pass
    CREWAI_AVAILABLE = False
from pydantic import BaseModel, Field, PrivateAttr

if TYPE_CHECKING:
    from .project_context import ProjectContext
    from .rag import RAGRetriever


# ═══════════════════════════════════════════════════════════════
# Backup / Rollback infrastructure
# ═══════════════════════════════════════════════════════════════

BACKUP_DIR = Path.home() / ".codegraph" / "backups"


def _backup_file(source_path: Path, rel_path: str, tag: str = "") -> str:
    """Create a timestamped backup of a file before modification.

    Returns:
        Backup ID (timestamp-based) for rollback.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_id = f"{ts}_{tag}" if tag else ts
    dest_dir = BACKUP_DIR / backup_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    src = source_path / rel_path
    if src.exists():
        dest = dest_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    # Write metadata
    meta = dest_dir / ".backup_meta.json"
    existing: list = []
    if meta.exists():
        existing = json.loads(meta.read_text())
    existing.append({"file": rel_path, "timestamp": ts, "source": str(source_path)})
    meta.write_text(json.dumps(existing, indent=2))

    return backup_id


def rollback_file(backup_id: str, rel_path: str, target_root: Path) -> bool:
    """Restore a file from a backup.

    Returns True if successfully restored.
    """
    backup_dir = BACKUP_DIR / backup_id
    src = backup_dir / rel_path
    if not src.exists():
        return False
    dest = target_root / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def list_backups(limit: int = 20) -> List[Dict]:
    """Return recent backups."""
    if not BACKUP_DIR.exists():
        return []
    results = []
    for d in sorted(BACKUP_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        meta_file = d / ".backup_meta.json"
        files: list = []
        if meta_file.exists():
            files = json.loads(meta_file.read_text())
        results.append({"id": d.name, "files": files})
        if len(results) >= limit:
            break
    return results


# ═══════════════════════════════════════════════════════════════
# Tool input schemas (Pydantic v2)
# ═══════════════════════════════════════════════════════════════

class PathInput(BaseModel):
    path: str = Field(default=".", description="Relative path inside the project")


class FilePathInput(BaseModel):
    path: str = Field(..., description="Relative file path inside the project")


class WriteFileInput(BaseModel):
    path: str = Field(..., description="Relative file path to write")
    content: str = Field(..., description="Full file content to write")


class PatchFileInput(BaseModel):
    path: str = Field(..., description="Relative file path to patch")
    old_text: str = Field(..., description="Exact text to find in the file")
    new_text: str = Field(..., description="Replacement text")


class QueryInput(BaseModel):
    query: str = Field(..., description="Natural language search query")


class GrepInput(BaseModel):
    pattern: str = Field(..., description="Text or pattern to search for in files")
    path: str = Field(default=".", description="Directory to search in (relative)")


class RollbackInput(BaseModel):
    backup_id: str = Field(..., description="Backup ID to restore from")
    path: str = Field(..., description="Relative file path to restore")


# ═══════════════════════════════════════════════════════════════
# Base tool with context
# ═══════════════════════════════════════════════════════════════

class ContextTool(BaseTool):
    """Base class that holds a ProjectContext."""
    _context: Any = PrivateAttr()

    def __init__(self, context: Any, **kwargs):
        super().__init__(**kwargs)
        self._context = context


# ═══════════════════════════════════════════════════════════════
# FILE OPERATIONS TOOLS
# ═══════════════════════════════════════════════════════════════

class ListDirectoryTool(ContextTool):
    name: str = "list_directory"
    description: str = (
        "List files and directories in the project. "
        "Input: {\"path\": \".\"} (relative path, default root). "
        "Returns directory tree with file sizes."
    )
    args_schema: Type[BaseModel] = PathInput

    def _run(self, path: str = ".") -> str:
        try:
            items = self._context.list_directory(path)
            if not items:
                return f"Directory '{path}' is empty or does not exist."

            dirs = [i for i in items if i["type"] == "dir"]
            files = [i for i in items if i["type"] == "file"]
            lines = [f"📂 {path}/  ({len(dirs)} dirs, {len(files)} files)"]

            for d in sorted(dirs, key=lambda x: x["name"]):
                lines.append(f"  📁 {d['name']}/")
            for f in sorted(files, key=lambda x: x["name"]):
                kb = f["size"] / 1024 if f["size"] else 0
                lines.append(f"  📄 {f['name']}  ({kb:.1f} KB)")
            return "\n".join(lines)
        except Exception as e:
            return f"Error listing directory: {e}"


class ReadFileTool(ContextTool):
    name: str = "read_file"
    description: str = (
        "Read the full contents of a file. "
        "Input: {\"path\": \"relative/file.py\"}. "
        "Returns file content with line numbers."
    )
    args_schema: Type[BaseModel] = FilePathInput

    def _run(self, path: str) -> str:
        try:
            content = self._context.read_file(path)
            # Add line numbers for easier reference
            numbered = []
            for i, line in enumerate(content.split("\n"), 1):
                numbered.append(f"{i:4d} │ {line}")
            return f"── {path} ──\n" + "\n".join(numbered)
        except Exception as e:
            return f"Error reading file: {e}"


class WriteFileTool(ContextTool):
    name: str = "write_file"
    description: str = (
        "Write/create a file in the project (creates backup first). "
        "Input: {\"path\": \"relative/file.py\", \"content\": \"file content\"}. "
        "Use this for creating new files or completely rewriting existing ones."
    )
    args_schema: Type[BaseModel] = WriteFileInput

    def _run(self, path: str, content: str) -> str:
        try:
            ctx = self._context
            # Backup existing file
            backup_id = ""
            if ctx.has_source_access and ctx.file_exists(path):
                backup_id = _backup_file(ctx.source_path, path, tag="write")

            ctx.write_file(path, content, create_dirs=True)
            msg = f"✅ Wrote {path} ({len(content)} chars)"
            if backup_id:
                msg += f"\n   Backup: {backup_id} (use rollback_file to undo)"
            return msg
        except Exception as e:
            return f"Error writing file: {e}"


class PatchFileTool(ContextTool):
    name: str = "patch_file"
    description: str = (
        "Modify a specific part of a file by replacing exact text (creates backup first). "
        "Input: {\"path\": \"file.py\", \"old_text\": \"text to find\", \"new_text\": \"replacement\"}. "
        "Use this for targeted edits instead of rewriting the whole file."
    )
    args_schema: Type[BaseModel] = PatchFileInput

    def _run(self, path: str, old_text: str, new_text: str) -> str:
        try:
            ctx = self._context
            content = ctx.read_file(path)

            if old_text not in content:
                return f"❌ Could not find the specified text in {path}. Read the file first to get the exact text."

            count = content.count(old_text)
            # Backup
            backup_id = _backup_file(ctx.source_path, path, tag="patch")

            new_content = content.replace(old_text, new_text, 1)
            ctx.write_file(path, new_content)

            return (
                f"✅ Patched {path} ({count} occurrence(s) found, replaced first)\n"
                f"   Backup: {backup_id}"
            )
        except Exception as e:
            return f"Error patching file: {e}"


class DeleteFileTool(ContextTool):
    name: str = "delete_file"
    description: str = (
        "Delete a file from the project (creates backup first). "
        "Input: {\"path\": \"relative/file.py\"}."
    )
    args_schema: Type[BaseModel] = FilePathInput

    def _run(self, path: str) -> str:
        try:
            ctx = self._context
            if not ctx.file_exists(path):
                return f"❌ File not found: {path}"

            # Backup first
            backup_id = _backup_file(ctx.source_path, path, tag="delete")
            full_path = ctx.source_path / path
            full_path.unlink()

            # Remove stale nodes from the code graph
            ctx.remove_from_index(path)

            return f"✅ Deleted {path}\n   Backup: {backup_id} (use rollback_file to restore)"
        except Exception as e:
            return f"Error deleting file: {e}"


class RollbackFileTool(ContextTool):
    name: str = "rollback_file"
    description: str = (
        "Restore a file from a previous backup. "
        "Input: {\"backup_id\": \"20260213_143000_write\", \"path\": \"file.py\"}."
    )
    args_schema: Type[BaseModel] = RollbackInput

    def _run(self, backup_id: str, path: str) -> str:
        try:
            ctx = self._context
            if not ctx.has_source_access:
                return "❌ No source access for rollback."
            ok = rollback_file(backup_id, path, ctx.source_path)
            if ok:
                return f"✅ Restored {path} from backup {backup_id}"
            return f"❌ Backup file not found: {backup_id}/{path}"
        except Exception as e:
            return f"Error during rollback: {e}"


class ListBackupsTool(ContextTool):
    name: str = "list_backups"
    description: str = "List recent file backups for rollback. No input required."

    def _run(self, **kwargs) -> str:
        backups = list_backups(limit=10)
        if not backups:
            return "No backups found."
        lines = ["Recent backups:"]
        for b in backups:
            files_str = ", ".join(f["file"] for f in b["files"]) if b["files"] else "(unknown)"
            lines.append(f"  🔖 {b['id']}  →  {files_str}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# CODE ANALYSIS TOOLS
# ═══════════════════════════════════════════════════════════════

class SearchCodeTool(BaseTool):
    name: str = "search_code"
    description: str = (
        "Semantic search across indexed code using RAG. "
        "Input: {\"query\": \"authentication login logic\"}. "
        "Returns matching functions, classes, and code snippets."
    )
    args_schema: Type[BaseModel] = QueryInput
    _rag: Any = PrivateAttr()

    def __init__(self, rag_retriever: Any, **kwargs):
        super().__init__(**kwargs)
        self._rag = rag_retriever

    def _run(self, query: str) -> str:
        try:
            results = self._rag.search(query, top_k=8)
            if not results:
                return f"No results found for: {query}"
            lines = [f"Found {len(results)} results for '{query}':\n"]
            for i, r in enumerate(results, 1):
                lines.append(
                    f"{i}. [{r.node_type}] {r.qualname}\n"
                    f"   File: {r.file_path}:{r.start_line}-{r.end_line}\n"
                    f"   Score: {r.score:.2f}\n"
                    f"   Preview: {r.snippet[:120]}…\n"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"Error searching: {e}"


class GrepTool(ContextTool):
    name: str = "grep_in_project"
    description: str = (
        "Search for exact text across project files (like grep). "
        "Input: {\"pattern\": \"def my_function\", \"path\": \".\"}. "
        "Returns matching lines with file paths and line numbers."
    )
    args_schema: Type[BaseModel] = GrepInput

    def _run(self, pattern: str, path: str = ".") -> str:
        try:
            ctx = self._context
            if not ctx.has_source_access:
                return "❌ No source access."

            root = ctx.source_path / path
            matches = []
            max_results = 30

            for fpath in root.rglob("*"):
                if not fpath.is_file():
                    continue
                if fpath.suffix not in (".py", ".js", ".ts", ".json", ".toml", ".yaml", ".yml", ".md", ".txt", ".html", ".css", ".sh"):
                    continue
                # Skip hidden dirs, __pycache__, node_modules, .git
                parts = fpath.relative_to(ctx.source_path).parts
                if any(p.startswith(".") or p in ("__pycache__", "node_modules", ".git", "htmlcov") for p in parts):
                    continue
                try:
                    text = fpath.read_text(encoding="utf-8", errors="ignore")
                    for line_no, line in enumerate(text.split("\n"), 1):
                        if pattern.lower() in line.lower():
                            rel = str(fpath.relative_to(ctx.source_path))
                            matches.append(f"  {rel}:{line_no}  {line.strip()}")
                            if len(matches) >= max_results:
                                break
                except Exception:
                    continue
                if len(matches) >= max_results:
                    break

            if not matches:
                return f"No matches for '{pattern}'"
            return f"Found {len(matches)} match(es) for '{pattern}':\n" + "\n".join(matches)
        except Exception as e:
            return f"Error: {e}"


class ProjectSummaryTool(ContextTool):
    name: str = "get_project_summary"
    description: str = (
        "Get project statistics: file count, node/edge count, structure. "
        "No input required."
    )

    def _run(self, **kwargs) -> str:
        try:
            s = self._context.get_project_summary()
            lines = [
                f"Project: {s['project_name']}",
                f"Source:  {s.get('source_path', 'N/A')}",
                f"Indexed: {s['indexed_files']} files, {s['total_nodes']} symbols, {s['total_edges']} edges",
            ]
            if s.get("node_types"):
                lines.append("Symbol types:")
                for ntype, count in sorted(s["node_types"].items()):
                    lines.append(f"  {ntype}: {count}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"


class FileTreeTool(ContextTool):
    name: str = "file_tree"
    description: str = (
        "Show the full project directory tree (recursive). "
        "Input: {\"path\": \".\"} (optional, defaults to root). "
        "Useful for understanding project structure quickly."
    )
    args_schema: Type[BaseModel] = PathInput

    def _run(self, path: str = ".") -> str:
        try:
            ctx = self._context
            if not ctx.has_source_access:
                return "❌ No source access."

            root = ctx.source_path / path
            lines = [f"📂 {path}/"]
            self._walk(root, ctx.source_path, lines, prefix="  ", depth=0, max_depth=4)
            if len(lines) > 100:
                lines = lines[:100]
                lines.append("  ... (truncated)")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def _walk(current: Path, root: Path, lines: list, prefix: str, depth: int, max_depth: int):
        if depth >= max_depth:
            return
        try:
            entries = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return

        for entry in entries:
            name = entry.name
            if name.startswith(".") or name in ("__pycache__", "node_modules", ".git", "htmlcov"):
                continue
            if name.endswith(".egg-info"):
                continue

            if entry.is_dir():
                lines.append(f"{prefix}📁 {name}/")
                FileTreeTool._walk(entry, root, lines, prefix + "  ", depth + 1, max_depth)
            else:
                lines.append(f"{prefix}📄 {name}")


# ═══════════════════════════════════════════════════════════════
# TOOL FACTORY
# ═══════════════════════════════════════════════════════════════

def create_tools(context: Any, rag_retriever: Any) -> dict:
    """Create all tools for CrewAI agents.

    Returns dict with categorized tool lists:
        file_ops, code_analysis, all
    """
    # File operations
    list_dir = ListDirectoryTool(context=context)
    read_file = ReadFileTool(context=context)
    write_file = WriteFileTool(context=context)
    patch_file = PatchFileTool(context=context)
    delete_file = DeleteFileTool(context=context)
    rollback = RollbackFileTool(context=context)
    list_bkp = ListBackupsTool(context=context)
    file_tree = FileTreeTool(context=context)

    # Code analysis
    search_code = SearchCodeTool(rag_retriever=rag_retriever)
    grep = GrepTool(context=context)
    summary = ProjectSummaryTool(context=context)

    file_ops = [list_dir, read_file, write_file, patch_file, delete_file, rollback, list_bkp, file_tree]
    code_analysis = [search_code, grep, summary]

    return {
        "file_ops": file_ops,
        "code_analysis": code_analysis,
        "all": file_ops + code_analysis,
    }
