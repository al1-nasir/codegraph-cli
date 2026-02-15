"""Typer-based CLI for CodeGraph local code intelligence."""

from __future__ import annotations

from difflib import get_close_matches
from pathlib import Path
from typing import Dict, Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from . import __version__, config
from .cli_chat import chat_app
from .cli_explore import explore_app
from .cli_export import export_app
from .cli_groups import analyze_grp, config_grp, project_grp
from .cli_health import health_app
from .cli_quickstart import quickstart_app
from .cli_setup import setup as setup_wizard, set_llm, unset_llm, show_llm
from .cli_setup import set_embedding, unset_embedding, show_embedding
from .cli_suggestions import show_next_steps
from .cli_watch import watch_app
from .graph_export import export_dot, export_html
from .orchestrator import MCPOrchestrator
from .storage import GraphStore, ProjectManager

console = Console()

app = typer.Typer(
    help="🧠 CodeGraph CLI — AI-powered code intelligence & multi-agent assistant.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=False,
)

# ── Top-level groups ─────────────────────────────────────────────
app.add_typer(config_grp, name="config", help="Configuration management")
app.add_typer(project_grp, name="project", help="Project management")
app.add_typer(analyze_grp, name="analyze", help="Code analysis")
app.add_typer(chat_app, name="chat", help="Interactive chat with AI agents")
app.add_typer(explore_app, name="explore", help="Visual code explorer in browser")
app.add_typer(export_app, name="export", help="Export project documentation")


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        typer.echo(f"CodeGraph CLI v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
):
    """CodeGraph CLI: Local-first code intelligence with AI-powered impact analysis."""
    pass


def _project_name_from_path(project_path: Path) -> str:
    return project_path.resolve().name.replace(" ", "_")


def _open_current_store(pm: ProjectManager) -> GraphStore:
    project = pm.get_current_project()
    if not project:
        raise typer.BadParameter("No project loaded. Use 'cg load-project <name>' or run 'cg index <path>'.")
    project_dir = pm.project_dir(project)
    if not project_dir.exists():
        raise typer.BadParameter(f"Loaded project '{project}' does not exist in memory.")
    return GraphStore(project_dir)


def index_project(
    project_path: Path = typer.Argument(..., exists=True, file_okay=False, help="Path to source project."),
    project_name: Optional[str] = typer.Option(None, "--name", "-n", help="Explicit memory name for project."),
    llm_model: str = typer.Option("qwen2.5-coder:7b", help="Local LLM model name for reasoning operations."),
    llm_provider: str = typer.Option("ollama", help="LLM provider: ollama, groq, openai, anthropic."),
    llm_api_key: Optional[str] = typer.Option(None, help="API key for cloud LLM providers."),
):
    """📦 Parse and index a project into local semantic memory.

    Example:
      cg index ./my-project
      cg index ./backend --name my-api
    """
    from datetime import datetime
    import time as _time

    pm = ProjectManager()
    resolved_path = project_path.resolve()
    name = project_name or _project_name_from_path(resolved_path)
    project_dir = pm.create_or_get_project(name)

    store = GraphStore(project_dir)
    orchestrator = MCPOrchestrator(
        store,
        llm_model=llm_model,
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
    )

    start_t = _time.time()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Indexing project...", total=None)
        stats = orchestrator.index(resolved_path)
        progress.update(task, completed=100, total=100, description="[green]Indexing complete")
    elapsed = _time.time() - start_t

    # Store project metadata including source path and embedding info
    emb_model_key = getattr(orchestrator.embedding_model, 'model_key', 'hash')
    emb_dim = getattr(orchestrator.embedding_model, 'dim', 256)
    store.set_metadata({
        **store.get_metadata(),
        "project_name": name,
        "source_path": str(resolved_path),
        "indexed_at": datetime.now().isoformat(),
        "embedding_model": emb_model_key,
        "embedding_dim": emb_dim,
    })

    pm.set_current_project(name)
    store.close()

    console.print(f"[green]✓[/green] Indexed [bold]'{resolved_path}'[/bold] as project [bold]'{name}'[/bold] in {elapsed:.1f}s")
    console.print(f"  Nodes: {stats['nodes']} | Edges: {stats['edges']}")
    show_next_steps("index")


def list_projects():
    """📋 List all persisted project memories.

    Example:
      cg list-projects
    """
    pm = ProjectManager()
    projects = pm.list_projects()
    current = pm.get_current_project()

    if not projects:
        typer.echo("No projects indexed yet.")
        raise typer.Exit(code=0)

    for p in projects:
        marker = "*" if p == current else " "
        typer.echo(f"{marker} {p}")


def load_project(project_name: str = typer.Argument(..., help="Name of project memory to load.")):
    """🔄 Switch active project memory.

    Example:
      cg load-project my-api
    """
    pm = ProjectManager()
    if project_name not in pm.list_projects():
        raise typer.BadParameter(f"Project '{project_name}' not found.")
    pm.set_current_project(project_name)
    typer.echo(f"Loaded project '{project_name}'.")


def unload_project():
    """Unload active project memory without deleting data."""
    pm = ProjectManager()
    pm.unload_project()
    typer.echo("Unloaded active project.")


def delete_project(project_name: str = typer.Argument(..., help="Project memory to delete.")):
    """🗑️  Delete persisted project memory.

    Example:
      cg delete-project old-project
    """
    pm = ProjectManager()
    deleted = pm.delete_project(project_name)
    if not deleted:
        raise typer.BadParameter(f"Project '{project_name}' not found.")
    if pm.get_current_project() == project_name:
        pm.unload_project()
    typer.echo(f"Deleted project '{project_name}'.")


def merge_projects(
    source_project: str = typer.Argument(..., help="Project to merge from."),
    target_project: str = typer.Argument(..., help="Project to merge into."),
):
    """🔀 Merge one project memory into another.

    Example:
      cg merge-projects frontend backend
    """
    pm = ProjectManager()
    if source_project not in pm.list_projects() or target_project not in pm.list_projects():
        raise typer.BadParameter("Both source and target projects must exist.")

    source_store = GraphStore(pm.project_dir(source_project))
    target_store = GraphStore(pm.project_dir(target_project))
    target_store.merge_from(source_store, source_project)
    source_store.close()
    target_store.close()

    typer.echo(f"Merged '{source_project}' into '{target_project}'.")


def search(
    query: str = typer.Argument(..., help="Semantic query for code discovery."),
    top_k: int = typer.Option(5, min=1, max=30, help="Maximum number of matches."),
):
    """🔍 Semantic search across your codebase (alias: find).

    Example:
      cg search "database migration logic"
      cg search "JWT token validation"
      cg search "authentication" --top-k 10
    """
    pm = ProjectManager()
    store = _open_current_store(pm)
    orchestrator = MCPOrchestrator(store)

    with console.status("[cyan]Searching...[/cyan]"):
        results = orchestrator.search(query, top_k=top_k)

    if not results:
        console.print("[yellow]No semantic matches found.[/yellow]")
        store.close()
        raise typer.Exit(code=0)

    for item in results:
        typer.echo(f"[{item.node_type}] {item.qualname}  score={item.score:.3f}")
        typer.echo(f"  {item.file_path}:{item.start_line}-{item.end_line}")
        snippet = item.snippet.strip().splitlines()
        if snippet:
            typer.echo(f"  {snippet[0][:120]}")

    store.close()
    show_next_steps("search")


def impact(
    symbol: str = typer.Argument(..., help="Function/class/module symbol to analyze."),
    hops: int = typer.Option(2, min=1, max=6, help="Dependency traversal depth."),
    show_graph: bool = typer.Option(True, "--show-graph/--no-graph", help="Include ASCII graph output."),
    llm_provider: str = typer.Option(
        config.LLM_PROVIDER,
        help="LLM provider: ollama, groq, openai, anthropic, gemini, openrouter.",
    ),
    llm_api_key: Optional[str] = typer.Option(
        config.LLM_API_KEY or None,
        help="API key for cloud LLM providers.",
    ),
    llm_model: str = typer.Option(
        config.LLM_MODEL,
        help="LLM model name.",
    ),
):
    """📊 Multi-hop impact analysis using graph + RAG + local LLM.

    Example:
      cg impact "UserService.authenticate"
      cg impact "process_payment" --hops 3
    """
    pm = ProjectManager()
    store = _open_current_store(pm)
    orchestrator = MCPOrchestrator(
        store,
        llm_model=llm_model,
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
    )

    report = orchestrator.impact(symbol, hops=hops)
    
    # Check if symbol was found
    if "not found" in report.explanation.lower() and not report.impacted:
        typer.echo(f"❌ Symbol '{symbol}' not found in current project.", err=True)
        
        # Try to find similar symbols
        search_results = orchestrator.search(symbol, top_k=3)
        if search_results:
            typer.echo(f"\n💡 Did you mean one of these?", err=True)
            for result in search_results:
                typer.echo(f"   - {result.qualname} ({result.node_type})", err=True)
            typer.echo(f"\n💡 Tip: Use 'cg search {symbol}' to find similar symbols", err=True)
        store.close()
        raise typer.Exit(code=1)
    
    typer.echo(f"Root: {report.root}")
    if report.impacted:
        typer.echo("Impacted symbols:")
        for impacted in report.impacted:
            typer.echo(f"- {impacted}")
    else:
        typer.echo("Impacted symbols: none found")

    if show_graph:
        typer.echo("\nASCII graph:")
        typer.echo(report.ascii_graph)

    typer.echo("\nExplanation:")
    typer.echo(report.explanation)
    store.close()


def graph(
    symbol: str = typer.Argument(..., help="Function/class/module symbol to inspect."),
    depth: int = typer.Option(2, min=1, max=6, help="Traversal depth."),
):
    """🕸️  Show lightweight ASCII dependency graph around a symbol.

    Example:
      cg graph "UserService"
      cg graph "main" --depth 4
    """
    pm = ProjectManager()
    store = _open_current_store(pm)
    orchestrator = MCPOrchestrator(store)
    text = orchestrator.graph(symbol, depth=depth)
    typer.echo(text)
    store.close()


def export_graph(
    symbol: str = typer.Argument("", help="Optional focus symbol to export local subgraph."),
    fmt: str = typer.Option("html", "--format", "-f", help="Export format: html or dot."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path."),
):
    """📤 Export graph to standalone HTML or Graphviz DOT.

    Example:
      cg export-graph
      cg export-graph UserService --format dot
      cg export-graph --output my_graph.html
    """
    fmt = fmt.lower()
    if fmt not in {"html", "dot"}:
        raise typer.BadParameter("Format must be one of: html, dot")

    pm = ProjectManager()
    store = _open_current_store(pm)
    current = pm.get_current_project() or "project"

    if output is None:
        output = Path.cwd() / f"{current}_graph.{fmt}"

    if fmt == "html":
        export_html(store, output, focus=symbol)
    else:
        export_dot(store, output, focus=symbol)

    typer.echo(f"Exported graph to {output}")
    store.close()


def current_project():
    """📌 Print active project memory name.

    Example:
      cg current-project
    """
    pm = ProjectManager()
    current = pm.get_current_project()
    typer.echo(current or "No project loaded")


def rag_context(
    query: str = typer.Argument(..., help="Query to retrieve code context without analysis."),
    top_k: int = typer.Option(6, min=1, max=30, help="Number of snippets to fetch."),
):
    """📄 Retrieve top semantic snippets to inspect RAG context directly.

    Example:
      cg rag-context "authentication flow"
      cg rag-context "database models" --top-k 10
    """
    pm = ProjectManager()
    store = _open_current_store(pm)
    orchestrator = MCPOrchestrator(store)
    typer.echo(orchestrator.rag_context(query, top_k=top_k))
    store.close()


# ===================================================================
# Tree command - Show project structure
# ===================================================================

def _build_tree_structure(store: "GraphStore", full: bool = False) -> Dict:
    """Build a nested tree structure from indexed nodes.
    
    Args:
        store: GraphStore instance with indexed nodes
        full: If True, include functions/classes breakdown
        
    Returns:
        Nested dict representing the tree structure
    """
    from collections import defaultdict
    
    # Get all nodes grouped by file
    nodes_by_file = store.all_by_file()
    
    # Build tree structure
    tree: Dict = defaultdict(lambda: {"type": "directory", "children": {}, "functions": [], "classes": []})
    
    for file_path, nodes in nodes_by_file.items():
        # Split file path into parts
        parts = Path(file_path).parts
        current = tree
        
        # Navigate/create directory structure
        for part in parts[:-1]:
            if part not in current:
                current[part] = {"type": "directory", "children": {}, "functions": [], "classes": []}
            current = current[part]["children"]
        
        # Add file node
        filename = parts[-1] if parts else file_path
        file_node = {
            "type": "file",
            "path": file_path,
            "functions": [],
            "classes": [],
            "other": [],
        }
        
        if full:
            # Categorize nodes by type
            for node in nodes:
                node_type = node.get("node_type", "unknown")
                name = node.get("name", "unknown")
                qualname = node.get("qualname", name)
                start_line = node.get("start_line", 0)
                end_line = node.get("end_line", 0)
                
                node_info = {
                    "name": name,
                    "qualname": qualname,
                    "lines": f"L{start_line}-{end_line}",
                }
                
                if node_type == "function":
                    file_node["functions"].append(node_info)
                elif node_type == "class":
                    file_node["classes"].append(node_info)
                else:
                    file_node["other"].append(node_info)
        
        current[filename] = file_node
    
    return dict(tree)


def _render_tree(tree: Dict, prefix: str = "", full: bool = False, is_last: bool = True) -> str:
    """Render tree structure as ASCII art.
    
    Args:
        tree: Nested dict representing the tree
        prefix: Current line prefix for indentation
        full: If True, show functions/classes breakdown
        is_last: Whether this is the last item at current level
        
    Returns:
        ASCII tree string
    """
    lines = []
    entries = sorted(tree.keys())
    
    for i, name in enumerate(entries):
        node = tree[name]
        is_last_item = (i == len(entries) - 1)
        
        # Choose connector
        connector = "    " if is_last_item else "   |"
        child_prefix = prefix + connector
        
        # Determine node type indicator
        if isinstance(node, dict):
            node_type = node.get("type", "unknown")
            
            if node_type == "directory":
                # Directory - show with trailing slash
                lines.append(f"{prefix}{'`-- ' if is_last_item else '|-- '}{name}/")
                # Recursively render children
                children = node.get("children", {})
                if children:
                    lines.append(_render_tree(children, child_prefix, full, is_last_item))
                    
            elif node_type == "file":
                # File - show filename
                lines.append(f"{prefix}{'`-- ' if is_last_item else '|-- '}{name}")
                
                if full:
                    # Show functions
                    functions = node.get("functions", [])
                    for j, func in enumerate(functions):
                        is_last_func = (j == len(functions) - 1) and not node.get("classes")
                        func_connector = "    " if is_last_func else "   |"
                        lines.append(f"{child_prefix}{'`-- ' if is_last_func else '|-- '}[fn] {func['name']} ({func['lines']})")
                    
                    # Show classes
                    classes = node.get("classes", [])
                    for j, cls in enumerate(classes):
                        is_last_cls = (j == len(classes) - 1) and not node.get("other")
                        cls_connector = "    " if is_last_cls else "   |"
                        lines.append(f"{child_prefix}{'`-- ' if is_last_cls else '|-- '}[cls] {cls['name']} ({cls['lines']})")
                    
                    # Show other nodes (modules, imports, etc.)
                    other = node.get("other", [])
                    for j, item in enumerate(other):
                        is_last_other = j == len(other) - 1
                        lines.append(f"{child_prefix}{'`-- ' if is_last_other else '|-- '}[{item.get('name', 'unknown')}]")
        else:
            # Simple entry
            lines.append(f"{prefix}{'`-- ' if is_last_item else '|-- '}{name}")
    
    return "\n".join(lines)


def tree_command(
    full: bool = typer.Option(
        False,
        "--full",
        "-f",
        help="Show full breakdown including functions and classes within each file.",
    ),
    path: Optional[Path] = typer.Option(
        None,
        "--path",
        "-p",
        help="Filter tree to show only files under a specific path prefix.",
    ),
):
    """Show the directory tree structure of the currently indexed project.
    
    By default, shows files and directories. Use --full to see functions
    and classes within each file.
    
    Examples:
        cg tree              # Show basic file/directory tree
        cg tree --full       # Show tree with functions/classes breakdown
        cg tree -p src/      # Show only files under src/ directory
    """
    pm = ProjectManager()
    store = _open_current_store(pm)
    
    # Get project metadata for display
    metadata = store.get_metadata()
    project_name = metadata.get("project_name", pm.get_current_project() or "unknown")
    source_path = metadata.get("source_path", "unknown")
    
    typer.echo(f"Project: {project_name}")
    typer.echo(f"Source: {source_path}")
    typer.echo("")
    
    # Build and render tree
    tree = _build_tree_structure(store, full=full)
    
    if not tree:
        typer.echo("No files indexed in this project.")
        store.close()
        raise typer.Exit(code=0)
    
    # Filter by path if specified
    if path:
        path_str = str(path).rstrip("/")
        # Navigate to the specified sub-tree
        parts = Path(path_str).parts
        current = tree
        for part in parts:
            if part in current:
                node = current[part]
                if isinstance(node, dict) and node.get("type") == "directory":
                    current = node.get("children", {})
                else:
                    current = {part: node}
                    break
            else:
                typer.echo(f"Path '{path}' not found in indexed project.")
                store.close()
                raise typer.Exit(code=1)
        tree = current
    
    # Render the tree
    tree_output = _render_tree(tree, full=full)
    typer.echo(tree_output)
    
    # Show summary stats
    nodes = store.get_nodes()
    files = set(n["file_path"] for n in nodes)
    functions = sum(1 for n in nodes if n["node_type"] == "function")
    classes = sum(1 for n in nodes if n["node_type"] == "class")
    
    typer.echo("")
    typer.echo(f"Summary: {len(files)} files, {functions} functions, {classes} classes, {len(nodes)} total nodes")

    # Detect unindexed files and warn
    source = metadata.get("source_path") or metadata.get("project_root")
    if source:
        unindexed = _detect_unindexed_files(Path(source), files)
        if unindexed:
            typer.echo("")
            typer.echo(
                typer.style(
                    f"⚠  {len(unindexed)} file(s) on disk not in index:",
                    fg=typer.colors.YELLOW,
                )
            )
            for f in sorted(unindexed)[:10]:
                typer.echo(f"   + {f}")
            if len(unindexed) > 10:
                typer.echo(f"   … and {len(unindexed) - 10} more")
            typer.echo(
                f"\nRun {typer.style('cg analyze sync', fg=typer.colors.CYAN)} to incrementally index new/changed files."
            )

    store.close()


# ===================================================================
# Helpers: detect out-of-sync files
# ===================================================================

def _detect_unindexed_files(
    source_root: Path, indexed_files: set[str],
) -> list[str]:
    """Return relative paths of parseable files on disk that are NOT in the index."""
    from .parser import LANGUAGE_MAP, SKIP_DIRS

    unindexed: list[str] = []
    if not source_root.is_dir():
        return unindexed

    for ext in LANGUAGE_MAP:
        for fp in sorted(source_root.rglob(f"*{ext}")):
            if any(part in SKIP_DIRS for part in fp.parts):
                continue
            rel = str(fp.relative_to(source_root))
            if rel not in indexed_files:
                unindexed.append(rel)
    return unindexed


# ===================================================================
# Sync command — incremental index of new / changed files
# ===================================================================

def sync_command(
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n",
        help="Only list what would be synced; don't modify the index.",
    ),
):
    """Incrementally sync the index with the source directory.

    Detects new and deleted files relative to the current index and
    updates accordingly — much faster than a full re-index.

    Examples:
        cg analyze sync            # sync new/deleted files
        cg analyze sync --dry-run  # preview changes only
    """
    from .embeddings import get_embedder
    from .parser import LANGUAGE_MAP, SKIP_DIRS

    pm = ProjectManager()
    store = _open_current_store(pm)
    metadata = store.get_metadata()
    source = metadata.get("source_path") or metadata.get("project_root")

    if not source:
        typer.echo("❌ No source path recorded for this project. Re-index with: cg project index <path>")
        store.close()
        raise typer.Exit(1)

    source_root = Path(source)
    if not source_root.is_dir():
        typer.echo(f"❌ Source path no longer exists: {source}")
        store.close()
        raise typer.Exit(1)

    # Gather indexed files
    nodes = store.get_nodes()
    indexed_files = set(n["file_path"] for n in nodes)

    # Gather files on disk
    disk_files: set[str] = set()
    for ext in LANGUAGE_MAP:
        for fp in sorted(source_root.rglob(f"*{ext}")):
            if any(part in SKIP_DIRS for part in fp.parts):
                continue
            disk_files.add(str(fp.relative_to(source_root)))

    new_files = sorted(disk_files - indexed_files)
    deleted_files = sorted(indexed_files - disk_files)

    if not new_files and not deleted_files:
        typer.echo("✅ Index is already in sync — no changes detected.")
        store.close()
        return

    # Report
    if new_files:
        typer.echo(typer.style(f"\n📄 {len(new_files)} new file(s):", bold=True))
        for f in new_files:
            typer.echo(f"   + {f}")

    if deleted_files:
        typer.echo(typer.style(f"\n🗑  {len(deleted_files)} deleted file(s):", bold=True))
        for f in deleted_files:
            typer.echo(f"   - {f}")

    if dry_run:
        typer.echo(f"\n(dry run — no changes made)")
        store.close()
        return

    # Apply changes
    embedder = get_embedder()
    model_key = getattr(embedder, "model_key", "hash")

    added_nodes = 0
    removed_nodes = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        total = len(new_files) + len(deleted_files)
        task = progress.add_task("Syncing...", total=total)

        for f in deleted_files:
            removed_nodes += store.remove_nodes_for_file(f)
            progress.advance(task)

        for f in new_files:
            fp = source_root / f
            added_nodes += store.index_single_file(
                file_path=fp,
                project_root=source_root,
                embedder=embedder,
                model_key=model_key,
            )
            progress.advance(task)

    typer.echo(
        f"\n✅ Sync complete: "
        f"+{added_nodes} nodes ({len(new_files)} files), "
        f"-{removed_nodes} nodes ({len(deleted_files)} files)"
    )
    store.close()


# ===================================================================
# Wire functions into groups (the ONLY way to reach these commands)
# ===================================================================

# Config group: setup, LLM, and embedding management
config_grp.command("setup")(setup_wizard)
config_grp.command("set-llm")(set_llm)
config_grp.command("unset-llm")(unset_llm)
config_grp.command("show-llm")(show_llm)
config_grp.command("set-embedding")(set_embedding)
config_grp.command("unset-embedding")(unset_embedding)
config_grp.command("show-embedding")(show_embedding)

# Project group: index, load, list, delete, merge, current, init, watch
project_grp.command("index")(index_project)
project_grp.command("list")(list_projects)
project_grp.command("load")(load_project)
project_grp.command("unload")(unload_project)
project_grp.command("delete")(delete_project)
project_grp.command("merge")(merge_projects)
project_grp.command("current")(current_project)
project_grp.add_typer(quickstart_app, name="init", help="🚀 Quick-start wizard")
project_grp.add_typer(watch_app, name="watch", help="👀 Auto-reindex on file changes")

# Analyze group: search, impact, graph, export-graph, rag-context, tree, health
analyze_grp.command("search")(search)
analyze_grp.command("impact")(impact)
analyze_grp.command("graph")(graph)
analyze_grp.command("export-graph")(export_graph)
analyze_grp.command("rag-context")(rag_context)
analyze_grp.command("tree")(tree_command)
analyze_grp.command("sync")(sync_command)
analyze_grp.add_typer(health_app, name="health", help="🏥 Project health dashboard")


# ===================================================================
# Fuzzy command matching for typos
# ===================================================================


def _get_all_command_names() -> list[str]:
    """Collect all registered command names including groups."""
    names = []
    for cmd in app.registered_commands:
        if cmd.name:
            names.append(cmd.name)
    for group in app.registered_groups:
        if group.name:
            names.append(group.name)
    return names


def cli_main() -> None:
    """Entry point with fuzzy command matching on unknown commands.

    This wraps the Typer app invocation to catch unknown-command errors
    (exit code 2) and suggest similar commands using difflib.
    """
    import sys

    try:
        app()
    except SystemExit as e:
        if e.code == 2 and len(sys.argv) > 1:
            unknown = sys.argv[1]
            if not unknown.startswith("-"):
                all_commands = _get_all_command_names()
                suggestions = get_close_matches(unknown, all_commands, n=3, cutoff=0.5)
                if suggestions:
                    console.print(f"\n[yellow]Unknown command '[bold]{unknown}[/bold]'. Did you mean:[/yellow]")
                    for s in suggestions:
                        console.print(f"  [cyan]cg {s}[/cyan]")
                    console.print()
        raise


if __name__ == "__main__":
    cli_main()
