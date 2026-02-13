"""Typer-based CLI for CodeGraph local code intelligence."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from . import __version__, config
from .cli_chat import chat_app
from .cli_setup import setup as setup_wizard, set_llm, unset_llm, show_llm
from .cli_v2 import v2_app
from .graph_export import export_dot, export_html
from .orchestrator import MCPOrchestrator
from .storage import GraphStore, ProjectManager

app = typer.Typer(
    help="🧠 CodeGraph CLI — AI-powered code intelligence & multi-agent assistant.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Register v2 commands
app.add_typer(v2_app, name="v2")

# Register chat commands
app.add_typer(chat_app, name="chat")

# Register setup wizard as direct command
app.command("setup")(setup_wizard)

# Register LLM management commands
app.command("set-llm")(set_llm)
app.command("unset-llm")(unset_llm)
app.command("show-llm")(show_llm)


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
    )
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


@app.command("index")
def index_project(
    project_path: Path = typer.Argument(..., exists=True, file_okay=False, help="Path to source project."),
    project_name: Optional[str] = typer.Option(None, "--name", "-n", help="Explicit memory name for project."),
    llm_model: str = typer.Option("qwen2.5-coder:7b", help="Local LLM model name for reasoning operations."),
    llm_provider: str = typer.Option("ollama", help="LLM provider: ollama, groq, openai, anthropic."),
    llm_api_key: Optional[str] = typer.Option(None, help="API key for cloud LLM providers."),
):
    """Parse and index a project into local semantic memory."""
    from datetime import datetime
    
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
    stats = orchestrator.index(resolved_path)
    
    # Store project metadata including source path
    store.set_metadata({
        **store.get_metadata(),
        "project_name": name,
        "source_path": str(resolved_path),
        "indexed_at": datetime.now().isoformat()
    })
    
    pm.set_current_project(name)
    store.close()

    typer.echo(f"Indexed '{resolved_path}' as project '{name}'.")
    typer.echo(f"Nodes: {stats['nodes']} | Edges: {stats['edges']}")


@app.command("list-projects")
def list_projects():
    """List all persisted project memories."""
    pm = ProjectManager()
    projects = pm.list_projects()
    current = pm.get_current_project()

    if not projects:
        typer.echo("No projects indexed yet.")
        raise typer.Exit(code=0)

    for p in projects:
        marker = "*" if p == current else " "
        typer.echo(f"{marker} {p}")


@app.command("load-project")
def load_project(project_name: str = typer.Argument(..., help="Name of project memory to load.")):
    """Switch active project memory."""
    pm = ProjectManager()
    if project_name not in pm.list_projects():
        raise typer.BadParameter(f"Project '{project_name}' not found.")
    pm.set_current_project(project_name)
    typer.echo(f"Loaded project '{project_name}'.")


@app.command("unload-project")
def unload_project():
    """Unload active project memory without deleting data."""
    pm = ProjectManager()
    pm.unload_project()
    typer.echo("Unloaded active project.")


@app.command("delete-project")
def delete_project(project_name: str = typer.Argument(..., help="Project memory to delete.")):
    """Delete persisted project memory."""
    pm = ProjectManager()
    deleted = pm.delete_project(project_name)
    if not deleted:
        raise typer.BadParameter(f"Project '{project_name}' not found.")
    if pm.get_current_project() == project_name:
        pm.unload_project()
    typer.echo(f"Deleted project '{project_name}'.")


@app.command("merge-projects")
def merge_projects(
    source_project: str = typer.Argument(..., help="Project to merge from."),
    target_project: str = typer.Argument(..., help="Project to merge into."),
):
    """Merge one project memory into another."""
    pm = ProjectManager()
    if source_project not in pm.list_projects() or target_project not in pm.list_projects():
        raise typer.BadParameter("Both source and target projects must exist.")

    source_store = GraphStore(pm.project_dir(source_project))
    target_store = GraphStore(pm.project_dir(target_project))
    target_store.merge_from(source_store, source_project)
    source_store.close()
    target_store.close()

    typer.echo(f"Merged '{source_project}' into '{target_project}'.")


@app.command("search")
def search(
    query: str = typer.Argument(..., help="Semantic query for code discovery."),
    top_k: int = typer.Option(5, min=1, max=30, help="Maximum number of matches."),
):
    """Run semantic search across currently loaded project memory."""
    pm = ProjectManager()
    store = _open_current_store(pm)
    orchestrator = MCPOrchestrator(store)
    results = orchestrator.search(query, top_k=top_k)

    if not results:
        typer.echo("No semantic matches found.")
        store.close()
        raise typer.Exit(code=0)

    for item in results:
        typer.echo(f"[{item.node_type}] {item.qualname}  score={item.score:.3f}")
        typer.echo(f"  {item.file_path}:{item.start_line}-{item.end_line}")
        snippet = item.snippet.strip().splitlines()
        if snippet:
            typer.echo(f"  {snippet[0][:120]}")

    store.close()


@app.command("impact")
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
    """Run multi-hop impact analysis using graph + RAG + local LLM."""
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


@app.command("graph")
def graph(
    symbol: str = typer.Argument(..., help="Function/class/module symbol to inspect."),
    depth: int = typer.Option(2, min=1, max=6, help="Traversal depth."),
):
    """Show lightweight ASCII dependency graph around a symbol."""
    pm = ProjectManager()
    store = _open_current_store(pm)
    orchestrator = MCPOrchestrator(store)
    text = orchestrator.graph(symbol, depth=depth)
    typer.echo(text)
    store.close()


@app.command("export-graph")
def export_graph(
    symbol: str = typer.Argument("", help="Optional focus symbol to export local subgraph."),
    fmt: str = typer.Option("html", "--format", "-f", help="Export format: html or dot."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path."),
):
    """Export graph to standalone HTML or Graphviz DOT."""
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


@app.command("current-project")
def current_project():
    """Print active project memory name."""
    pm = ProjectManager()
    current = pm.get_current_project()
    typer.echo(current or "No project loaded")


@app.command("rag-context")
def rag_context(
    query: str = typer.Argument(..., help="Query to retrieve code context without analysis."),
    top_k: int = typer.Option(6, min=1, max=30, help="Number of snippets to fetch."),
):
    """Retrieve top semantic snippets to inspect RAG context directly."""
    pm = ProjectManager()
    store = _open_current_store(pm)
    orchestrator = MCPOrchestrator(store)
    typer.echo(orchestrator.rag_context(query, top_k=top_k))
    store.close()


if __name__ == "__main__":
    app()
