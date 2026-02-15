"""Quick-start wizard for CodeGraph CLI — get started in 30 seconds."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import config
from .config import BASE_DIR

console = Console()

quickstart_app = typer.Typer(help="🚀 Quick setup and demo")


def detect_project_type() -> str:
    """Auto-detect project type from common files in the current directory."""
    cwd = Path.cwd()
    detectors = [
        ("package.json", "JavaScript/TypeScript"),
        ("tsconfig.json", "TypeScript"),
        ("setup.py", "Python"),
        ("pyproject.toml", "Python"),
        ("setup.cfg", "Python"),
        ("requirements.txt", "Python"),
        ("go.mod", "Go"),
        ("Cargo.toml", "Rust"),
        ("pom.xml", "Java"),
        ("build.gradle", "Java/Kotlin"),
        ("build.gradle.kts", "Kotlin"),
        ("Gemfile", "Ruby"),
        ("composer.json", "PHP"),
        ("mix.exs", "Elixir"),
        ("CMakeLists.txt", "C/C++"),
        ("Makefile", "C/C++"),
        (".csproj", "C#/.NET"),
        ("pubspec.yaml", "Dart/Flutter"),
        ("Package.swift", "Swift"),
    ]
    for filename, project_type in detectors:
        if (cwd / filename).exists():
            return project_type
        # Also check for glob patterns (e.g. *.csproj)
        if filename.startswith(".") and list(cwd.glob(f"*{filename}")):
            return project_type
    return "Unknown"


def config_exists() -> bool:
    """Check if CodeGraph configuration already exists."""
    config_file = BASE_DIR / "config.toml"
    return config_file.exists()


def has_active_project() -> bool:
    """Check if a project is currently loaded."""
    try:
        from .storage import ProjectManager
        pm = ProjectManager()
        return pm.get_current_project() is not None
    except Exception:
        return False


def run_setup() -> None:
    """Run the interactive setup wizard."""
    try:
        from .cli_setup import setup
        setup()
    except (typer.Exit, SystemExit):
        pass


def run_index(path: str = ".") -> Optional[dict]:
    """Index the current directory and return stats."""
    from .orchestrator import MCPOrchestrator
    from .storage import GraphStore, ProjectManager

    project_path = Path(path).resolve()
    pm = ProjectManager()
    name = project_path.name.replace(" ", "_")
    project_dir = pm.create_or_get_project(name)

    store = GraphStore(project_dir)
    orchestrator = MCPOrchestrator(store)
    stats = orchestrator.index(project_path)

    from datetime import datetime
    store.set_metadata({
        **store.get_metadata(),
        "project_name": name,
        "source_path": str(project_path),
        "indexed_at": datetime.now().isoformat(),
    })
    pm.set_current_project(name)
    store.close()
    return stats


def demo_search_based_on_project_type(project_type: str) -> None:
    """Run a demo search relevant to the detected project type."""
    demo_queries = {
        "Python": "main entry point or application startup",
        "JavaScript/TypeScript": "main entry point or app initialization",
        "TypeScript": "main entry point or app initialization",
        "Go": "main function or server startup",
        "Rust": "main function or entry point",
        "Java": "main class or application entry",
        "Java/Kotlin": "main class or application entry",
        "Kotlin": "main class or application entry",
        "Ruby": "application controller or main module",
        "PHP": "index or main controller",
        "C/C++": "main function",
        "C#/.NET": "program entry point",
    }
    query = demo_queries.get(project_type, "main entry point")

    try:
        from .storage import ProjectManager, GraphStore
        from .orchestrator import MCPOrchestrator

        pm = ProjectManager()
        project = pm.get_current_project()
        if not project:
            return
        project_dir = pm.project_dir(project)
        store = GraphStore(project_dir)
        orchestrator = MCPOrchestrator(store)
        results = orchestrator.search(query, top_k=3)

        if results:
            console.print(f"  [green]✓[/green] Found {len(results)} results for [cyan]'{query}'[/cyan]")
            for item in results[:3]:
                console.print(
                    f"    [dim]•[/dim] [{item.node_type}] {item.qualname}  "
                    f"[dim]score={item.score:.3f}[/dim]"
                )
        else:
            console.print("  [yellow]No results yet — try a specific search after indexing[/yellow]")
        store.close()
    except Exception:
        console.print("  [yellow]Demo search skipped (index may still be building)[/yellow]")


def show_quickstart_next_steps(project_type: str) -> None:
    """Show contextual next-step suggestions after quickstart."""
    suggestions = {
        "Python": [
            "[cyan]cg search[/cyan] 'database models'          — Find code semantically",
            "[cyan]cg chat start[/cyan]                        — Chat with AI about your code",
            "[cyan]cg v2 generate[/cyan] 'add API endpoint'    — Generate new code",
        ],
        "JavaScript/TypeScript": [
            "[cyan]cg search[/cyan] 'API routes'               — Find code semantically",
            "[cyan]cg chat start[/cyan]                        — Chat with AI about your code",
            "[cyan]cg v2 generate[/cyan] 'add REST endpoint'   — Generate new code",
        ],
        "Go": [
            "[cyan]cg search[/cyan] 'HTTP handler'             — Find code semantically",
            "[cyan]cg chat start[/cyan]                        — Chat with AI about your code",
            "[cyan]cg v2 generate[/cyan] 'add gRPC service'    — Generate new code",
        ],
    }
    defaults = [
        "[cyan]cg search[/cyan] 'your query'               — Find code semantically",
        "[cyan]cg chat start[/cyan]                        — Chat with AI about your code",
        "[cyan]cg v2 generate[/cyan] 'description'         — Generate new code",
    ]
    steps = suggestions.get(project_type, defaults)

    console.print()
    console.print(
        Panel.fit(
            "\n".join([f"  {step}" for step in steps]),
            title="[bold green]🎉 Try these commands next:[/bold green]",
            border_style="green",
            padding=(0, 1),
        )
    )
    console.print()


@quickstart_app.command("run")
def quickstart(
    path: str = typer.Argument(".", help="Path to the project to index."),
    skip_setup: bool = typer.Option(False, "--skip-setup", help="Skip LLM setup even if not configured."),
    skip_index: bool = typer.Option(False, "--skip-index", help="Skip auto-indexing."),
):
    """🚀 Quick setup wizard — get started in 30 seconds.

    Example:
      cg quickstart
      cg quickstart ./my-project
      cg quickstart --skip-setup
    """
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]🚀 CodeGraph Quick Start[/bold cyan]\n"
            "[dim]AI-powered code intelligence in 30 seconds[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    # 1. Detect project type
    project_type = detect_project_type()
    if project_type != "Unknown":
        console.print(f"  [green]✓[/green] Detected [bold]{project_type}[/bold] project")
    else:
        console.print("  [yellow]⚠[/yellow] Could not detect project type — will index all supported files")

    # 2. Check if setup needed
    if not skip_setup and not config_exists():
        console.print()
        console.print("  [yellow]⚙️  No LLM configured yet[/yellow]")
        do_setup = typer.confirm("  Run interactive setup?", default=True)
        if do_setup:
            run_setup()
        else:
            console.print("  [dim]Skipped — using defaults (ollama/hash). Run 'cg config setup' later.[/dim]")
    else:
        console.print(f"  [green]✓[/green] Configuration found")

    # 3. Index current directory
    if not skip_index:
        console.print()
        console.print("  [cyan]📊 Indexing your project...[/cyan]")
        try:
            stats = run_index(path)
            if stats:
                console.print(
                    f"  [green]✓[/green] Indexed [bold]{stats['nodes']}[/bold] nodes "
                    f"and [bold]{stats['edges']}[/bold] edges"
                )
            else:
                console.print("  [green]✓[/green] Indexing complete")
        except Exception as e:
            console.print(f"  [red]✗[/red] Indexing failed: {e}")
            console.print("  [dim]You can index manually later with: cg index <path>[/dim]")
    else:
        console.print("  [dim]Skipped indexing[/dim]")

    # 4. Run demo search
    if not skip_index:
        console.print()
        console.print("  [magenta]🔍 Running sample search...[/magenta]")
        demo_search_based_on_project_type(project_type)

    # 5. Show next steps
    show_quickstart_next_steps(project_type)
