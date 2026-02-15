"""Interactive TUI menu for CodeGraph CLI."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()


def show_interactive_menu() -> None:
    """Display the interactive TUI menu when no command is provided."""
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]🧠 CodeGraph CLI[/bold cyan]\n"
            "[dim]AI-powered code intelligence & multi-agent assistant[/dim]",
            border_style="cyan",
        )
    )

    while True:
        console.print("\n[bold]What would you like to do?[/bold]\n")

        choices = [
            "1. 🔍 Search my codebase",
            "2. 💬 Chat with AI about my code",
            "3. ✨ Generate new code",
            "4. 📊 Analyze code impact",
            "5. 🔧 Review and improve code",
            "6. ⚙️  Configure settings",
            "7. 📚 Learn more / tutorial",
            "8. 🏥 Project health dashboard",
            "0. Exit",
        ]
        for choice in choices:
            console.print(f"  {choice}")

        selection = Prompt.ask(
            "\nChoice",
            choices=["0", "1", "2", "3", "4", "5", "6", "7", "8"],
            default="0",
        )

        if selection == "0":
            console.print("[cyan]Goodbye![/cyan]")
            break

        elif selection == "1":
            query = Prompt.ask("Search query")
            if query.strip():
                _run_search(query.strip())

        elif selection == "2":
            _run_chat()

        elif selection == "3":
            desc = Prompt.ask("What code to generate")
            if desc.strip():
                _run_generate(desc.strip())

        elif selection == "4":
            symbol = Prompt.ask("Symbol to analyze")
            if symbol.strip():
                _run_impact(symbol.strip())

        elif selection == "5":
            path = Prompt.ask("File path to review")
            if path.strip():
                _run_review(path.strip())

        elif selection == "6":
            _run_setup()

        elif selection == "7":
            _show_tutorial()

        elif selection == "8":
            _run_health()


def _run_search(query: str) -> None:
    """Run search from TUI."""
    try:
        from .storage import ProjectManager, GraphStore
        from .orchestrator import MCPOrchestrator

        pm = ProjectManager()
        project = pm.get_current_project()
        if not project:
            console.print("[red]✗[/red] No project loaded. Run 'cg index <path>' first.")
            return
        store = GraphStore(pm.project_dir(project))
        orchestrator = MCPOrchestrator(store)
        results = orchestrator.search(query, top_k=5)
        if not results:
            console.print("[yellow]No results found.[/yellow]")
        else:
            for item in results:
                console.print(
                    f"  [{item.node_type}] {item.qualname}  "
                    f"[dim]score={item.score:.3f}[/dim]"
                )
                console.print(f"    [dim]{item.file_path}:{item.start_line}-{item.end_line}[/dim]")
        store.close()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def _run_chat() -> None:
    """Start chat from TUI."""
    try:
        # Import and invoke via typer to handle all the setup
        console.print("[cyan]Starting chat... (use /exit to return)[/cyan]")
        from .cli_chat import start_chat
        start_chat()
    except (typer.Exit, SystemExit):
        pass
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def _run_generate(description: str) -> None:
    """Run code generation from TUI."""
    try:
        from .cli_v2 import generate_code
        generate_code(description)
    except (typer.Exit, SystemExit):
        pass
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def _run_impact(symbol: str) -> None:
    """Run impact analysis from TUI."""
    try:
        from .storage import ProjectManager, GraphStore
        from .orchestrator import MCPOrchestrator

        pm = ProjectManager()
        project = pm.get_current_project()
        if not project:
            console.print("[red]✗[/red] No project loaded.")
            return
        store = GraphStore(pm.project_dir(project))
        orchestrator = MCPOrchestrator(store)
        report = orchestrator.impact(symbol, hops=2)
        console.print(f"Root: {report.root}")
        if report.impacted:
            console.print("Impacted symbols:")
            for imp in report.impacted:
                console.print(f"  • {imp}")
        console.print(f"\n{report.explanation}")
        store.close()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def _run_review(path: str) -> None:
    """Run code review from TUI."""
    try:
        from .cli_v2 import review_code
        review_code(path)
    except (typer.Exit, SystemExit):
        pass
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def _run_setup() -> None:
    """Run setup wizard from TUI."""
    try:
        from .cli_setup import setup
        setup()
    except (typer.Exit, SystemExit):
        pass
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def _run_health() -> None:
    """Run health dashboard from TUI."""
    try:
        from .cli_health import health_dashboard
        health_dashboard()
    except (typer.Exit, SystemExit):
        pass
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def _show_tutorial() -> None:
    """Show interactive tutorial."""
    console.print(
        Panel(
            "📚 [bold]Quick Tutorial[/bold]\n\n"
            "1. [cyan]Index your project[/cyan]:  cg index ./my-project\n"
            "2. [cyan]Search code[/cyan]:          cg search 'authentication'\n"
            "3. [cyan]Chat with AI[/cyan]:         cg chat start\n"
            "4. [cyan]Generate code[/cyan]:        cg v2 generate 'add API endpoint'\n"
            "5. [cyan]Impact analysis[/cyan]:      cg impact 'symbol_name'\n"
            "6. [cyan]Code review[/cyan]:          cg v2 review path/to/file.py\n"
            "7. [cyan]Project health[/cyan]:       cg health dashboard\n\n"
            "For full reference:  cg cheatsheet\n"
            "For documentation:   cg learn",
            border_style="blue",
        )
    )
