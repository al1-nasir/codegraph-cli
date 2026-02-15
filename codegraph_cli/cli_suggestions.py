"""Contextual next-step suggestions after command completion."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel

console = Console()

NEXT_STEPS: dict[str, list[str]] = {
    "index": [
        "[cyan]cg search[/cyan] 'your query'             Search code semantically",
        "[cyan]cg chat start[/cyan]                      Ask questions about your code",
        "[cyan]cg impact[/cyan] 'symbol'                 See what depends on a function",
    ],
    "search": [
        "[cyan]cg chat start[/cyan]                      Discuss results with AI",
        "[cyan]cg v2 review[/cyan] <file>                Get AI code review",
        "[cyan]cg graph[/cyan] 'symbol'                  Visualize dependencies",
    ],
    "generate": [
        "[cyan]cg v2 review[/cyan] <file>                Review generated code",
        "[cyan]cg v2 test unit[/cyan] <symbol>           Generate tests for new code",
        "[cyan]cg export-graph[/cyan]                    Visualize changes",
    ],
    "diagnose": [
        "[cyan]cg v2 test unit[/cyan] <symbol>           Add tests for fixed code",
        "[cyan]cg v2 review[/cyan] <file>                Review the fixes",
    ],
    "chat": [
        "[cyan]cg search[/cyan] 'query'                  Find specific code",
        "[cyan]cg v2 generate[/cyan] 'description'       Generate new code",
    ],
    "review": [
        "[cyan]cg v2 diagnose fix[/cyan] <path>          Auto-fix detected issues",
        "[cyan]cg v2 refactor rename[/cyan] <old> <new>  Rename symbols safely",
        "[cyan]cg v2 test unit[/cyan] <symbol>           Generate tests",
    ],
    "refactor": [
        "[cyan]cg v2 review[/cyan] <file>                Review refactored code",
        "[cyan]cg v2 test unit[/cyan] <symbol>           Add tests after refactoring",
    ],
    "test": [
        "[cyan]cg v2 review[/cyan] <file>                Review generated tests",
        "[cyan]cg chat start[/cyan]                      Discuss test strategy with AI",
    ],
    "impact": [
        "[cyan]cg graph[/cyan] 'symbol'                  Visualize dependency graph",
        "[cyan]cg search[/cyan] 'related query'          Find related code",
        "[cyan]cg chat start[/cyan]                      Deep-dive with AI",
    ],
    "quickstart": [
        "[cyan]cg search[/cyan] 'your query'             Search code semantically",
        "[cyan]cg chat start[/cyan]                      Chat with AI about your code",
        "[cyan]cg v2 generate[/cyan] 'description'       Generate new code",
    ],
}


def show_next_steps(command_name: str) -> None:
    """Show contextual next steps after command completion.

    Args:
        command_name: The name of the command that just completed.
    """
    steps = NEXT_STEPS.get(command_name)
    if not steps:
        return

    console.print(
        "\n"
        + Panel.fit(
            "\n".join([f"  {step}" for step in steps]),
            title="[bold cyan]💡 Next steps:[/bold cyan]",
            border_style="cyan",
            padding=(0, 1),
        ).markup  # type: ignore[attr-defined]
        if False
        else ""
    )
    # Use direct Panel printing to avoid markup issues
    console.print()
    console.print(
        Panel(
            "\n".join([f"  {step}" for step in steps]),
            title="[bold cyan]💡 Next steps:[/bold cyan]",
            border_style="cyan",
            padding=(0, 1),
        )
    )
