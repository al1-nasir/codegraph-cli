"""Project health dashboard for CodeGraph CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table

from . import config
from .storage import GraphStore, ProjectManager

console = Console()

health_app = typer.Typer(help="🏥 Project health dashboard")


def _render_bar(percentage: float) -> str:
    """Render a simple text progress bar."""
    filled = int(percentage / 10)
    empty = 10 - filled
    bar = "█" * filled + "░" * empty
    if percentage >= 80:
        color = "green"
    elif percentage >= 60:
        color = "yellow"
    else:
        color = "red"
    return f"[{color}]{bar}[/{color}] {percentage:.0f}%"


def _score_color(score: float) -> str:
    if score >= 80:
        return "green"
    elif score >= 60:
        return "yellow"
    return "red"


def _analyze_code_quality(store: GraphStore) -> Dict[str, Any]:
    """Analyze code quality from indexed nodes."""
    nodes = store.get_nodes()
    if not nodes:
        return {"score": 0, "total_nodes": 0, "functions": 0, "classes": 0}

    functions = [n for n in nodes if n.get("node_type") == "function"]
    classes = [n for n in nodes if n.get("node_type") == "class"]

    # Heuristic quality score based on:
    # - docstring presence
    # - reasonable function size
    # - naming conventions
    score_points = 0
    total_checks = 0

    for fn in functions:
        total_checks += 1
        snippet = fn.get("snippet", "")
        # Has docstring?
        if '"""' in snippet or "'''" in snippet:
            score_points += 1
        else:
            score_points += 0.3
        # Reasonable size (< 50 lines)
        start = fn.get("start_line", 0)
        end = fn.get("end_line", 0)
        if 0 < (end - start) < 50:
            score_points += 0.5
            total_checks += 0.5
        else:
            total_checks += 0.5

    quality_pct = (score_points / max(total_checks, 1)) * 100
    return {
        "score": min(quality_pct, 100),
        "total_nodes": len(nodes),
        "functions": len(functions),
        "classes": len(classes),
        "documented": sum(
            1 for fn in functions if '"""' in fn.get("snippet", "") or "'''" in fn.get("snippet", "")
        ),
    }


def _analyze_complexity(store: GraphStore) -> Dict[str, Any]:
    """Estimate code complexity from indexed functions."""
    nodes = store.get_nodes()
    functions = [n for n in nodes if n.get("node_type") == "function"]

    if not functions:
        return {"avg": 0, "max": 0, "high_complexity": []}

    complexities = []
    high_complexity = []

    for fn in functions:
        snippet = fn.get("snippet", "")
        # Rough cyclomatic complexity estimate
        complexity = 1  # base
        for keyword in ["if ", "elif ", "else:", "for ", "while ", "except ", "and ", "or ", "case "]:
            complexity += snippet.count(keyword)

        complexities.append(complexity)
        if complexity > 10:
            high_complexity.append({
                "name": fn.get("qualname", fn.get("name", "unknown")),
                "complexity": complexity,
                "file": fn.get("file_path", ""),
            })

    avg_complexity = sum(complexities) / len(complexities) if complexities else 0
    max_complexity = max(complexities) if complexities else 0

    return {
        "avg": avg_complexity,
        "max": max_complexity,
        "high_complexity": sorted(high_complexity, key=lambda x: x["complexity"], reverse=True)[:10],
    }


def _analyze_security(store: GraphStore) -> Dict[str, Any]:
    """Quick security scan from indexed code."""
    nodes = store.get_nodes()
    critical = 0
    high = 0
    medium = 0

    security_patterns = {
        "critical": ["eval(", "exec(", "os.system(", "subprocess.call(shell=True"],
        "high": ["pickle.loads", "yaml.load(", "password", "secret", "SQL"],
        "medium": ["print(", "TODO", "FIXME", "HACK"],
    }

    for node in nodes:
        snippet = node.get("snippet", "")
        for pattern in security_patterns["critical"]:
            if pattern in snippet:
                critical += 1
        for pattern in security_patterns["high"]:
            if pattern.lower() in snippet.lower():
                high += 1
        for pattern in security_patterns["medium"]:
            if pattern in snippet:
                medium += 1

    return {"critical": critical, "high": high, "medium": medium}


def _generate_recommendations(
    quality: Dict[str, Any],
    complexity: Dict[str, Any],
    security: Dict[str, Any],
) -> List[str]:
    """Generate actionable recommendations."""
    recs = []

    documented = quality.get("documented", 0)
    total_fns = quality.get("functions", 0)
    if total_fns > 0 and documented / total_fns < 0.5:
        recs.append(
            f"Add docstrings: only {documented}/{total_fns} functions are documented. "
            "Run 'cg v2 review <file>' for suggestions."
        )

    if security["critical"] > 0:
        recs.append(
            f"Fix {security['critical']} critical security pattern(s) (eval/exec/os.system). "
            "Run 'cg v2 review <file> --check security'."
        )

    if security["high"] > 0:
        recs.append(
            f"Review {security['high']} high-severity pattern(s). "
            "Run 'cg v2 review <file> --check security'."
        )

    if complexity["max"] > 15:
        recs.append(
            f"Refactor high-complexity functions (max: {complexity['max']}). "
            "Run 'cg v2 refactor extract-function' to simplify."
        )

    if quality["score"] < 70:
        recs.append("Run 'cg v2 review <file>' on low-quality modules to get improvement suggestions.")

    if not recs:
        recs.append("Your project looks healthy! Keep up the good work. 🎉")

    return recs


@health_app.command("dashboard")
def health_dashboard():
    """🏥 Project health dashboard.

    Shows code quality, complexity, security patterns, and recommendations.

    Example:
      cg health dashboard
    """
    pm = ProjectManager()
    project = pm.get_current_project()
    if not project:
        console.print("[red]✗[/red] No project loaded. Run 'cg index <path>' first.")
        raise typer.Exit(1)

    project_dir = pm.project_dir(project)
    store = GraphStore(project_dir)

    console.print(f"\n[bold cyan]🏥 Analyzing project health for '{project}'...[/bold cyan]\n")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Running analyzers...", total=3)

        progress.update(task, description="[cyan]Analyzing code quality...")
        quality = _analyze_code_quality(store)
        progress.advance(task)

        progress.update(task, description="[cyan]Analyzing complexity...")
        complexity = _analyze_complexity(store)
        progress.advance(task)

        progress.update(task, description="[cyan]Scanning for security patterns...")
        security = _analyze_security(store)
        progress.advance(task)

    # ── Overall score ────────────────────────────────────────
    quality_score = quality["score"]
    complexity_penalty = min(complexity["avg"] * 2, 30)
    security_penalty = security["critical"] * 10 + security["high"] * 3
    overall = max(0, min(100, quality_score - complexity_penalty - security_penalty))
    color = _score_color(overall)

    console.print(
        Panel.fit(
            f"[bold {color}]{overall:.0f}%[/bold {color}]",
            title="[bold]Overall Health Score[/bold]",
            border_style=color,
        )
    )

    # ── Detailed metrics ─────────────────────────────────────
    table = Table(title="\nDetailed Metrics", show_header=True, show_lines=False)
    table.add_column("Metric", style="cyan", width=18)
    table.add_column("Score", width=24)
    table.add_column("Details", min_width=30)

    table.add_row(
        "Code Quality",
        _render_bar(quality_score),
        f"{quality['functions']} functions, {quality['classes']} classes, "
        f"{quality['documented']} documented",
    )

    complexity_pct = max(0, 100 - complexity["avg"] * 5)
    table.add_row(
        "Complexity",
        _render_bar(complexity_pct),
        f"Avg: {complexity['avg']:.1f}, Max: {complexity['max']}",
    )

    sec_pct = max(0, 100 - security["critical"] * 20 - security["high"] * 5 - security["medium"])
    sec_status = (
        "🔴 Critical" if security["critical"] > 0
        else "🟡 Warning" if security["high"] > 0
        else "🟢 Good"
    )
    table.add_row(
        "Security",
        sec_status,
        f"{security['critical']} critical, {security['high']} high, {security['medium']} medium",
    )

    table.add_row(
        "Total Nodes",
        str(quality["total_nodes"]),
        f"Indexed symbols in project '{project}'",
    )

    console.print(table)

    # ── High complexity functions ────────────────────────────
    if complexity["high_complexity"]:
        console.print("\n[bold yellow]⚠️  High-Complexity Functions[/bold yellow]")
        for item in complexity["high_complexity"][:5]:
            console.print(f"  • {item['name']} (complexity: {item['complexity']}) — {item['file']}")

    # ── Recommendations ──────────────────────────────────────
    recs = _generate_recommendations(quality, complexity, security)
    console.print(
        "\n"
        + Panel(
            "\n".join([f"  • {rec}" for rec in recs]),
            title="[bold yellow]📋 Recommendations[/bold yellow]",
            border_style="yellow",
        ).markup  # type: ignore
        if False
        else ""
    )
    console.print(
        Panel(
            "\n".join([f"  • {rec}" for rec in recs]),
            title="[bold yellow]📋 Recommendations[/bold yellow]",
            border_style="yellow",
        )
    )

    store.close()
