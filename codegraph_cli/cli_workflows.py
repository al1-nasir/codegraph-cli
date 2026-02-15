"""Combined workflow commands for CodeGraph CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from . import config
from .diff_engine import DiffEngine
from .storage import GraphStore, ProjectManager

console = Console()

workflows_app = typer.Typer(help="🔄 Combined workflow commands")


def _ensure_project() -> tuple[ProjectManager, str]:
    """Ensure a project is loaded and return (pm, project_name)."""
    pm = ProjectManager()
    project = pm.get_current_project()
    if not project:
        console.print("[red]✗[/red] No project loaded. Use 'cg load-project <name>' or run 'cg index <path>'.")
        raise typer.Exit(1)
    return pm, project


@workflows_app.command("review-and-fix")
def review_and_fix(
    path: str = typer.Argument(..., help="File to review and fix"),
    auto_apply: bool = typer.Option(False, "--apply", "-y", help="Auto-apply fixes without confirmation"),
    check: str = typer.Option("all", help="Check type: bugs, security, performance, all"),
    use_llm: bool = typer.Option(False, "--llm", help="Use LLM for deeper analysis"),
):
    """🔍 Complete code review → diagnose → fix workflow.

    Runs a full code review, diagnoses issues, and optionally applies fixes.

    Example:
      cg review-and-fix src/auth.py
      cg review-and-fix src/auth.py --apply
      cg review-and-fix src/auth.py --llm --apply
    """
    from .bug_detector import BugDetector
    from .security_scanner import SecurityScanner
    from .performance_analyzer import PerformanceAnalyzer
    from .llm import LocalLLM
    from .validation_engine import ValidationEngine
    from .models_v2 import CodeProposal

    pm, project = _ensure_project()

    file_path = Path(path)
    if not file_path.exists():
        console.print(f"[red]✗[/red] File not found: {path}")
        raise typer.Exit(1)

    project_dir = pm.project_dir(project)
    store = GraphStore(project_dir)
    llm = LocalLLM(
        model=config.LLM_MODEL,
        provider=config.LLM_PROVIDER,
        api_key=config.LLM_API_KEY,
    ) if use_llm else None

    # ── Step 1: Code Review ──────────────────────────────────
    console.print(f"\n[bold cyan]Step 1/3:[/bold cyan] Running code review on [bold]{path}[/bold]\n")

    all_issues = []

    if check in ("bugs", "all"):
        detector = BugDetector(store, llm)
        bug_issues = detector.analyze_file(str(file_path), use_llm=use_llm)
        all_issues.extend(bug_issues)
        console.print(f"  [dim]Bug detection:[/dim] {len(bug_issues)} issue(s)")

    if check in ("security", "all"):
        scanner = SecurityScanner(store)
        security_issues = scanner.scan_file(str(file_path))
        all_issues.extend(security_issues)
        console.print(f"  [dim]Security scan:[/dim] {len(security_issues)} issue(s)")

    if check in ("performance", "all"):
        analyzer = PerformanceAnalyzer(store)
        perf_issues = analyzer.analyze_file(str(file_path))
        all_issues.extend(perf_issues)
        console.print(f"  [dim]Performance:[/dim] {len(perf_issues)} issue(s)")

    if not all_issues:
        console.print("\n[green]✓ No issues found! Code looks good.[/green]")
        store.close()
        return

    # Show summary
    by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for issue in all_issues:
        sev = issue.get("severity", "low")
        by_severity[sev] = by_severity.get(sev, 0) + 1

    console.print(f"\n  Found [bold]{len(all_issues)}[/bold] total issues:")
    for sev, count in by_severity.items():
        if count:
            icon = {"critical": "🚨", "high": "🔴", "medium": "⚠️", "low": "ℹ️"}.get(sev, "•")
            console.print(f"    {icon} {sev}: {count}")

    # ── Step 2: Diagnose ─────────────────────────────────────
    console.print(f"\n[bold cyan]Step 2/3:[/bold cyan] Diagnosing fixable issues\n")

    validator = ValidationEngine()
    errors = validator.diagnose_project(file_path.parent if file_path.is_file() else file_path)

    if errors:
        console.print(f"  Found [bold]{len(errors)}[/bold] syntax error(s)")
    else:
        console.print("  [green]✓[/green] No syntax errors")

    # ── Step 3: Fix ──────────────────────────────────────────
    fixable = [e for e in errors if True]  # all errors are potentially fixable
    if not fixable:
        console.print(f"\n[bold cyan]Step 3/3:[/bold cyan] No auto-fixable issues found")
        console.print("\n[yellow]Review the issues above and fix manually.[/yellow]")

        # Show issues for manual review
        for issue in all_issues[:10]:
            line = issue.get("line", "?")
            msg = issue.get("message", issue.get("error", "Unknown"))
            sev = issue.get("severity", "low")
            console.print(f"  [dim]L{line}[/dim] [{sev}] {msg}")
            if "suggestion" in issue:
                console.print(f"        💡 {issue['suggestion']}")

        store.close()
        return

    console.print(f"\n[bold cyan]Step 3/3:[/bold cyan] Proposing fixes\n")

    changes = []
    for error in fixable:
        file_p = Path(error["file"])
        fix = validator.fix_common_errors(file_p)
        if fix:
            changes.append(fix)

    if not changes:
        console.print("[yellow]No automatic fixes available — manual fixes required.[/yellow]")
        store.close()
        return

    diff_engine = DiffEngine()
    proposal = CodeProposal(
        id="review-and-fix",
        description=f"Review & fix: {len(changes)} issue(s) in {path}",
        changes=changes,
    )
    preview = diff_engine.preview_changes(proposal)
    console.print(preview)

    if auto_apply:
        result = diff_engine.apply_changes(proposal, backup=True)
        if result.success:
            console.print(f"\n[green]✓ Fixed {len(result.files_changed)} file(s)[/green]")
            if result.backup_id:
                console.print(f"[dim]Backup: {result.backup_id} — rollback with: cg v2 rollback {result.backup_id}[/dim]")
        else:
            console.print(f"\n[red]✗ Failed to apply fixes: {result.error}[/red]")
    else:
        if typer.confirm("\nApply these fixes?", default=False):
            result = diff_engine.apply_changes(proposal, backup=True)
            if result.success:
                console.print(f"\n[green]✓ Fixed {len(result.files_changed)} file(s)[/green]")
                if result.backup_id:
                    console.print(f"[dim]Backup: {result.backup_id}[/dim]")
            else:
                console.print(f"\n[red]✗ Failed: {result.error}[/red]")
        else:
            console.print("[yellow]Fixes not applied.[/yellow]")

    store.close()


@workflows_app.command("full-analysis")
def full_analysis(
    symbol: str = typer.Argument(..., help="Symbol to analyze"),
    hops: int = typer.Option(2, min=1, max=6, help="Dependency traversal depth."),
    export: bool = typer.Option(False, "--export", "-e", help="Export graph to HTML"),
):
    """📊 Complete analysis: impact + graph + RAG context for a symbol.

    Example:
      cg full-analysis "UserService.login"
      cg full-analysis "process_payment" --export
      cg full-analysis "main" --hops 3
    """
    from .orchestrator import MCPOrchestrator
    from .graph_export import export_html

    pm, project = _ensure_project()
    project_dir = pm.project_dir(project)
    store = GraphStore(project_dir)
    orchestrator = MCPOrchestrator(store)

    console.print(f"\n[bold]Analyzing: [cyan]{symbol}[/cyan][/bold]\n")

    # 1. Impact analysis
    console.print("[cyan]1/3 Impact analysis...[/cyan]")
    report = orchestrator.impact(symbol, hops=hops)

    if "not found" in report.explanation.lower() and not report.impacted:
        console.print(f"[red]✗[/red] Symbol '{symbol}' not found in current project.")
        search_results = orchestrator.search(symbol, top_k=3)
        if search_results:
            console.print("\n[yellow]Did you mean:[/yellow]")
            for r in search_results:
                console.print(f"  • {r.qualname} ({r.node_type})")
        store.close()
        raise typer.Exit(1)

    console.print(f"  Root: {report.root}")
    console.print(f"  Impacted: {len(report.impacted)} symbol(s)")
    if report.impacted:
        for imp in report.impacted[:10]:
            console.print(f"    • {imp}")
        if len(report.impacted) > 10:
            console.print(f"    [dim]... and {len(report.impacted) - 10} more[/dim]")

    # 2. Dependency graph
    console.print("\n[cyan]2/3 Dependency graph...[/cyan]")
    graph_text = orchestrator.graph(symbol, depth=hops)
    console.print(graph_text)

    # 3. RAG context
    console.print("\n[cyan]3/3 RAG context...[/cyan]")
    rag_text = orchestrator.rag_context(symbol, top_k=6)
    # Count snippets (rough heuristic)
    snippet_count = rag_text.count("──") if rag_text else 0
    console.print(f"  Retrieved {max(snippet_count, 1)} context snippet(s)")

    # Summary
    console.print(f"\n[bold green]━━━ Summary ━━━[/bold green]")
    console.print(f"  Symbol:         {report.root}")
    console.print(f"  Impacted:       {len(report.impacted)} symbol(s)")
    console.print(f"  Traversal:      {hops} hop(s)")

    console.print(f"\n[bold]Explanation:[/bold]")
    console.print(report.explanation)

    if export:
        output_path = Path.cwd() / f"{symbol.replace('.', '_')}_graph.html"
        export_html(store, output_path, focus=symbol)
        console.print(f"\n[green]✓ Graph exported to {output_path}[/green]")

    store.close()
