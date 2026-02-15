"""CLI commands for v2.0 code generation features."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from . import config
from .cli_diagnose import diagnose_app
from .cli_refactor import refactor_app
from .cli_test import test_app
from .codegen_agent import CodeGenAgent
from .diff_engine import DiffEngine
from .llm import LocalLLM
from .storage import GraphStore, ProjectManager

# Create sub-app for v2 commands
v2_app = typer.Typer(help="v2.0 code generation features (experimental)")

# Register sub-commands under v2
v2_app.add_typer(refactor_app, name="refactor")
v2_app.add_typer(diagnose_app, name="diagnose")
v2_app.add_typer(test_app, name="test")


def _get_codegen_agent(pm: ProjectManager) -> CodeGenAgent:
    """Get CodeGenAgent with current project context."""
    project = pm.get_current_project()
    if not project:
        raise typer.BadParameter("No project loaded. Use 'cg load-project <name>' or run 'cg index <path>'.")
    
    project_dir = pm.project_dir(project)
    if not project_dir.exists():
        raise typer.BadParameter(f"Loaded project '{project}' does not exist in memory.")
    
    store = GraphStore(project_dir)
    llm = LocalLLM(
        model=config.LLM_MODEL,
        provider=config.LLM_PROVIDER,
        api_key=config.LLM_API_KEY
    )
    
    return CodeGenAgent(store, llm)


@v2_app.command("generate")
def generate_code(
    prompt: str = typer.Argument(..., help="Natural language description of code to generate"),
    context_file: Optional[str] = typer.Option(None, "--file", "-f", help="File to use as context"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path or directory"),
    preview_only: bool = typer.Option(False, "--preview", "-p", help="Preview changes without applying"),
    auto_apply: bool = typer.Option(False, "--auto-apply", "-y", help="Apply changes without confirmation"),
    llm_provider: str = typer.Option(config.LLM_PROVIDER, help="LLM provider"),
    llm_api_key: Optional[str] = typer.Option(config.LLM_API_KEY, help="API key for LLM"),
    llm_model: str = typer.Option(config.LLM_MODEL, help="LLM model"),
):
    """✨ Generate code from natural language description (v2.0 experimental).

    Example:
      cg v2 generate 'add REST API endpoint for users'
      cg v2 generate 'add login function' --file auth.py
      cg v2 generate 'create data model' --output models/ --auto-apply
    """
    pm = ProjectManager()
    agent = _get_codegen_agent(pm)
    
    typer.echo("🤖 Generating code...")
    
    # Generate code
    proposal = agent.generate(prompt, context_file=context_file)
    
    # Update file paths if output directory specified
    if output:
        output_path = Path(output)
        for change in proposal.changes:
            if output_path.is_dir() or not output_path.suffix:
                # Output is a directory
                output_path.mkdir(parents=True, exist_ok=True)
                filename = Path(change.file_path).name
                change.file_path = str(output_path / filename)
            else:
                # Output is a specific file
                change.file_path = str(output_path)
    
    # Preview changes
    diff_engine = DiffEngine()
    preview = diff_engine.preview_changes(proposal)
    typer.echo(preview)
    
    # Analyze impact
    typer.echo("")
    impact = agent.preview_impact(proposal)
    typer.echo(impact)
    
    # Apply if requested
    if preview_only:
        typer.echo("\n📋 Preview only mode - no changes applied")
        return
    
    if not auto_apply:
        apply = typer.confirm("\n❓ Apply these changes?", default=False)
        if not apply:
            typer.echo("❌ Changes not applied")
            return
    
    # Apply changes
    typer.echo("\n✨ Applying changes...")
    result = agent.apply_changes(proposal)
    
    if result.success:
        typer.echo(f"✅ Successfully applied changes to {len(result.files_changed)} file(s)")
        if result.backup_id:
            typer.echo(f"💾 Backup created: {result.backup_id}")
            typer.echo(f"   Rollback with: cg v2 rollback {result.backup_id}")
    else:
        typer.echo(f"❌ Failed to apply changes: {result.error}")


@v2_app.command("rollback")
def rollback_changes(
    backup_id: str = typer.Argument(..., help="Backup ID to rollback to"),
):
    """⏪ Rollback to a previous backup.

    Example:
      cg v2 rollback backup_20240101_120000
    """
    diff_engine = DiffEngine()
    
    typer.echo(f"🔄 Rolling back to backup: {backup_id}")
    success = diff_engine.rollback(backup_id)
    
    if success:
        typer.echo("✅ Successfully rolled back changes")
    else:
        typer.echo(f"❌ Failed to rollback - backup not found: {backup_id}")


@v2_app.command("list-backups")
def list_backups():
    """📦 List all available backups.

    Example:
      cg v2 list-backups
    """
    diff_engine = DiffEngine()
    backups = diff_engine.list_backups()
    
    if not backups:
        typer.echo("No backups found")
        return
    
    typer.echo(f"📦 Found {len(backups)} backup(s):\n")
    for backup in backups:
        typer.echo(f"ID: {backup['backup_id']}")
        typer.echo(f"   Description: {backup['description']}")
        typer.echo(f"   Timestamp: {backup['timestamp']}")
        typer.echo(f"   Files: {len(backup['files'])}")
        typer.echo("")


@v2_app.command("review")
def review_code(
    file_path: str = typer.Argument(..., help="File to review"),
    check: str = typer.Option("all", help="Check type: bugs, security, performance, all"),
    severity: str = typer.Option("all", help="Minimum severity: low, medium, high, critical"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    use_llm: bool = typer.Option(False, "--llm", help="Use LLM for deeper analysis"),
    show_fixes: bool = typer.Option(False, "--fix", help="Show auto-fix suggestions"),
):
    """🔍 Run AI-powered code review on a file.

    Example:
      cg v2 review src/auth.py
      cg v2 review src/models.py --check security --llm
      cg v2 review src/api.py --fix --verbose
    """
    from .bug_detector import BugDetector
    from .security_scanner import SecurityScanner
    from .performance_analyzer import PerformanceAnalyzer
    
    pm = ProjectManager()
    project = pm.get_current_project()
    if not project:
        typer.echo("❌ No project loaded. Use 'cg load-project <name>' first.", err=True)
        raise typer.Exit(1)
    
    project_dir = pm.project_dir(project)
    store = GraphStore(project_dir)
    llm = LocalLLM(
        model=config.LLM_MODEL,
        provider=config.LLM_PROVIDER,
        api_key=config.LLM_API_KEY
    ) if use_llm else None
    
    all_issues = []
    
    # Run selected checks
    typer.echo(f"🔍 Analyzing {file_path}...")
    if use_llm:
        typer.echo("  🤖 LLM analysis enabled")
    if show_fixes:
        typer.echo("  🔧 Auto-fix suggestions enabled")
    typer.echo("")
    
    if check in ["bugs", "all"]:
        detector = BugDetector(store, llm)
        bug_issues = detector.analyze_file(file_path, use_llm=use_llm)
        all_issues.extend(bug_issues)
        if verbose:
            typer.echo(f"  Bug detection: {len(bug_issues)} issue(s)")
    
    if check in ["security", "all"]:
        scanner = SecurityScanner(store)
        security_issues = scanner.scan_file(file_path, generate_fixes=show_fixes)
        all_issues.extend(security_issues)
        if verbose:
            typer.echo(f"  Security scan: {len(security_issues)} issue(s)")
    
    if check in ["performance", "all"]:
        analyzer = PerformanceAnalyzer(store)
        perf_issues = analyzer.analyze_file(file_path)
        all_issues.extend(perf_issues)
        if verbose:
            typer.echo(f"  Performance analysis: {len(perf_issues)} issue(s)")
    
    # Filter by severity
    if severity != "all":
        severity_order = ["low", "medium", "high", "critical"]
        min_level = severity_order.index(severity)
        all_issues = [
            i for i in all_issues 
            if severity_order.index(i["severity"]) >= min_level
        ]
    
    # Display results
    if not all_issues:
        typer.echo("✅ No issues found!")
        store.close()
        return
    
    typer.echo(f"\n🔍 Found {len(all_issues)} issue(s):\n")
    
    # Group by severity
    by_severity = {"critical": [], "high": [], "medium": [], "low": []}
    for issue in all_issues:
        by_severity[issue["severity"]].append(issue)
    
    severity_icons = {
        "critical": "🚨",
        "high": "🔴",
        "medium": "⚠️",
        "low": "ℹ️"
    }
    
    # Display grouped by severity
    for sev in ["critical", "high", "medium", "low"]:
        issues = by_severity[sev]
        if not issues:
            continue
        
        typer.echo(f"{severity_icons[sev]} {sev.upper()} ({len(issues)} issue(s)):")
        for issue in sorted(issues, key=lambda x: x["line"]):
            typer.echo(f"  Line {issue['line']}: {issue['message']}")
            typer.echo(f"    Type: {issue['type']}")
            
            if verbose and "code_snippet" in issue:
                typer.echo(f"    Code: {issue['code_snippet']}")
            
            # Show LLM explanation if available
            if "llm_explanation" in issue:
                typer.echo(f"    🤖 Analysis: {issue['llm_explanation']}")
            
            typer.echo(f"    💡 {issue['suggestion']}")
            
            # Show auto-fix if available
            if show_fixes and "auto_fix" in issue:
                typer.echo(f"    🔧 Auto-fix:")
                for line in issue["auto_fix"].split("\n"):
                    typer.echo(f"       {line}")
            
            typer.echo("")
    
    store.close()

