"""CLI commands for refactoring operations."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from . import config
from .diff_engine import DiffEngine
from .refactor_agent import RefactorAgent
from .storage import GraphStore, ProjectManager

# Create sub-app for refactor commands
refactor_app = typer.Typer(help="Safe refactoring with dependency tracking")


def _get_refactor_agent(pm: ProjectManager) -> RefactorAgent:
    """Get RefactorAgent with current project context."""
    project = pm.get_current_project()
    if not project:
        raise typer.BadParameter("No project loaded. Use 'cg load-project <name>' or run 'cg index <path>'.")
    
    project_dir = pm.project_dir(project)
    if not project_dir.exists():
        raise typer.BadParameter(f"Loaded project '{project}' does not exist in memory.")
    
    store = GraphStore(project_dir)
    return RefactorAgent(store)


@refactor_app.command("rename")
def rename_symbol(
    old_name: str = typer.Argument(..., help="Current symbol name"),
    new_name: str = typer.Argument(..., help="New symbol name"),
    preview_only: bool = typer.Option(False, "--preview", "-p", help="Preview changes without applying"),
    auto_apply: bool = typer.Option(False, "--auto-apply", "-y", help="Apply changes without confirmation"),
):
    """✏️  Rename a symbol and update all references.

    Example:
      cg v2 refactor rename "old_function" "new_function"
      cg v2 refactor rename "UserModel" "Account" --preview
    """
    pm = ProjectManager()
    agent = _get_refactor_agent(pm)
    
    typer.echo(f"🔄 Renaming '{old_name}' to '{new_name}'...")
    
    try:
        plan = agent.rename_symbol(old_name, new_name)
    except ValueError as e:
        typer.echo(f"❌ Error: {e}")
        raise typer.Exit(1)
    
    # Show refactoring plan
    typer.echo(f"\n📝 Refactoring Plan: {plan.description}")
    typer.echo(f"   Files to modify: {len(plan.changes)}")
    typer.echo(f"   Call sites to update: {plan.num_call_sites}")
    typer.echo("")
    
    # Show changes
    diff_engine = DiffEngine()
    for change in plan.changes:
        typer.echo(f"{'='*60}")
        typer.echo(f"[MODIFY] {change.file_path}")
        typer.echo(f"{'='*60}")
        if change.diff:
            typer.echo(change.diff)
        typer.echo("")
    
    if preview_only:
        typer.echo("📋 Preview only mode - no changes applied")
        return
    
    if not auto_apply:
        apply = typer.confirm("\n❓ Apply refactoring?", default=False)
        if not apply:
            typer.echo("❌ Refactoring cancelled")
            return
    
    # Apply changes
    typer.echo("\n✨ Applying refactoring...")
    
    # Create a CodeProposal-like structure for applying
    from .models_v2 import CodeProposal
    proposal = CodeProposal(
        id=str(plan.description),
        description=plan.description,
        changes=plan.changes
    )
    
    result = diff_engine.apply_changes(proposal, backup=True)
    
    if result.success:
        typer.echo(f"✅ Successfully refactored {len(result.files_changed)} file(s)")
        if result.backup_id:
            typer.echo(f"💾 Backup created: {result.backup_id}")
            typer.echo(f"   Rollback with: cg v2 rollback {result.backup_id}")
    else:
        typer.echo(f"❌ Failed to apply refactoring: {result.error}")


@refactor_app.command("extract-function")
def extract_function(
    file_path: str = typer.Argument(..., help="File containing code to extract"),
    start_line: int = typer.Argument(..., help="Start line number"),
    end_line: int = typer.Argument(..., help="End line number"),
    function_name: str = typer.Argument(..., help="Name for the new function"),
    preview_only: bool = typer.Option(False, "--preview", "-p", help="Preview changes without applying"),
    auto_apply: bool = typer.Option(False, "--auto-apply", "-y", help="Apply changes without confirmation"),
):
    """📤 Extract code range into a new function.

    Example:
      cg v2 refactor extract-function src/handler.py 10 25 process_request
      cg v2 refactor extract-function src/utils.py 5 15 validate_input --preview
    """
    pm = ProjectManager()
    agent = _get_refactor_agent(pm)
    
    typer.echo(f"📤 Extracting lines {start_line}-{end_line} to function '{function_name}'...")
    
    try:
        plan = agent.extract_function(file_path, start_line, end_line, function_name)
    except ValueError as e:
        typer.echo(f"❌ Error: {e}")
        raise typer.Exit(1)
    
    # Show plan
    typer.echo(f"\n📝 Refactoring Plan: {plan.description}")
    typer.echo("")
    
    # Show changes
    diff_engine = DiffEngine()
    for change in plan.changes:
        typer.echo(f"{'='*60}")
        typer.echo(f"[MODIFY] {change.file_path}")
        typer.echo(f"{'='*60}")
        if change.diff:
            typer.echo(change.diff)
        typer.echo("")
    
    if preview_only:
        typer.echo("📋 Preview only mode - no changes applied")
        return
    
    if not auto_apply:
        apply = typer.confirm("\n❓ Apply refactoring?", default=False)
        if not apply:
            typer.echo("❌ Refactoring cancelled")
            return
    
    # Apply changes
    typer.echo("\n✨ Applying refactoring...")
    
    from .models_v2 import CodeProposal
    proposal = CodeProposal(
        id=str(plan.description),
        description=plan.description,
        changes=plan.changes
    )
    
    result = diff_engine.apply_changes(proposal, backup=True)
    
    if result.success:
        typer.echo(f"✅ Successfully refactored {len(result.files_changed)} file(s)")
        if result.backup_id:
            typer.echo(f"💾 Backup created: {result.backup_id}")
    else:
        typer.echo(f"❌ Failed to apply refactoring: {result.error}")


@refactor_app.command("extract-service")
def extract_service(
    symbols: List[str] = typer.Argument(..., help="Function names to extract (space-separated)"),
    target_file: str = typer.Option(..., "--target", "-t", help="Target service file path"),
    preview_only: bool = typer.Option(False, "--preview", "-p", help="Preview changes without applying"),
    auto_apply: bool = typer.Option(False, "--auto-apply", "-y", help="Apply changes without confirmation"),
):
    """📤 Extract multiple functions to a new service file.

    Example:
      cg v2 refactor extract-service send_email notify_user --target src/notifications.py
    """
    pm = ProjectManager()
    agent = _get_refactor_agent(pm)
    
    typer.echo(f"📤 Extracting {len(symbols)} function(s) to {target_file}...")
    
    try:
        plan = agent.extract_service(symbols, target_file)
    except ValueError as e:
        typer.echo(f"❌ Error: {e}")
        raise typer.Exit(1)
    
    # Show plan
    typer.echo(f"\n📝 Refactoring Plan: {plan.description}")
    typer.echo(f"   Files to create: {sum(1 for c in plan.changes if c.change_type == 'create')}")
    typer.echo(f"   Files to modify: {sum(1 for c in plan.changes if c.change_type == 'modify')}")
    typer.echo(f"   Call sites to update: {plan.num_call_sites}")
    typer.echo("")
    
    # Show changes
    diff_engine = DiffEngine()
    for change in plan.changes:
        typer.echo(f"{'='*60}")
        typer.echo(f"[{change.change_type.upper()}] {change.file_path}")
        typer.echo(f"{'='*60}")
        if change.change_type == "create":
            typer.echo(change.new_content[:500] + "..." if len(change.new_content or "") > 500 else change.new_content)
        elif change.diff:
            typer.echo(change.diff)
        typer.echo("")
    
    if preview_only:
        typer.echo("📋 Preview only mode - no changes applied")
        return
    
    if not auto_apply:
        apply = typer.confirm("\n❓ Apply refactoring?", default=False)
        if not apply:
            typer.echo("❌ Refactoring cancelled")
            return
    
    # Apply changes
    typer.echo("\n✨ Applying refactoring...")
    
    from .models_v2 import CodeProposal
    proposal = CodeProposal(
        id=str(plan.description),
        description=plan.description,
        changes=plan.changes
    )
    
    result = diff_engine.apply_changes(proposal, backup=True)
    
    if result.success:
        typer.echo(f"✅ Successfully refactored {len(result.files_changed)} file(s)")
        if result.backup_id:
            typer.echo(f"💾 Backup created: {result.backup_id}")
    else:
        typer.echo(f"❌ Failed to apply refactoring: {result.error}")
