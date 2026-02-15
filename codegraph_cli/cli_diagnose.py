"""CLI commands for error detection and fixing."""

from __future__ import annotations

from pathlib import Path

import typer

from .diff_engine import DiffEngine
from .models_v2 import CodeProposal
from .storage import ProjectManager
from .validation_engine import ValidationEngine

# Create sub-app for diagnostic commands
diagnose_app = typer.Typer(help="Detect and fix code errors")


@diagnose_app.command("check")
def check_errors(
    path: str = typer.Argument(".", help="Path to check (default: current directory)"),
):
    """🔍 Scan project for syntax errors.

    Example:
      cg v2 diagnose check
      cg v2 diagnose check ./src
    """
    project_path = Path(path).resolve()
    
    if not project_path.exists():
        typer.echo(f"❌ Path not found: {path}")
        raise typer.Exit(1)
    
    typer.echo(f"🔍 Scanning {project_path} for errors...")
    
    validator = ValidationEngine()
    errors = validator.diagnose_project(project_path)
    
    if not errors:
        typer.echo("✅ No syntax errors found!")
        return
    
    typer.echo(f"\n📋 Found {len(errors)} error(s):\n")
    
    for i, error in enumerate(errors, 1):
        typer.echo(f"{i}. {error['file']}:{error['line']}")
        typer.echo(f"   {error['type']}: {error['error']}")
        typer.echo("")


@diagnose_app.command("fix")
def fix_errors(
    path: str = typer.Argument(".", help="Path to fix (default: current directory)"),
    preview_only: bool = typer.Option(False, "--preview", "-p", help="Preview fixes without applying"),
    auto_apply: bool = typer.Option(False, "--auto-apply", "-y", help="Apply fixes without confirmation"),
):
    """🔧 Automatically fix common syntax errors.

    Example:
      cg v2 diagnose fix
      cg v2 diagnose fix ./src --preview
      cg v2 diagnose fix ./src --auto-apply
    """
    project_path = Path(path).resolve()
    
    if not project_path.exists():
        typer.echo(f"❌ Path not found: {path}")
        raise typer.Exit(1)
    
    typer.echo(f"🔧 Fixing errors in {project_path}...")
    
    validator = ValidationEngine()
    errors = validator.diagnose_project(project_path)
    
    if not errors:
        typer.echo("✅ No errors to fix!")
        return
    
    typer.echo(f"\n📋 Found {len(errors)} error(s), attempting fixes...\n")
    
    # Try to fix each file
    changes = []
    for error in errors:
        file_path = Path(error['file'])
        
        typer.echo(f"🔧 Fixing {file_path.name}:{error['line']}...")
        typer.echo(f"   Problem: {error['error']}")
        
        fix = validator.fix_common_errors(file_path)
        
        if fix:
            changes.append(fix)
            typer.echo(f"   ✅ Fix applied")
        else:
            typer.echo(f"   ⚠️  Could not auto-fix (manual intervention needed)")
        typer.echo("")
    
    if not changes:
        typer.echo("❌ No automatic fixes available. Manual fixes required.")
        return
    
    # Show preview
    diff_engine = DiffEngine()
    proposal = CodeProposal(
        id="fix-errors",
        description=f"Fix {len(changes)} syntax error(s)",
        changes=changes
    )
    
    typer.echo("="*60)
    typer.echo("PROPOSED FIXES")
    typer.echo("="*60)
    preview = diff_engine.preview_changes(proposal)
    typer.echo(preview)
    
    if preview_only:
        typer.echo("\n📋 Preview only mode - no changes applied")
        return
    
    if not auto_apply:
        apply = typer.confirm(f"\n❓ Apply {len(changes)} fix(es)?", default=False)
        if not apply:
            typer.echo("❌ Fixes not applied")
            return
    
    # Apply fixes
    typer.echo("\n✨ Applying fixes...")
    result = diff_engine.apply_changes(proposal, backup=True)
    
    if result.success:
        typer.echo(f"✅ Successfully fixed {len(result.files_changed)} file(s)")
        if result.backup_id:
            typer.echo(f"💾 Backup created: {result.backup_id}")
            typer.echo(f"   Rollback with: cg v2 rollback {result.backup_id}")
        
        # Re-check for remaining errors
        typer.echo("\n🔍 Re-checking for errors...")
        remaining = validator.diagnose_project(project_path)
        if remaining:
            typer.echo(f"⚠️  {len(remaining)} error(s) remain (require manual fixing)")
        else:
            typer.echo("✅ All errors fixed!")
    else:
        typer.echo(f"❌ Failed to apply fixes: {result.error}")
