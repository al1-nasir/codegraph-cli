"""Undo/redo system and change history for CodeGraph CLI."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import BASE_DIR

console = Console()

history_app = typer.Typer(help="📜 Change history and undo/redo")

HISTORY_FILE = BASE_DIR / "change_history.json"
MAX_HISTORY = 100


class ChangeHistory:
    """Track code changes for undo/redo support."""

    def __init__(self):
        self.history: List[dict] = self._load()
        self.current_index: int = len(self.history) - 1

    def _load(self) -> List[dict]:
        if HISTORY_FILE.exists():
            try:
                return json.loads(HISTORY_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _save(self) -> None:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Trim to max history
        data = self.history[-MAX_HISTORY:]
        HISTORY_FILE.write_text(json.dumps(data, indent=2))

    def record_change(
        self,
        change_type: str,
        description: str,
        files_changed: List[str],
        backup_id: str,
    ) -> None:
        """Record a change for undo/redo."""
        # Truncate any "future" entries if we've undone things
        self.history = self.history[: self.current_index + 1]
        self.history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "type": change_type,
                "description": description,
                "files": files_changed,
                "backup_id": backup_id,
            }
        )
        self.current_index = len(self.history) - 1
        self._save()

    def can_undo(self) -> bool:
        return self.current_index >= 0 and len(self.history) > 0

    def can_redo(self) -> bool:
        return self.current_index < len(self.history) - 1

    def get_undo_entry(self) -> Optional[dict]:
        if not self.can_undo():
            return None
        entry = self.history[self.current_index]
        self.current_index -= 1
        self._save()
        return entry

    def get_redo_entry(self) -> Optional[dict]:
        if not self.can_redo():
            return None
        self.current_index += 1
        entry = self.history[self.current_index]
        self._save()
        return entry


@history_app.command("undo")
def undo(
    steps: int = typer.Option(1, "--steps", "-n", min=1, help="Number of steps to undo."),
):
    """⏪ Undo last code changes.

    Restores files from backups created during generate, refactor, or fix operations.

    Example:
      cg undo
      cg undo --steps 3
    """
    from .diff_engine import DiffEngine

    hist = ChangeHistory()

    if not hist.can_undo():
        console.print("[yellow]No changes to undo.[/yellow]")
        return

    diff_engine = DiffEngine()
    undone = 0

    for _ in range(steps):
        entry = hist.get_undo_entry()
        if not entry:
            console.print("[yellow]No more changes to undo.[/yellow]")
            break

        console.print(f"  [yellow]⏪ Undoing:[/yellow] {entry['description']}")

        success = diff_engine.rollback(entry["backup_id"])
        if success:
            console.print(f"  [green]✓[/green] Restored {len(entry['files'])} file(s)")
            undone += 1
        else:
            console.print(f"  [red]✗[/red] Backup not found: {entry['backup_id']}")

    if undone:
        console.print(f"\n[green]Undid {undone} change(s).[/green]")
        console.print("[dim]Use 'cg redo' to reapply.[/dim]")


@history_app.command("redo")
def redo(
    steps: int = typer.Option(1, "--steps", "-n", min=1, help="Number of steps to redo."),
):
    """⏩ Redo previously undone changes.

    Example:
      cg redo
      cg redo --steps 2
    """
    hist = ChangeHistory()

    if not hist.can_redo():
        console.print("[yellow]No changes to redo.[/yellow]")
        return

    console.print("[cyan]Redo is not yet fully supported — please re-run the command.[/cyan]")
    console.print("[dim]History tracking is active; full redo support coming soon.[/dim]")


@history_app.command("show")
def show_history(
    limit: int = typer.Option(10, "--limit", "-n", min=1, help="Number of entries to show."),
):
    """📜 Show change history.

    Example:
      cg history show
      cg history show --limit 20
    """
    hist = ChangeHistory()

    if not hist.history:
        console.print("[yellow]No change history yet.[/yellow]")
        console.print("[dim]History is recorded when you generate, refactor, or fix code.[/dim]")
        return

    table = Table(title="Change History", show_lines=False)
    table.add_column("#", style="dim", width=4)
    table.add_column("Time", style="cyan", width=16)
    table.add_column("Type", style="magenta", width=12)
    table.add_column("Description", min_width=30)
    table.add_column("Files", justify="right", style="green", width=6)

    entries = hist.history[-limit:]
    for i, change in enumerate(reversed(entries), 1):
        try:
            timestamp = datetime.fromisoformat(change["timestamp"])
            time_str = timestamp.strftime("%Y-%m-%d %H:%M")
        except (ValueError, KeyError):
            time_str = "unknown"

        marker = " ◄" if (len(hist.history) - i) == hist.current_index else ""
        table.add_row(
            str(i),
            time_str,
            change.get("type", "unknown"),
            change.get("description", "—") + marker,
            str(len(change.get("files", []))),
        )

    console.print(table)
    console.print(f"\n[dim]Showing {len(entries)} of {len(hist.history)} entries[/dim]")


@history_app.command("clear")
def clear_history():
    """🗑️  Clear all change history.

    Example:
      cg history clear
    """
    if not HISTORY_FILE.exists():
        console.print("[yellow]No history to clear.[/yellow]")
        return

    if typer.confirm("Clear all change history?", default=False):
        HISTORY_FILE.unlink(missing_ok=True)
        console.print("[green]✓ History cleared.[/green]")
    else:
        console.print("[dim]Cancelled.[/dim]")
