"""Watch mode for auto-reindexing on file changes."""

from __future__ import annotations

import time
from pathlib import Path

import typer
from rich.console import Console

console = Console()

watch_app = typer.Typer(help="👀 Watch mode for auto-reindexing")

# Supported code file extensions
WATCHED_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".go", ".rs", ".java", ".kt", ".kts",
    ".rb", ".php", ".ex", ".exs",
    ".c", ".cpp", ".h", ".hpp", ".cs",
    ".swift", ".dart",
}


class CodeChangeHandler:
    """Handle file system events and trigger re-indexing."""

    def __init__(self, reindex_callback, debounce_seconds: float = 2.0):
        from watchdog.events import FileSystemEventHandler
        self._base_class = FileSystemEventHandler
        self.reindex_callback = reindex_callback
        self.last_reindex = 0.0
        self.debounce_seconds = debounce_seconds
        self._pending_files: set[str] = set()

    def dispatch(self, event):
        """Route events to handler methods."""
        if event.is_directory:
            return
        if hasattr(event, "src_path"):
            self._handle_change(event.src_path)

    def _handle_change(self, src_path: str):
        file_path = Path(src_path)
        if file_path.suffix not in WATCHED_EXTENSIONS:
            return

        # Skip hidden/temp files
        if any(part.startswith(".") for part in file_path.parts):
            return

        now = time.time()
        self._pending_files.add(str(file_path))

        if now - self.last_reindex < self.debounce_seconds:
            return

        # Flush pending
        files = list(self._pending_files)
        self._pending_files.clear()
        self.last_reindex = now

        for f in files:
            self.reindex_callback(Path(f))


@watch_app.command("start")
def watch(
    path: str = typer.Argument(".", help="Path to watch for changes."),
    interval: float = typer.Option(2.0, "--interval", "-i", help="Debounce interval in seconds."),
    full_reindex: bool = typer.Option(False, "--full", help="Re-index entire project on each change."),
):
    """👀 Watch mode — auto-reindex on file changes.

    Monitors your project directory for file changes and automatically
    re-indexes modified files to keep the code graph up to date.

    Example:
      cg watch
      cg watch ./src --interval 5
      cg watch --full
    """
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        console.print("[red]✗[/red] watchdog is not installed.")
        console.print("[dim]Install with: pip install watchdog[/dim]")
        raise typer.Exit(1)

    watch_path = Path(path).resolve()
    if not watch_path.exists():
        console.print(f"[red]✗[/red] Path not found: {path}")
        raise typer.Exit(1)

    from .storage import ProjectManager, GraphStore
    from .orchestrator import MCPOrchestrator

    pm = ProjectManager()
    project = pm.get_current_project()
    if not project:
        console.print("[red]✗[/red] No project loaded. Run 'cg index <path>' first.")
        raise typer.Exit(1)

    project_dir = pm.project_dir(project)
    reindex_count = 0

    def reindex_file(file_path: Path):
        nonlocal reindex_count
        try:
            if full_reindex:
                store = GraphStore(project_dir)
                orchestrator = MCPOrchestrator(store)
                stats = orchestrator.index(watch_path)
                store.close()
                console.print(
                    f"  [green]✓[/green] Full re-index: {stats['nodes']} nodes, {stats['edges']} edges"
                )
            else:
                # Incremental: re-index the single changed file
                store = GraphStore(project_dir)
                orchestrator = MCPOrchestrator(store)
                # For now, do a full re-index (single-file not yet supported in orchestrator)
                stats = orchestrator.index(watch_path)
                store.close()
                console.print(f"  [green]✓[/green] Re-indexed ({file_path.name} changed)")
            reindex_count += 1
        except Exception as e:
            console.print(f"  [red]✗[/red] Re-index failed: {e}")

    console.print(f"\n[bold green]👀 Watching[/bold green] [cyan]{watch_path}[/cyan] for changes...")
    console.print(f"[dim]  Debounce:  {interval}s")
    console.print(f"  Mode:      {'full re-index' if full_reindex else 'incremental'}")
    console.print(f"  Project:   {project}")
    console.print(f"  Press Ctrl+C to stop[/dim]\n")

    handler = CodeChangeHandler(reindex_file, debounce_seconds=interval)

    # Wrap as proper watchdog handler
    class WatchdogAdapter(FileSystemEventHandler):
        def on_modified(self, event):
            handler.dispatch(event)

        def on_created(self, event):
            handler.dispatch(event)

    observer = Observer()
    observer.schedule(WatchdogAdapter(), str(watch_path), recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        console.print(f"\n[yellow]Stopped watching.[/yellow] Re-indexed {reindex_count} time(s).")

    observer.join()
