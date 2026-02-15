"""Interactive chat CLI for conversational coding assistance."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from . import config
from .chat_agent import ChatAgent
from .chat_session import SessionManager
from .llm import LocalLLM
from .orchestrator import MCPOrchestrator
from .rag import RAGRetriever
from .storage import GraphStore, ProjectManager

console = Console()


def _term_width() -> int:
    """Get terminal width, default 80."""
    return shutil.get_terminal_size((80, 24)).columns


def _print_welcome(project_name: str, use_crew: bool, provider: str, model: str, auto_context: dict | None = None):
    """Print the modern welcome banner."""
    mode = "CrewAI Multi-Agent" if use_crew else "Chat"

    console.print()
    console.print(Panel(f"⚡ CodeGraph {mode}", style="bold cyan", width=min(_term_width(), 70)))
    console.print(f"  [dim]Project[/dim]  [white]{project_name}[/white]")
    console.print(f"  [dim]LLM[/dim]      [white]{provider}/{model}[/white]")

    # Show auto-context summary
    if auto_context:
        summary = auto_context.get("summary", {})
        indexed_files = summary.get("indexed_files", 0)
        total_nodes = summary.get("total_nodes", 0)
        node_types = summary.get("node_types", {})
        if indexed_files or total_nodes:
            funcs = node_types.get("function", 0)
            classes = node_types.get("class", 0)
            console.print(f"  [dim]Indexed[/dim]  [white]{indexed_files} files • {funcs} functions • {classes} classes[/white]")
        recent = auto_context.get("recent_files", [])
        if recent:
            console.print(f"  [dim]Recent[/dim]   [white]{', '.join(recent[:3])}[/white]")

    console.print("  [dim]Type [yellow]/help[/yellow] for commands, [yellow]/exit[/yellow] to quit[/dim]")
    console.print(Rule(style="dim"))
    console.print()


def _print_help(use_crew: bool):
    """Print help with styled formatting."""
    console.print()
    table = Table(show_header=False, box=None, padding=(0, 2), title="📖 Commands", title_style="bold cyan")
    table.add_column(style="yellow", min_width=22)
    table.add_column(style="dim")
    cmds = [
        ("/exit", "Exit chat session"),
        ("/clear", "Clear conversation history & start fresh"),
        ("/new", "Start a brand new session"),
        ("/help", "Show this help"),
    ]
    if use_crew:
        cmds.extend([
            ("/backups", "List all file backups"),
            ("/rollback <file>", "Rollback a file to its last backup"),
            ("/undo <file>", "Alias for /rollback"),
        ])
    else:
        cmds.extend([
            ("/apply", "Apply pending code proposal"),
            ("/preview", "Preview pending changes"),
        ])
    for cmd, desc in cmds:
        table.add_row(cmd, desc)
    console.print(table)
    console.print()


def _print_response(text: str):
    """Print assistant response with styling."""
    console.print()
    console.print("  [green]●[/green] [bold]Assistant[/bold]")
    for line in text.split("\n"):
        console.print(f"  {line}", highlight=False)
    console.print()


def _print_status(emoji: str, msg: str, style: str = "green"):
    """Print a status message."""
    console.print(f"  [{style}]{emoji}  {msg}[/{style}]")


def start_chat_repl(
    agent,  # Can be ChatAgent or CrewChatAgent
    session_manager: SessionManager,
    project_name: str,
    session_id: Optional[str] = None,
    use_crew: bool = False,
    provider: str = "",
    model: str = "",
    auto_context: dict | None = None,
):
    """Start interactive REPL for chat."""
    # Load or create session
    if session_id:
        session = session_manager.load_session(session_id)
        if not session:
            _print_status("🆕", "Session not found. Starting new session.", "yellow")
            session = session_manager.create_session(project_name)
        else:
            _print_status("📂", f"Resumed session ({session.message_count} messages)")
    else:
        # Try to load latest session for this project
        latest_id = session_manager.get_latest_session(project_name)
        if latest_id:
            session = session_manager.load_session(latest_id)
            _print_status("📂", f"Resumed session ({session.message_count} messages)")
        else:
            session = session_manager.create_session(project_name)
            _print_status("🆕", "Started new chat session")
    
    # Hydrate agent with session history (enables crew continuity across restarts)
    if hasattr(agent, 'load_session_history') and session.messages:
        agent.load_session_history(session)

    # Welcome
    _print_welcome(project_name, use_crew, provider, model, auto_context=auto_context)
    
    # REPL loop
    while True:
        try:
            # Prompt
            try:
                user_input = console.input("  [blue]●[/blue] [bold]You ›[/bold] ").strip()
            except EOFError:
                console.print("\n  [dim]👋 Goodbye! Session saved.[/dim]\n")
                break
            
            if not user_input:
                continue
            
            # ── Handle commands ──────────────────────────────
            if user_input.startswith("/"):
                cmd = user_input.lower().split()[0]
                
                if cmd == "/exit":
                    session_manager.save_session(session)
                    console.print("\n  [dim]👋 Goodbye! Session saved.[/dim]\n")
                    break
                
                elif cmd == "/clear":
                    session.clear_history()
                    session.clear_proposals()
                    session_manager.save_session(session)
                    # Clear agent-side memory (crew mode)
                    if hasattr(agent, 'clear_history'):
                        agent.clear_history()
                    _print_status("🧹", "Conversation cleared. Fresh start!")
                    console.print()
                    continue
                
                elif cmd == "/new":
                    session_manager.save_session(session)
                    session = session_manager.create_session(project_name)
                    # Clear agent-side memory (crew mode)
                    if hasattr(agent, 'clear_history'):
                        agent.clear_history()
                    _print_status("🆕", "New session started.")
                    console.print()
                    continue
                
                elif cmd == "/help":
                    _print_help(use_crew)
                    continue
                
                elif not use_crew:
                    if cmd == "/apply":
                        if hasattr(agent, 'apply_pending_proposal'):
                            result = agent.apply_pending_proposal(session)
                            _print_status("📋", result)
                        else:
                            _print_status("📋", "No pending proposals.", "yellow")
                        continue
                    elif cmd == "/preview":
                        if session.pending_proposals:
                            for i, prop in enumerate(session.pending_proposals):
                                console.print(f"\n  [bold]Proposal {i+1}:[/bold] {prop.description}")
                                for ch in prop.changes:
                                    icon = {"create": "🆕", "modify": "✏️", "delete": "🗑️"}.get(ch.change_type, "📄")
                                    console.print(f"    {icon} {ch.file_path}")
                        else:
                            _print_status("📋", "No pending proposals.", "yellow")
                        console.print()
                        continue
                
                elif use_crew:
                    if cmd in ("/rollback", "/undo"):
                        parts = user_input.split(maxsplit=1)
                        if len(parts) < 2:
                            _print_status("❓", "Usage: /rollback <file_path> [timestamp]", "yellow")
                            continue
                        args = parts[1].split()
                        file_path = args[0]
                        ts = args[1] if len(args) > 1 else None
                        result = agent.rollback(file_path, ts)
                        _print_status("⏪", result)
                        console.print()
                        continue
                    
                    elif cmd == "/backups":
                        backups = agent.list_all_backups()
                        if not backups:
                            _print_status("📦", "No backups found.", "yellow")
                        else:
                            console.print("\n  [bold cyan]📦 File Backups[/bold cyan]")
                            console.print(Rule(style="dim"))
                            for b in backups:
                                ts = b["timestamp"]
                                fp = b["original_path"]
                                console.print(f"  [white]{ts}[/white]  [dim]{fp}[/dim]")
                            console.print("\n  [dim]Use /rollback <file_path> to restore[/dim]")
                        console.print()
                        continue
                
                _print_status("❓", f"Unknown command: {cmd}. Type /help", "yellow")
                continue
            
            # ── Process message ──────────────────────────────
            session.add_message("user", user_input, datetime.now().isoformat())
            
            # Show thinking indicator
            console.print("\n  [dim]⏳ Thinking...[/dim]", end="")
            
            if use_crew:
                response = agent.process_message(user_input)
            else:
                response = agent.process_message(user_input, session)
            
            # Clear thinking indicator
            console.print(f"\r{' ' * 30}\r", end="")
            
            # Save & display
            session.add_message("assistant", response, datetime.now().isoformat())
            session_manager.save_session(session)
            
            _print_response(response)
            
        except KeyboardInterrupt:
            session_manager.save_session(session)
            console.print("\n\n  [dim]👋 Goodbye! Session saved.[/dim]\n")
            break
        except Exception as e:
            console.print(f"\n  [red]❌ Error: {str(e)}[/red]\n")


# ── Typer app ────────────────────────────────────────────────
chat_app = typer.Typer(help="💬 Interactive chat with AI agents")


def _gather_auto_context(project_context) -> dict:
    """Gather automatic context from the project for chat enrichment.

    Collects project summary (file count, symbol count, node types)
    and recently modified files so the chat session starts with
    useful awareness of the codebase.
    """
    auto_ctx: dict = {}

    try:
        summary = project_context.get_project_summary()
        auto_ctx["summary"] = summary
    except Exception:
        auto_ctx["summary"] = {}

    # Recently modified source files
    try:
        if project_context.has_source_access:
            items = project_context.list_directory(".")
            files = [f for f in items if f["type"] == "file"]
            files.sort(key=lambda f: f.get("modified", ""), reverse=True)
            auto_ctx["recent_files"] = [f["name"] for f in files[:5]]
        else:
            auto_ctx["recent_files"] = []
    except Exception:
        auto_ctx["recent_files"] = []

    return auto_ctx


@chat_app.command("start")
def start_chat(
    session_id: Optional[str] = typer.Option(None, "--session", "-s", help="Resume specific session ID"),
    llm_model: str = typer.Option(config.LLM_MODEL, help="LLM model to use"),
    llm_provider: str = typer.Option(config.LLM_PROVIDER, help="LLM provider"),
    llm_api_key: Optional[str] = typer.Option(config.LLM_API_KEY, help="API key for cloud providers"),
    llm_endpoint: Optional[str] = typer.Option(config.LLM_ENDPOINT, help="LLM endpoint URL"),
    use_crew: bool = typer.Option(False, "--crew", help="Use CrewAI multi-agent system"),
    new_session: bool = typer.Option(False, "--new", "-n", help="Force start a new session"),
):
    """💬 Start interactive chat session.

    Example:
      cg chat start
      cg chat start --crew
      cg chat start --new
      cg chat start --session abc123
    """
    from .embeddings import get_embedder
    from .project_context import ProjectContext
    
    pm = ProjectManager()
    project = pm.get_current_project()
    
    if not project:
        console.print("\n  [red]❌ No project loaded.[/red]")
        console.print("  [dim]Use: cg load-project <name>  or  cg index <path>[/dim]\n")
        raise typer.Exit(1)
    
    # Initialize components
    context = ProjectContext(project, pm)
    embedding_model = get_embedder()
    llm = LocalLLM(model=llm_model, provider=llm_provider, api_key=llm_api_key, endpoint=llm_endpoint)
    rag_retriever = RAGRetriever(context.store, embedding_model)
    
    if use_crew:
        try:
            from .crew_chat import CrewChatAgent
        except ImportError:
            console.print("\n  [red]CrewAI is not installed.[/red]")
            console.print("  [dim]Install with: pip install codegraph-cli\\[crew][/dim]\n")
            raise typer.Exit(1)
        console.print("\n  [magenta]Initializing CrewAI multi-agent system...[/magenta]")
        agent = CrewChatAgent(context, llm, rag_retriever)
    else:
        orchestrator = MCPOrchestrator(
            context.store,
            llm_model=llm_model,
            llm_provider=llm_provider,
            llm_api_key=llm_api_key,
            llm_endpoint=llm_endpoint
        )
        agent = ChatAgent(context, llm, orchestrator, rag_retriever)
    
    session_manager = SessionManager()
    
    # Gather auto-context from the project for enriched chat experience
    auto_context = _gather_auto_context(context)
    
    # Force new session if requested
    effective_session_id = None if new_session else session_id
    
    try:
        start_chat_repl(
            agent, session_manager, project, effective_session_id,
            use_crew=use_crew,
            provider=llm_provider,
            model=llm_model,
            auto_context=auto_context,
        )
    finally:
        context.close()


@chat_app.command("list")
def list_sessions(
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Filter by project")
):
    """📋 List all chat sessions.

    Example:
      cg chat list
      cg chat list --project my-api
    """
    session_manager = SessionManager()
    sessions = session_manager.list_sessions(project_name=project)
    
    if not sessions:
        console.print("\n  [dim]No chat sessions found.[/dim]\n")
        return
    
    console.print(f"\n  [bold cyan]📋 Chat Sessions ({len(sessions)})[/bold cyan]")
    console.print(Rule(style="dim"))
    
    for i, sess in enumerate(sessions, 1):
        created = sess['created_at'][:16].replace("T", " ")
        msgs = sess['message_count']
        proj = sess['project_name']
        sid = sess['id'][:8]
        
        console.print(f"  [white]{i}.[/white] [bold]{proj}[/bold]  [dim]({msgs} msgs, {created})[/dim]  [dim]id:{sid}…[/dim]")
    
    console.print()


@chat_app.command("delete")
def delete_session(
    session_id: str = typer.Argument(..., help="Session ID to delete")
):
    """🗑️  Delete a chat session.

    Example:
      cg chat delete abc12345
    """
    session_manager = SessionManager()
    
    if session_manager.delete_session(session_id):
        _print_status("✅", f"Deleted session {session_id[:8]}…")
    else:
        console.print("  [red]❌ Session not found[/red]")
        raise typer.Exit(1)


@chat_app.command("export")
def export_session(
    session_id: str = typer.Argument(..., help="Session ID to export"),
    fmt: str = typer.Option("markdown", "--format", "-f", help="Export format: markdown, json"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
):
    """📤 Export chat session to a file.

    Example:
      cg chat export abc12345
      cg chat export abc12345 --format json
      cg chat export abc12345 --output conversation.md
    """
    import json as _json

    session_manager = SessionManager()
    session = session_manager.load_session(session_id)

    if not session:
        console.print(f"  [red]❌ Session '{session_id}' not found[/red]")
        raise typer.Exit(1)

    if fmt == "markdown":
        lines = [
            f"# Chat Session: {session.project_name}",
            f"\nSession ID: {session.id}",
            f"Messages: {session.message_count}\n",
            "---\n",
        ]
        for msg in session.messages:
            role = "**You**" if msg["role"] == "user" else "**CodeGraph**"
            lines.append(f"### {role}\n")
            lines.append(f"{msg['content']}\n")
            lines.append("\n---\n")
        content = "\n".join(lines)
        ext = ".md"

    elif fmt == "json":
        data = {
            "session_id": session.id,
            "project_name": session.project_name,
            "message_count": session.message_count,
            "messages": session.messages,
        }
        content = _json.dumps(data, indent=2)
        ext = ".json"

    else:
        console.print(f"  [red]❌ Unknown format: {fmt}[/red]")
        raise typer.Exit(1)

    filename = output or f"{session_id[:12]}_export{ext}"
    Path(filename).write_text(content)
    _print_status("✅", f"Exported to {filename}")
