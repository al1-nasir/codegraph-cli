"""Interactive chat CLI for conversational coding assistance."""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

from . import config
from .chat_agent import ChatAgent
from .chat_session import SessionManager
from .crew_chat import CrewChatAgent
from .llm import LocalLLM
from .orchestrator import MCPOrchestrator
from .rag import RAGRetriever
from .storage import GraphStore, ProjectManager


# ── Theme colors ──────────────────────────────────────────────
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_CYAN = "\033[36m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_MAGENTA = "\033[35m"
C_RED = "\033[31m"
C_BLUE = "\033[34m"
C_WHITE = "\033[97m"
C_BG_DARK = "\033[48;5;235m"


def _term_width() -> int:
    """Get terminal width, default 80."""
    return shutil.get_terminal_size((80, 24)).columns


def _box(text: str, color: str = C_CYAN, width: int = 0) -> str:
    """Draw a box around text."""
    w = width or min(_term_width(), 70)
    inner = w - 4
    lines = text.split("\n")
    out = [f"{color}╭{'─' * (w - 2)}╮{C_RESET}"]
    for line in lines:
        padded = line.ljust(inner)[:inner]
        out.append(f"{color}│{C_RESET} {padded} {color}│{C_RESET}")
    out.append(f"{color}╰{'─' * (w - 2)}╯{C_RESET}")
    return "\n".join(out)


def _divider(char: str = "─", color: str = C_DIM) -> str:
    w = min(_term_width(), 70)
    return f"{color}{char * w}{C_RESET}"


def _print_welcome(project_name: str, use_crew: bool, provider: str, model: str):
    """Print the modern welcome banner."""
    mode = "CrewAI Multi-Agent" if use_crew else "Chat"
    
    banner = (
        f"  {C_BOLD}{C_CYAN}⚡ CodeGraph {mode}{C_RESET}\n"
        f"  {C_DIM}Project: {C_WHITE}{project_name}{C_RESET}\n"
        f"  {C_DIM}LLM:     {C_WHITE}{provider}/{model}{C_RESET}"
    )
    print(f"\n{_box(f'⚡ CodeGraph {mode}', C_CYAN)}")
    print(f"  {C_DIM}Project  {C_RESET}{C_WHITE}{project_name}{C_RESET}")
    print(f"  {C_DIM}LLM      {C_RESET}{C_WHITE}{provider}/{model}{C_RESET}")
    print(f"  {C_DIM}Type {C_YELLOW}/help{C_DIM} for commands, {C_YELLOW}/exit{C_DIM} to quit{C_RESET}")
    print(_divider())
    print()


def _print_help(use_crew: bool):
    """Print help with styled formatting."""
    print(f"\n  {C_BOLD}{C_CYAN}📖 Commands{C_RESET}")
    print(_divider("─", C_DIM))
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
        print(f"  {C_YELLOW}{cmd:<22}{C_RESET}{C_DIM}{desc}{C_RESET}")
    print()


def _print_response(text: str):
    """Print assistant response with styling."""
    print(f"\n  {C_GREEN}●{C_RESET} {C_BOLD}Assistant{C_RESET}")
    # Indent each line for clean look
    for line in text.split("\n"):
        print(f"  {line}")
    print()


def _print_status(emoji: str, msg: str, color: str = C_GREEN):
    """Print a status message."""
    print(f"  {color}{emoji}  {msg}{C_RESET}")


def start_chat_repl(
    agent,  # Can be ChatAgent or CrewChatAgent
    session_manager: SessionManager,
    project_name: str,
    session_id: Optional[str] = None,
    use_crew: bool = False,
    provider: str = "",
    model: str = "",
):
    """Start interactive REPL for chat."""
    # Load or create session
    if session_id:
        session = session_manager.load_session(session_id)
        if not session:
            _print_status("🆕", "Session not found. Starting new session.", C_YELLOW)
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
    
    # Welcome
    _print_welcome(project_name, use_crew, provider, model)
    
    # REPL loop
    while True:
        try:
            # Prompt
            try:
                user_input = input(f"  {C_BLUE}●{C_RESET} {C_BOLD}You ›{C_RESET} ").strip()
            except EOFError:
                print(f"\n  {C_DIM}👋 Goodbye! Session saved.{C_RESET}\n")
                break
            
            if not user_input:
                continue
            
            # ── Handle commands ──────────────────────────────
            if user_input.startswith("/"):
                cmd = user_input.lower().split()[0]
                
                if cmd == "/exit":
                    session_manager.save_session(session)
                    print(f"\n  {C_DIM}👋 Goodbye! Session saved.{C_RESET}\n")
                    break
                
                elif cmd == "/clear":
                    session.clear_history()
                    session.clear_proposals()
                    session_manager.save_session(session)
                    _print_status("🧹", "Conversation cleared. Fresh start!", C_GREEN)
                    print()
                    continue
                
                elif cmd == "/new":
                    session_manager.save_session(session)
                    session = session_manager.create_session(project_name)
                    _print_status("🆕", "New session started.", C_GREEN)
                    print()
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
                            _print_status("📋", "No pending proposals.", C_YELLOW)
                        continue
                    elif cmd == "/preview":
                        if session.pending_proposals:
                            for i, prop in enumerate(session.pending_proposals):
                                print(f"\n  {C_BOLD}Proposal {i+1}:{C_RESET} {prop.description}")
                                for ch in prop.changes:
                                    icon = {"create": "🆕", "modify": "✏️", "delete": "🗑️"}.get(ch.change_type, "📄")
                                    print(f"    {icon} {ch.file_path}")
                        else:
                            _print_status("📋", "No pending proposals.", C_YELLOW)
                        print()
                        continue
                
                elif use_crew:
                    if cmd in ("/rollback", "/undo"):
                        parts = user_input.split(maxsplit=1)
                        if len(parts) < 2:
                            _print_status("❓", "Usage: /rollback <file_path> [timestamp]", C_YELLOW)
                            continue
                        args = parts[1].split()
                        file_path = args[0]
                        ts = args[1] if len(args) > 1 else None
                        result = agent.rollback(file_path, ts)
                        _print_status("⏪", result)
                        print()
                        continue
                    
                    elif cmd == "/backups":
                        backups = agent.list_all_backups()
                        if not backups:
                            _print_status("📦", "No backups found.", C_YELLOW)
                        else:
                            print(f"\n  {C_BOLD}{C_CYAN}📦 File Backups{C_RESET}")
                            print(_divider("─", C_DIM))
                            for b in backups:
                                ts = b["timestamp"]
                                fp = b["original_path"]
                                print(f"  {C_WHITE}{ts}{C_RESET}  {C_DIM}{fp}{C_RESET}")
                            print(f"\n  {C_DIM}Use /rollback <file_path> to restore{C_RESET}")
                        print()
                        continue
                
                _print_status("❓", f"Unknown command: {cmd}. Type /help", C_YELLOW)
                continue
            
            # ── Process message ──────────────────────────────
            session.add_message("user", user_input, datetime.now().isoformat())
            
            # Show thinking indicator
            print(f"\n  {C_DIM}⏳ Thinking...{C_RESET}", end="", flush=True)
            
            if use_crew:
                response = agent.process_message(user_input)
            else:
                response = agent.process_message(user_input, session)
            
            # Clear thinking indicator
            print(f"\r{' ' * 30}\r", end="")
            
            # Save & display
            session.add_message("assistant", response, datetime.now().isoformat())
            session_manager.save_session(session)
            
            _print_response(response)
            
        except KeyboardInterrupt:
            session_manager.save_session(session)
            print(f"\n\n  {C_DIM}👋 Goodbye! Session saved.{C_RESET}\n")
            break
        except Exception as e:
            print(f"\n  {C_RED}❌ Error: {str(e)}{C_RESET}\n")


# ── Typer app ────────────────────────────────────────────────
chat_app = typer.Typer(help="💬 Interactive chat with AI agents")


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
    """Start interactive chat session."""
    from .embeddings import get_embedder
    from .project_context import ProjectContext
    
    pm = ProjectManager()
    project = pm.get_current_project()
    
    if not project:
        print(f"\n  {C_RED}❌ No project loaded.{C_RESET}")
        print(f"  {C_DIM}Use: cg load-project <name>  or  cg index <path>{C_RESET}\n")
        raise typer.Exit(1)
    
    # Initialize components
    context = ProjectContext(project, pm)
    embedding_model = get_embedder()
    llm = LocalLLM(model=llm_model, provider=llm_provider, api_key=llm_api_key, endpoint=llm_endpoint)
    rag_retriever = RAGRetriever(context.store, embedding_model)
    
    if use_crew:
        print(f"\n  {C_MAGENTA}🤖 Initializing CrewAI multi-agent system...{C_RESET}")
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
    
    # Force new session if requested
    effective_session_id = None if new_session else session_id
    
    try:
        start_chat_repl(
            agent, session_manager, project, effective_session_id,
            use_crew=use_crew,
            provider=llm_provider,
            model=llm_model,
        )
    finally:
        context.close()


@chat_app.command("list")
def list_sessions(
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Filter by project")
):
    """List all chat sessions."""
    session_manager = SessionManager()
    sessions = session_manager.list_sessions(project_name=project)
    
    if not sessions:
        print(f"\n  {C_DIM}No chat sessions found.{C_RESET}\n")
        return
    
    print(f"\n  {C_BOLD}{C_CYAN}📋 Chat Sessions ({len(sessions)}){C_RESET}")
    print(_divider())
    
    for i, sess in enumerate(sessions, 1):
        created = sess['created_at'][:16].replace("T", " ")
        msgs = sess['message_count']
        proj = sess['project_name']
        sid = sess['id'][:8]
        
        print(f"  {C_WHITE}{i}.{C_RESET} {C_BOLD}{proj}{C_RESET}  {C_DIM}({msgs} msgs, {created}){C_RESET}  {C_DIM}id:{sid}…{C_RESET}")
    
    print()


@chat_app.command("delete")
def delete_session(
    session_id: str = typer.Argument(..., help="Session ID to delete")
):
    """Delete a chat session."""
    session_manager = SessionManager()
    
    if session_manager.delete_session(session_id):
        _print_status("✅", f"Deleted session {session_id[:8]}…")
    else:
        print(f"  {C_RED}❌ Session not found{C_RESET}")
        raise typer.Exit(1)
