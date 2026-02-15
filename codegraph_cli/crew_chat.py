"""CrewAI-based chat agent with multi-agent collaboration + rollback."""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Dict, List

from datetime import datetime

try:
    from crewai import Agent, Crew, Task
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

from .crew_agents import (
    create_code_analysis_agent,
    create_code_gen_agent,
    create_coordinator_agent,
    create_file_ops_agent,
)
from .crew_tools import create_tools, list_backups, rollback_file
from .llm import LocalLLM, create_crewai_llm

if TYPE_CHECKING:
    from .project_context import ProjectContext
    from .rag import RAGRetriever


class CrewChatAgent:
    """CrewAI-based chat agent with multi-agent collaboration and rollback."""

    def __init__(
        self,
        context: ProjectContext,
        llm: LocalLLM,
        rag_retriever: RAGRetriever,
    ):
        self.context = context
        self.llm = llm
        self.rag_retriever = rag_retriever

        # ── Silence CrewAI / LiteLLM noise ────────────────
        for logger_name in (
            "crewai", "crewai.agents", "crewai.crews",
            "crewai.tasks", "litellm", "httpx",
        ):
            logging.getLogger(logger_name).setLevel(logging.CRITICAL)
        os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")

        # ── Provider env vars for LiteLLM compatibility ───
        if llm.api_key:
            provider = (llm.provider_name or "").lower()
            if provider == "openrouter" or (llm.endpoint and "openrouter.ai" in llm.endpoint):
                os.environ["OPENROUTER_API_KEY"] = llm.api_key
                if llm.endpoint:
                    os.environ["OPENAI_API_BASE"] = llm.endpoint
            elif provider == "gemini":
                os.environ["GEMINI_API_KEY"] = llm.api_key
            elif provider == "anthropic":
                os.environ["ANTHROPIC_API_KEY"] = llm.api_key
            elif provider == "openai":
                os.environ["OPENAI_API_KEY"] = llm.api_key
            elif provider == "groq":
                os.environ["GROQ_API_KEY"] = llm.api_key

        # Suppress warnings
        import warnings
        warnings.filterwarnings("ignore", module="crewai")

        import contextlib
        self._stderr_suppress = contextlib.redirect_stderr(open(os.devnull, "w"))
        self._stderr_suppress.__enter__()

        # ── Create tools & agents ─────────────────────────
        tools = create_tools(context, rag_retriever)
        crew_llm = create_crewai_llm(llm)

        proj_summary = context.get_project_summary()
        project_ctx = (
            f"{proj_summary.get('project_name', 'Unknown')} "
            f"at {proj_summary.get('source_path', '.')} "
            f"with {proj_summary.get('indexed_files', 0)} indexed files"
        )

        self.file_ops_agent = create_file_ops_agent(tools["file_ops"], crew_llm, project_ctx)
        self.code_gen_agent = create_code_gen_agent(tools["all"], crew_llm, project_ctx)
        self.code_analysis_agent = create_code_analysis_agent(tools["code_analysis"], crew_llm, project_ctx)
        # Give coordinator ALL tools so it can read files, search code, and delegate
        self.coordinator = create_coordinator_agent(crew_llm, project_ctx, tools=tools["all"])

        # ── Conversation memory for multi-turn continuity ─
        self._history: list[dict] = []
        self._max_history_pairs: int = 10

    # ── Public API ────────────────────────────────────────

    def process_message(self, user_message: str) -> str:
        """Process a user message via CrewAI multi-agent pipeline.

        Injects conversation history into the task description so agents
        can follow up on previous suggestions and requests.
        """
        try:
            proj_summary = self.context.get_project_summary()

            # ── Build context with project info + conversation history ──
            context_parts = [
                f"Project: {proj_summary.get('project_name', 'Unknown')}, "
                f"Root: {proj_summary.get('source_path', '.')}, "
                f"Indexed files: {proj_summary.get('indexed_files', 0)}",
            ]

            history_block = self._format_history()
            if history_block:
                context_parts.append(
                    "\n── PREVIOUS CONVERSATION (use this to understand follow-up requests) ──\n"
                    f"{history_block}\n"
                    "── END PREVIOUS CONVERSATION ──"
                )

            context_parts.append(f"\nCurrent User Request: {user_message}")

            task = Task(
                description="\n".join(context_parts),
                expected_output=(
                    "Concrete results based on actual project files.\n"
                    "CRITICAL: If the user refers to something from the previous conversation "
                    "(e.g. 'apply those changes', 'implement what you suggested', 'do it', "
                    "'make those improvements'), you MUST look at the PREVIOUS CONVERSATION "
                    "section above to understand exactly what was discussed, then ACT on it "
                    "using write_file or patch_file tools.\n"
                    "For code changes: MUST use write_file or patch_file tools to actually "
                    "modify files. Don't just describe changes — actually apply them and "
                    "confirm they were applied. After making changes, read the file to verify."
                ),
                agent=self.coordinator,
            )

            crew = Crew(
                agents=[
                    self.coordinator,
                    self.file_ops_agent,
                    self.code_gen_agent,
                    self.code_analysis_agent,
                ],
                tasks=[task],
                verbose=False,
                process="sequential",
            )

            result = crew.kickoff()
            raw = str(result.raw) if hasattr(result, "raw") else str(result)
            response = self._clean_response(raw)

            # ── Store in conversation memory ──
            self._history.append({"role": "user", "content": user_message})
            self._history.append({"role": "assistant", "content": response})
            self._trim_history()

            return response

        except Exception as e:
            return f"❌ Error: {e}\n\nPlease try rephrasing your request."

    # ── Response sanitisation ─────────────────────────────

    @staticmethod
    def _clean_response(text: str) -> str:
        """Strip LLM reasoning artifacts from the crew output.

        Some models (e.g. DeepSeek, StepFun) put their real answer inside
        <think>…</think> tags and only emit a tiny fragment afterwards.
        If stripping <think> would leave less than 100 useful chars, we
        extract the *content* of the think block as the real answer.
        """
        # ── Phase 1: Handle <think> blocks smartly ────────
        think_match = re.search(r"<think>([\s\S]*?)</think>", text)
        if think_match:
            think_content = think_match.group(1).strip()
            without_think = re.sub(r"<think>[\s\S]*?</think>", "", text)
            without_think = re.sub(r"</?think>", "", without_think).strip()
            # If the part outside <think> is too short, the real answer
            # is inside the think block
            if len(without_think) < 100 and len(think_content) > len(without_think):
                text = think_content
            else:
                text = without_think
        else:
            # Stray opening / closing think tags (no matched pair)
            text = re.sub(r"</?think>", "", text)

        # ── Phase 2: Strip tool-call / XML artifacts ──────
        text = re.sub(r"<tool_call>[\s\S]*?</tool_call>", "", text)
        text = re.sub(r"</?tool_call>", "", text)
        text = re.sub(
            r"</?(?:function|parameter|result|output|observation)(?:=[^>]*)?>\s*",
            "", text,
        )

        # ── Phase 3: Clean up whitespace ──────────────────
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ── Conversation memory helpers ───────────────────────

    def _format_history(self) -> str:
        """Format conversation history for injection into task context.

        Keeps the last 4 exchanges (8 messages).  The most recent 2
        exchanges are preserved in full; older ones are compressed to
        ~800 chars each to stay within LLM context limits.
        """
        if not self._history:
            return ""

        # Show at most the last 8 messages (4 exchanges)
        recent = self._history[-8:]
        lines: list[str] = []
        total = len(recent)

        for i, msg in enumerate(recent):
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            # Last 2 exchanges (4 msgs) get generous space; older ones compressed
            is_recent = i >= total - 4
            max_len = 3000 if is_recent else 800
            if len(content) > max_len:
                content = content[:max_len] + "\n... (truncated)"
            lines.append(f"[{role}]:\n{content}")

        if len(self._history) > 8:
            lines.insert(0, f"... ({len(self._history) - 8} older messages omitted)")

        return "\n\n".join(lines)

    def _trim_history(self):
        """Keep only the last N exchange pairs to avoid unbounded growth."""
        max_messages = self._max_history_pairs * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    def clear_history(self):
        """Clear conversation memory (called on /clear and /new)."""
        self._history.clear()

    def load_session_history(self, session) -> None:
        """Load conversation history from a persisted ChatSession.

        Called when resuming a previous session so the crew retains
        context from earlier exchanges.
        """
        self._history.clear()
        for msg in session.messages:
            role = msg.role if hasattr(msg, "role") else msg["role"]
            content = msg.content if hasattr(msg, "content") else msg["content"]
            self._history.append({"role": role, "content": content})
        self._trim_history()

    # ── Rollback helpers (called from REPL) ───────────────

    def list_all_backups(self) -> List[Dict]:
        """Return all available file backups."""
        return list_backups()

    def rollback(self, file_path: str, timestamp: str | None = None) -> str:
        """Rollback a file to its backup.

        Args:
            file_path: The file to rollback (can be relative or absolute).
            timestamp: Specific backup timestamp, or None for latest.

        Returns:
            Status message.
        """
        return rollback_file(file_path, timestamp)

    # ── Stats ─────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "agents": {
                "coordinator": self.coordinator.role,
                "file_ops": self.file_ops_agent.role,
                "code_gen": self.code_gen_agent.role,
                "code_analysis": self.code_analysis_agent.role,
            },
            "project": self.context.project_name,
            "has_source_access": self.context.has_source_access,
        }
