"""CrewAI-based chat agent with multi-agent collaboration + rollback."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Dict, List

from datetime import datetime

from crewai import Agent, Crew, Task

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
            provider = (llm.provider or "").lower()
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
        self.coordinator = create_coordinator_agent(crew_llm, project_ctx)

    # ── Public API ────────────────────────────────────────

    def process_message(self, user_message: str) -> str:
        """Process a user message via CrewAI multi-agent pipeline."""
        try:
            proj_summary = self.context.get_project_summary()
            context_str = (
                f"Project: {proj_summary.get('project_name', 'Unknown')}, "
                f"Root: {proj_summary.get('source_path', '.')}, "
                f"Indexed files: {proj_summary.get('indexed_files', 0)}\n\n"
            )

            task = Task(
                description=f"{context_str}User Request: {user_message}",
                expected_output=(
                    "A specific, concrete answer based on actual project files and code. "
                    "Use tools to explore the project. For code changes, use write_file or "
                    "patch_file tools which automatically create backups. "
                    "Don't give generic explanations."
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
            return str(result.raw) if hasattr(result, "raw") else str(result)

        except Exception as e:
            return f"❌ Error: {e}\n\nPlease try rephrasing your request."

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
