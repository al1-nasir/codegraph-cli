"""CrewAI agents for CodeGraph CLI — specialized multi-agent system."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

try:
    from crewai import Agent
    CREWAI_AVAILABLE = True
except ImportError:
    Agent = None  # type: ignore
    CREWAI_AVAILABLE = False

if TYPE_CHECKING:
    from .crew_tools import create_tools


def create_file_ops_agent(tools: list, llm, project_context: str = "") -> Agent:
    """File system agent — reads, writes, patches, and manages files + backups."""
    ctx = f"\n\nCurrent Project: {project_context}" if project_context else ""
    return Agent(
        role="File System Engineer",
        goal=(
            "Handle all file operations: read files, write new files, patch existing code, "
            "delete files, and manage backups/rollbacks. Always create backups before modifying files."
        ),
        backstory=(
            "You are an expert file system engineer. You navigate project directories, "
            "read source code, write and patch files precisely. You ALWAYS create a backup "
            "before modifying any file so changes can be rolled back. When writing code, you "
            "produce complete, working, well-formatted files. When patching, you use exact "
            "text matching to make surgical edits. You use the file_tree tool to understand "
            f"project structure before making changes.{ctx}"
        ),
        tools=tools,
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=15,
    )


def create_code_gen_agent(tools: list, llm, project_context: str = "") -> Agent:
    """Code generation & refactoring agent — writes, modifies, and improves code."""
    ctx = f"\n\nCurrent Project: {project_context}" if project_context else ""
    return Agent(
        role="Senior Software Developer",
        goal=(
            "Generate high-quality code, implement features, refactor existing code, "
            "and fix bugs. Read existing code first to match project style. Always use "
            "patch_file for edits and write_file for new files."
        ),
        backstory=(
            "You are a senior software developer with deep expertise in Python, JavaScript, "
            "TypeScript, and system design. You follow these principles:\n"
            "1. ALWAYS read the existing file before modifying it\n"
            "2. Use patch_file for targeted edits (preferred) or write_file for new/rewritten files\n"
            "3. Match the existing code style, imports, and patterns\n"
            "4. Include proper error handling, type hints, and docstrings\n"
            "5. Use search_code and grep_in_project to understand how code is used before changing it\n"
            "6. When asked to improve/refactor, explain what you changed and why\n"
            f"7. Generate complete, runnable code — never leave TODO placeholders{ctx}"
        ),
        tools=tools,
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=20,
    )


def create_code_analysis_agent(tools: list, llm, project_context: str = "") -> Agent:
    """Code analysis agent — searches, understands, and explains code."""
    ctx = f"\n\nCurrent Project: {project_context}" if project_context else ""
    return Agent(
        role="Code Intelligence Analyst",
        goal=(
            "Search, analyze, and explain code. Find relevant functions, trace dependencies, "
            "understand how things connect, and provide clear explanations."
        ),
        backstory=(
            "You are a code analysis expert. You use semantic search (search_code) to find "
            "relevant code by meaning, and grep_in_project to find exact text patterns. "
            "You read files to understand implementation details, trace call chains, and "
            "explain complex code clearly. When analyzing:\n"
            "1. Use search_code for finding code by concept/meaning\n"
            "2. Use grep_in_project for finding exact function/class usage\n"
            "3. Use read_file to get full context of interesting results\n"
            "4. Use get_project_summary and file_tree for structural overview\n"
            f"5. Provide clear, specific answers — never guess{ctx}"
        ),
        tools=tools,
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=15,
    )


def create_coordinator_agent(llm, project_context: str = "") -> Agent:
    """Coordinator agent — routes tasks to the right specialist."""
    ctx = f" Current Project: {project_context}." if project_context else ""
    return Agent(
        role="Project Coordinator",
        goal=(
            "Understand user requests and coordinate specialist agents. Route file operations "
            "to File System Engineer, code changes to Senior Software Developer, and analysis "
            "to Code Intelligence Analyst."
        ),
        backstory=(
            f"You are a project coordinator managing a team of AI specialists.{ctx}\n\n"
            "Your team:\n"
            "• File System Engineer — reads/writes/patches files, manages backups & rollbacks\n"
            "• Senior Software Developer — generates code, implements features, refactors\n"
            "• Code Intelligence Analyst — searches code, analyzes dependencies, explains logic\n\n"
            "RULES:\n"
            "1. ALWAYS delegate to the right specialist — never try to do tasks yourself\n"
            "2. For 'what is in this project' / 'show files' → delegate to File System Engineer\n"
            "3. For 'write code' / 'add feature' / 'fix bug' / 'refactor' → delegate to Senior Software Developer\n"
            "4. For 'find' / 'search' / 'explain' / 'how does X work' → delegate to Code Intelligence Analyst\n"
            "5. For complex tasks, break them into steps and delegate each step\n"
            "6. Always return concrete answers based on actual project data from tools"
        ),
        llm=llm,
        verbose=False,
        allow_delegation=True,
        max_iter=10,
    )
