"""Smart context management for chat mode using RAG.

Includes:
- **RepoMap**: lightweight tree representation of the codebase (filenames +
  symbols) designed to fit inside an LLM context window for agentic planning
  *before* deep retrieval.
- **ConversationMemory**: sliding-window compression for chat history.
- Intent detection and query extraction utilities.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .models_v2 import ChatMessage, ChatSession, CodeProposal
from .rag import RAGRetriever

logger = logging.getLogger(__name__)

# Directories to skip when building the repo map
_REPO_MAP_SKIP: Set[str] = {
    ".venv", "venv", "__pycache__", "node_modules", ".git",
    "site-packages", ".tox", ".pytest_cache", "build", "dist",
    ".mypy_cache", ".ruff_cache", "htmlcov", ".eggs",
    "egg-info", ".codegraph", "lancedb", ".chroma",
}

# File extensions treated as "source code" in the repo map
_SOURCE_EXTENSIONS: Set[str] = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".go", ".rs", ".java", ".rb",
    ".cpp", ".c", ".cs", ".h", ".hpp",
}


# ===================================================================
# RepoMap – Agentic Context Feature
# ===================================================================

class RepoMap:
    """Generate a lightweight tree representation of a codebase.

    The map lists every file together with its top-level symbols (classes,
    functions) and is compact enough to inject into an LLM context window.
    This gives the model a *birds-eye view* of the repo **before** it
    performs targeted retrieval.

    Example output::

        codegraph_cli/parser.py
          class: TreeSitterParser
          class: ASTFallbackParser
          class: PythonGraphParser
          function: _resolve_call_edges
        codegraph_cli/embeddings.py
          class: TransformerEmbedder
          class: HashEmbeddingModel
          function: get_embedder
          function: cosine_similarity

    Usage::

        repo_map = RepoMap(project_root, parser=my_parser)
        context = repo_map.generate(max_tokens=4000)
    """

    def __init__(
        self,
        project_root: Path,
        parser: Optional[Any] = None,
    ) -> None:
        """
        Args:
            project_root: Root directory of the project.
            parser:       Optional :class:`~codegraph_cli.parser.Parser` instance.
                          When provided, the map includes symbols per file.
        """
        self.project_root = project_root
        self.parser = parser

    def generate(self, max_tokens: int = 4000) -> str:
        """Build the repo map string, truncated to *max_tokens*.

        Args:
            max_tokens: Approximate token budget (1 token ~ 4 chars).

        Returns:
            A compact, indented text representation of the repo structure.
        """
        tree_lines: List[str] = []

        for file_path in sorted(self.project_root.rglob("*")):
            if any(part in _REPO_MAP_SKIP for part in file_path.parts):
                continue
            if file_path.is_dir():
                continue

            rel = file_path.relative_to(self.project_root)

            # For source files, attempt to list symbols
            if file_path.suffix in _SOURCE_EXTENSIONS and self.parser is not None:
                try:
                    nodes, _ = self.parser.parse_file(file_path)
                    tree_lines.append(str(rel))
                    for node in nodes:
                        if node.node_type == "module":
                            continue
                        # Indent by nesting depth
                        depth = node.qualname.count(".") - str(rel).replace("/", ".").removesuffix(".py").count(".")
                        indent = "  " * max(depth, 1)
                        tree_lines.append(f"{indent}{node.node_type}: {node.name}")
                except Exception:
                    tree_lines.append(str(rel))
            else:
                tree_lines.append(str(rel))

        # Truncate to fit token budget
        result = "\n".join(tree_lines)
        char_budget = max_tokens * 4
        if len(result) > char_budget:
            result = result[:char_budget]
            # Clean cut at last newline
            last_nl = result.rfind("\n")
            if last_nl > 0:
                result = result[:last_nl]
            result += "\n... (truncated)"

        return result

    def generate_for_files(self, file_paths: List[Path]) -> str:
        """Build a focused repo map for a subset of files.

        Useful when the agent already knows which files are relevant.
        """
        tree_lines: List[str] = []
        for file_path in sorted(file_paths):
            if not file_path.exists():
                continue
            try:
                rel = file_path.relative_to(self.project_root)
            except ValueError:
                rel = file_path
            if file_path.suffix in _SOURCE_EXTENSIONS and self.parser is not None:
                try:
                    nodes, _ = self.parser.parse_file(file_path)
                    tree_lines.append(str(rel))
                    for node in nodes:
                        if node.node_type == "module":
                            continue
                        depth = node.qualname.count(".") - str(rel).replace("/", ".").removesuffix(".py").count(".")
                        indent = "  " * max(depth, 1)
                        tree_lines.append(f"{indent}{node.node_type}: {node.name}")
                except Exception:
                    tree_lines.append(str(rel))
            else:
                tree_lines.append(str(rel))
        return "\n".join(tree_lines)


class ConversationMemory:
    """Manages conversation history with compression for token efficiency."""
    
    def __init__(self, max_recent: int = 3):
        """Initialize conversation memory.
        
        Args:
            max_recent: Number of recent messages to keep verbatim
        """
        self.max_recent = max_recent
        self.summary = ""
    
    def get_context_for_llm(
        self,
        session: ChatSession,
        token_budget: int = 1500
    ) -> List[Dict[str, str]]:
        """Get optimized conversation context for LLM.
        
        Args:
            session: Chat session with messages
            token_budget: Maximum tokens to use for conversation history
            
        Returns:
            List of message dicts for LLM
        """
        messages = session.messages
        
        if len(messages) <= self.max_recent:
            # All messages fit, return as-is
            return [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
        
        # Split into old and recent
        old_messages = messages[:-self.max_recent]
        recent_messages = messages[-self.max_recent:]
        
        # Summarize old messages if not already done
        if not self.summary and old_messages:
            self.summary = self._summarize_messages(old_messages)
        
        # Build context
        context = []
        
        # Add summary as system message
        if self.summary:
            context.append({
                "role": "system",
                "content": f"Previous conversation summary: {self.summary}"
            })
        
        # Add recent messages verbatim
        context.extend([
            {"role": msg.role, "content": msg.content}
            for msg in recent_messages
        ])
        
        return context
    
    def _summarize_messages(self, messages: List[ChatMessage]) -> str:
        """Summarize old messages to save tokens.
        
        Args:
            messages: Messages to summarize
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        for msg in messages:
            if msg.role == "user":
                # Extract intent from user messages
                content_preview = msg.content[:100].replace("\n", " ")
                summary_parts.append(f"User: {content_preview}")
            elif msg.role == "assistant":
                # Extract actions from assistant messages
                if "applied changes" in msg.content.lower():
                    summary_parts.append("Applied code changes")
                elif "proposal" in msg.content.lower():
                    summary_parts.append("Created code proposal")
                elif "refactor" in msg.content.lower():
                    summary_parts.append("Discussed refactoring")
        
        # Keep last 5 actions
        return " | ".join(summary_parts[-5:])


def detect_intent(message: str) -> str:
    """Detect user intent from message.
    
    Args:
        message: User message
        
    Returns:
        Intent string: list, search, generate, refactor, impact, explain, chat
    """
    message_lower = message.lower()
    
    # Read file intent (show/read specific file content)
    if any(kw in message_lower for kw in [
        "show me", "read", "what's in", "whats in", "what is in", "contents of",
        "display", "view", "open", "cat"
    ]) and any(ext in message_lower for ext in [".py", ".txt", ".md", ".json", ".yaml", ".toml"]):
        return "read"
    
    # List intent (list files, show files, what files in project)
    if any(kw in message_lower for kw in [
        "list", "show files", "what files", "all files", "files in",
        "what do we have", "what's here", "whats here", "list the things"
    ]):
        return "list"
    
    # Search intent
    if any(kw in message_lower for kw in ["find", "search", "where is", "show me", "locate"]):
        return "search"
    
    # Generate intent
    if any(kw in message_lower for kw in ["add", "create", "generate", "implement", "build", "make"]):
        return "generate"
    
    # Refactor intent
    if any(kw in message_lower for kw in ["refactor", "extract", "rename", "move", "reorganize"]):
        return "refactor"
    
    # Impact intent
    if any(kw in message_lower for kw in ["impact", "what breaks", "what depends", "who uses"]):
        return "impact"
    
    # Explain intent
    if any(kw in message_lower for kw in ["explain", "what does", "how does", "why", "describe"]):
        return "explain"
    
    # Default to chat
    return "chat"


def extract_queries_from_message(message: str, intent: str) -> List[str]:
    """Extract search queries from user message based on intent.
    
    Args:
        message: User message
        intent: Detected intent
        
    Returns:
        List of search queries
    """
    queries = []
    
    if intent == "search":
        # Direct search - use message as-is
        queries.append(message)
    
    elif intent == "generate":
        # Extract domain concepts
        # E.g., "Add password reset endpoint" -> ["password reset", "authentication", "email"]
        
        # Extract main concept (first few words after action verb)
        match = re.search(r'(?:add|create|implement|build|make)\s+(.+?)(?:\s+endpoint|\s+function|\s+class|$)', message, re.IGNORECASE)
        if match:
            main_concept = match.group(1).strip()
            queries.append(main_concept)
            
            # Add related concepts
            if "password" in message.lower():
                queries.extend(["authentication", "email sending", "token generation"])
            elif "payment" in message.lower():
                queries.extend(["payment processing", "transaction", "billing"])
            elif "user" in message.lower():
                queries.extend(["user management", "authentication", "registration"])
    
    elif intent == "refactor":
        # Extract target symbols
        # E.g., "Refactor payment processing" -> ["payment processing", "payment service"]
        match = re.search(r'(?:refactor|extract|rename)\s+(.+?)(?:\s+into|\s+to|$)', message, re.IGNORECASE)
        if match:
            target = match.group(1).strip()
            queries.append(target)
            queries.append(f"{target} service")
    
    elif intent in ["impact", "explain"]:
        # Extract symbol names
        # Look for function/class names (CamelCase or snake_case)
        symbols = re.findall(r'\b[A-Z][a-zA-Z0-9_]*\b|\b[a-z_][a-z0-9_]*\b', message)
        if symbols:
            queries.append(symbols[0])  # Use first symbol
    
    # Fallback: use entire message
    if not queries:
        queries.append(message)
    
    return queries[:3]  # Max 3 queries


def count_tokens(text: str) -> int:
    """Approximate token count.
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Approximate token count
    """
    # Rough approximation: 1 token ≈ 4 characters
    return len(text) // 4


def assemble_context_for_llm(
    user_message: str,
    session: ChatSession,
    rag_retriever: RAGRetriever,
    system_prompt: str,
    max_tokens: int = 8000
) -> List[Dict[str, str]]:
    """Assemble optimized context for LLM within token budget.
    
    Args:
        user_message: Current user message
        session: Chat session with history
        rag_retriever: RAG retriever for code search
        system_prompt: System prompt
        max_tokens: Maximum total tokens
        
    Returns:
        List of message dicts for LLM
    """
    context = []
    token_count = 0
    
    # 1. System prompt (always included)
    context.append({"role": "system", "content": system_prompt})
    token_count += count_tokens(system_prompt)
    
    # 2. Detect intent
    intent = detect_intent(user_message)
    
    # 3. Extract queries and retrieve code
    queries = extract_queries_from_message(user_message, intent)
    code_snippets = []
    
    for query in queries:
        results = rag_retriever.search(query, top_k=3)
        # Filter by minimum score
        filtered = [r for r in results if r.score >= 0.15]
        code_snippets.extend(filtered)
    
    # Deduplicate by node_id
    seen = set()
    unique_snippets = []
    for snippet in sorted(code_snippets, key=lambda x: x.score, reverse=True):
        if snippet.node_id not in seen:
            seen.add(snippet.node_id)
            unique_snippets.append(snippet)
    
    # Limit to top 10
    unique_snippets = unique_snippets[:10]
    
    # 4. Add RAG context (high priority)
    if unique_snippets:
        rag_context = format_code_snippets(unique_snippets)
        rag_tokens = count_tokens(rag_context)
        
        if token_count + rag_tokens < max_tokens - 2000:
            context.append({
                "role": "system",
                "content": f"Relevant code from codebase:\n\n{rag_context}"
            })
            token_count += rag_tokens
    
    # 5. Add pending proposals if exist
    if session.pending_proposals:
        proposal_text = format_proposals(session.pending_proposals)
        proposal_tokens = count_tokens(proposal_text)
        
        if token_count + proposal_tokens < max_tokens - 1500:
            context.append({
                "role": "system",
                "content": f"Pending proposals:\n\n{proposal_text}"
            })
            token_count += proposal_tokens
    
    # 6. Add conversation history (compressed)
    conv_memory = ConversationMemory(max_recent=3)
    conv_context = conv_memory.get_context_for_llm(
        session,
        token_budget=max_tokens - token_count - 500
    )
    context.extend(conv_context)
    
    # 7. Add current user message
    context.append({"role": "user", "content": user_message})
    
    return context


def format_code_snippets(snippets: List) -> str:
    """Format code snippets for LLM context.
    
    Args:
        snippets: List of SearchResult objects
        
    Returns:
        Formatted string
    """
    blocks = []
    
    for snippet in snippets:
        blocks.append(
            f"[{snippet.node_type}] {snippet.qualname}\n"
            f"Location: {snippet.file_path}:{snippet.start_line}\n"
            f"Relevance: {snippet.score:.2f}\n"
            f"```python\n{snippet.snippet[:800]}\n```"
        )
    
    return "\n\n".join(blocks)


def format_proposals(proposals: List[CodeProposal]) -> str:
    """Format pending proposals for LLM context.
    
    Args:
        proposals: List of CodeProposal objects
        
    Returns:
        Formatted string
    """
    blocks = []
    
    for i, proposal in enumerate(proposals, 1):
        blocks.append(
            f"Proposal {i}: {proposal.description}\n"
            f"Files affected: {proposal.num_files_changed}\n"
            f"Status: Pending user approval"
        )
    
    return "\n\n".join(blocks)
