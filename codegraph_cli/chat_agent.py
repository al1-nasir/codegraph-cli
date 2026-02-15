"""Chat agent for interactive conversational coding assistance."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from .chat_session import SessionManager
from .codegen_agent import CodeGenAgent
from .context_manager import SymbolMemory, assemble_context_for_llm, detect_intent
from .llm import LocalLLM
from .models_v2 import ChatSession, CodeProposal
from .orchestrator import MCPOrchestrator
from .rag import RAGRetriever
from .refactor_agent import RefactorAgent
from .storage import GraphStore


SYSTEM_PROMPT = """You are an AI coding assistant integrated with CodeGraph CLI.

You have access to a semantic code graph of the user's project and can:
- Search for code semantically
- Analyze impact of changes
- Generate new code
- Refactor existing code
- Explain code functionality

When the user asks you to make changes:
1. Use the provided code context to understand existing patterns
2. Generate code that follows the project's style
3. Explain what you're doing
4. Create a proposal that the user can review before applying

Be concise, helpful, and always explain your reasoning.
"""


class ChatAgent:
    """Orchestrates interactive chat with RAG-based context management and real file access."""
    
    def __init__(
        self,
        context: ProjectContext,
        llm: LocalLLM,
        orchestrator: MCPOrchestrator,
        rag_retriever: RAGRetriever
    ):
        """Initialize chat agent.
        
        Args:
            context: ProjectContext for real file access
            llm: LLM for generating responses
            orchestrator: Orchestrator for accessing other agents
            rag_retriever: RAG retriever for semantic search
        """
        self.context = context
        self.llm = llm
        self.orchestrator = orchestrator
        self.rag_retriever = rag_retriever
        self.session_manager = SessionManager()
        
        # Symbol memory — tracks recently discussed symbols & files
        # so we can skip redundant RAG queries.
        self.symbol_memory = SymbolMemory()
        
        # Initialize specialized agents
        from .codegen_agent import CodeGenAgent
        from .refactor_agent import RefactorAgent
        self.codegen_agent = CodeGenAgent(context.store, llm, project_context=context)
        self.refactor_agent = RefactorAgent(context.store)

        # Build enhanced system prompt with auto-context
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build system prompt enriched with project context.

        Includes project name, source path, indexed file/symbol counts,
        node-type breakdown, and recently modified files so the LLM has
        immediate awareness of the codebase.
        """
        base = SYSTEM_PROMPT

        try:
            summary = self.context.get_project_summary()
            parts = [
                "\n\nProject Context:",
                f"- Project: {summary.get('project_name', 'unknown')}",
                f"- Source: {summary.get('source_path', 'N/A')}",
                f"- Indexed: {summary.get('indexed_files', 0)} files, {summary.get('total_nodes', 0)} symbols",
            ]

            node_types = summary.get("node_types", {})
            if node_types:
                parts.append(
                    f"- Breakdown: {node_types.get('function', 0)} functions, "
                    f"{node_types.get('class', 0)} classes, "
                    f"{node_types.get('module', 0)} modules"
                )

            # Recently modified files
            if self.context.has_source_access:
                try:
                    items = self.context.list_directory(".")
                    files = [f for f in items if f["type"] == "file"]
                    files.sort(key=lambda f: f.get("modified", ""), reverse=True)
                    recent = [f["name"] for f in files[:5]]
                    if recent:
                        parts.append(f"- Recently modified: {', '.join(recent)}")
                except Exception:
                    pass

            return base + "\n".join(parts)
        except Exception:
            return base
    
    def process_message(
        self,
        user_message: str,
        session: ChatSession
    ) -> str:
        """Process user message and generate response.
        
        Note: The caller (REPL) is responsible for adding messages to
        the session.  This method does NOT add messages itself to avoid
        duplicate entries.
        
        Args:
            user_message: User's message
            session: Current chat session
            
        Returns:
            Assistant's response
        """
        # Detect intent
        intent = detect_intent(user_message)
        
        # Handle special intents with existing agents
        if intent == "read":
            response = self._handle_read(user_message)
        elif intent == "list":
            response = self._handle_list()
        elif intent == "search":
            response = self._handle_search(user_message)
        elif intent == "impact":
            response = self._handle_impact(user_message)
        elif intent == "generate":
            response = self._handle_generate(user_message, session)
        elif intent == "refactor":
            response = self._handle_refactor(user_message, session)
        else:
            # General chat - use LLM with RAG context
            response = self._handle_chat(user_message, session)
        
        # Save session
        self.session_manager.save_session(session)
        
        return response
    
    def _handle_list(self) -> str:
        """Handle list files intent - show actual project files."""
        if not self.context.has_source_access:
            return (
                "❌ Source directory not available for this project.\n"
                "Re-index the project to enable file access: cg index <path> --name <project>"
            )
        
        try:
            items = self.context.list_directory(".")
            
            if not items:
                return "📁 Project directory is empty."
            
            # Separate files and directories
            dirs = [item for item in items if item["type"] == "dir"]
            files = [item for item in items if item["type"] == "file"]
            
            response_parts = [f"📁 Project: {self.context.project_name}"]
            response_parts.append(f"📂 Location: {self.context.source_path}\n")
            
            # Show directories
            if dirs:
                response_parts.append(f"Directories ({len(dirs)}):")
                for d in dirs[:10]:  # Limit to 10
                    response_parts.append(f"  📂 {d['name']}/")
                if len(dirs) > 10:
                    response_parts.append(f"  ... and {len(dirs) - 10} more")
                response_parts.append("")
            
            # Show files
            if files:
                response_parts.append(f"Files ({len(files)}):")
                for f in files[:15]:  # Limit to 15
                    size_kb = f['size'] / 1024 if f['size'] else 0
                    response_parts.append(f"  📄 {f['name']} ({size_kb:.1f} KB)")
                if len(files) > 15:
                    response_parts.append(f"  ... and {len(files) - 15} more")
            
            # Show indexed files info
            indexed_files = self.context.get_indexed_files()
            response_parts.append(f"\n✅ Indexed: {len(indexed_files)} Python file(s)")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            return f"❌ Error listing directory: {str(e)}"
    
    def _handle_read(self, message: str) -> str:
        """Handle read file intent - show file contents."""
        if not self.context.has_source_access:
            return (
                "❌ Source directory not available for this project.\n"
                "Re-index the project to enable file access: cg index <path> --name <project>"
            )
        
        # Extract filename from message
        import re
        # Look for common file extensions
        file_pattern = r'([\w/.-]+\.(?:py|txt|md|json|yaml|yml|toml|cfg|ini|sh|js|ts|html|css))'
        matches = re.findall(file_pattern, message, re.IGNORECASE)
        
        if not matches:
            return "❌ Please specify a file to read (e.g., 'show me basic.py')"
        
        filename = matches[0]
        
        try:
            content = self.context.read_file(filename)
            
            # Detect language for syntax highlighting
            ext = filename.split('.')[-1]
            lang_map = {
                'py': 'python', 'js': 'javascript', 'ts': 'typescript',
                'md': 'markdown', 'json': 'json', 'yaml': 'yaml', 'yml': 'yaml',
                'sh': 'bash', 'html': 'html', 'css': 'css'
            }
            lang = lang_map.get(ext, '')
            
            response_parts = [f"📄 {filename}\n"]
            response_parts.append(f"```{lang}")
            response_parts.append(content)
            response_parts.append("```")
            
            # Add file info
            file_info = self.context.get_file_info(filename)
            if file_info:
                size_kb = file_info['size'] / 1024 if file_info['size'] else 0
                response_parts.append(f"\n📊 Size: {size_kb:.1f} KB | Modified: {file_info['modified'][:10]}")
            
            return "\n".join(response_parts)
            
        except FileNotFoundError:
            return f"❌ File not found: {filename}\nAvailable files: {', '.join([f['name'] for f in self.context.list_directory('.') if f['type'] == 'file'][:5])}"
        except Exception as e:
            return f"❌ Error reading file: {str(e)}"
    
    def _handle_search(self, message: str) -> str:
        """Handle search intent."""
        # Extract query (remove search keywords)
        query = message.lower()
        for kw in ["find", "search", "where is", "show me", "locate"]:
            query = query.replace(kw, "").strip()
        
        # Use RAG agent
        results = self.rag_retriever.search(query, top_k=5)
        
        if not results:
            return "I couldn't find any matching code. Try a different search query."
        
        # Format results
        response_parts = [f"Found {len(results)} results:\n"]
        for i, result in enumerate(results, 1):
            response_parts.append(
                f"{i}. [{result.node_type}] {result.qualname}\n"
                f"   Location: {result.file_path}:{result.start_line}\n"
                f"   Relevance: {result.score:.2f}\n"
            )
        
        return "\n".join(response_parts)
    
    def _handle_impact(self, message: str) -> str:
        """Handle impact analysis intent."""
        # Extract symbol name
        import re
        symbols = re.findall(r'\b[A-Z][a-zA-Z0-9_]*\b|\b[a-z_][a-z0-9_]*\b', message)
        
        if not symbols:
            return "Please specify a function or class name to analyze impact."
        
        symbol = symbols[0]
        
        # Use summarization agent for impact analysis
        try:
            report = self.orchestrator.summarization_agent.impact_analysis(symbol, hops=2)
            return f"**Impact Analysis for {symbol}:**\n\n{report.explanation}"
        except Exception as e:
            return f"Couldn't analyze impact: {str(e)}"
    
    def _handle_generate(self, message: str, session: ChatSession) -> str:
        """Handle code generation intent."""
        try:
            # Use codegen agent
            proposal = self.codegen_agent.generate(
                prompt=message,
                max_files=3
            )
            
            # Add to pending proposals
            session.pending_proposals.append(proposal)
            
            # Format response
            response_parts = [
                f"I've created a code proposal: {proposal.description}\n",
                f"Files to change: {proposal.num_files_changed}",
                f"  - New files: {proposal.num_files_created}",
                f"  - Modified files: {proposal.num_files_modified}\n",
                "To apply these changes, say 'apply' or '/apply'.",
                "To see the diff, say 'show diff' or '/preview'."
            ]
            
            return "\n".join(response_parts)
        except Exception as e:
            return f"I encountered an error generating code: {str(e)}\n\nCould you provide more details?"
    
    def _handle_refactor(self, message: str, session: ChatSession) -> str:
        """Handle refactoring intent."""
        # For now, provide guidance
        return (
            "I can help with refactoring! Here are some operations I support:\n\n"
            "- Rename a symbol: 'Rename process_payment to handle_payment'\n"
            "- Extract function: 'Extract lines 45-60 into a new function'\n"
            "- Extract service: 'Move payment functions to a new service'\n\n"
            "What would you like to refactor?"
        )
    
    def _handle_chat(self, message: str, session: ChatSession) -> str:
        """Handle general chat with LLM and RAG context."""
        # Assemble context using smart RAG strategy + symbol memory
        context_messages = assemble_context_for_llm(
            user_message=message,
            session=session,
            rag_retriever=self.rag_retriever,
            system_prompt=self.system_prompt,
            max_tokens=8000,
            symbol_memory=self.symbol_memory,
        )
        
        # Call LLM
        response = self.llm.chat_completion(
            messages=context_messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        if response:
            return response
        
        # Fallback if LLM fails
        return (
            "I'm having trouble connecting to the LLM. "
            "Try asking me to search for code, analyze impact, or generate code."
        )
    
    def apply_pending_proposal(self, session: ChatSession, proposal_index: int = 0) -> str:
        """Apply a pending proposal.
        
        Args:
            session: Chat session
            proposal_index: Index of proposal to apply (default: most recent)
            
        Returns:
            Result message
        """
        if not session.pending_proposals:
            return "No pending proposals to apply."
        
        if proposal_index >= len(session.pending_proposals):
            return f"Invalid proposal index. You have {len(session.pending_proposals)} pending proposal(s)."
        
        proposal = session.pending_proposals[proposal_index]
        
        try:
            # Apply using codegen agent
            result = self.codegen_agent.apply_changes(proposal, backup=True)
            
            if result.success:
                # Remove from pending
                session.pending_proposals.pop(proposal_index)
                self.session_manager.save_session(session)
                
                return (
                    f"✅ Successfully applied changes to {len(result.files_changed)} file(s).\n"
                    f"Backup ID: {result.backup_id}\n"
                    f"You can rollback with: cg v2 rollback {result.backup_id}"
                )
            else:
                return f"❌ Failed to apply changes: {result.error}"
        except Exception as e:
            return f"❌ Error applying changes: {str(e)}"
