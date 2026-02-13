"""CodeGenAgent for AI-powered code generation with impact analysis."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from .diff_engine import DiffEngine
from .llm import LocalLLM
from .models_v2 import ApplyResult, CodeProposal, FileChange
from .orchestrator import MCPOrchestrator
from .storage import GraphStore


class CodeGenAgent:
    """Generates code based on natural language requests with impact preview."""
    
    def __init__(
        self,
        store: GraphStore,
        llm: LocalLLM,
        diff_engine: Optional[DiffEngine] = None,
        project_context=None  # ProjectContext for file operations
    ):
        """Initialize CodeGenAgent.
        
        Args:
            store: Graph store for context
            llm: LLM for code generation
            diff_engine: Engine for managing diffs (optional)
            project_context: ProjectContext for file operations (optional)
        """
        self.store = store
        self.llm = llm
        self.diff_engine = diff_engine or DiffEngine()
        self.project_context = project_context
        # Pass provider name as string, not the provider object
        self.orchestrator = MCPOrchestrator(
            store,
            llm_model=llm.model,
            llm_provider=llm.provider_name,  # Fixed: use provider_name string
            llm_api_key=llm.api_key
        )
    
    def generate(
        self,
        prompt: str,
        context_file: Optional[str] = None,
        max_files: int = 3
    ) -> CodeProposal:
        """Generate code from natural language prompt.
        
        Args:
            prompt: Natural language description of what to generate
            context_file: Optional file to use as context
            max_files: Maximum number of context files to include
            
        Returns:
            CodeProposal with generated changes
        """
        # Gather context from codebase
        context = self._gather_context(prompt, context_file, max_files)
        
        # Generate code using LLM
        generation_prompt = self._build_generation_prompt(prompt, context)
        generated_code = self.llm.explain(generation_prompt)
        
        # Parse generated code into changes
        changes = self._parse_generated_code(generated_code, prompt)
        
        # Create proposal
        proposal = CodeProposal(
            id=str(uuid.uuid4()),
            description=prompt,
            changes=changes,
            metadata={"context_files": context.get("files", [])}
        )
        
        # Add diffs for modified files
        for change in proposal.changes:
            if change.change_type == "modify" and change.original_content and change.new_content:
                change.diff = self.diff_engine.create_diff(
                    change.original_content,
                    change.new_content,
                    change.file_path
                )
        
        return proposal
    
    def preview_impact(self, proposal: CodeProposal) -> str:
        """Analyze impact of proposed changes.
        
        Args:
            proposal: Code proposal to analyze
            
        Returns:
            Impact analysis summary
        """
        impact_lines = []
        
        # Analyze each change
        for change in proposal.changes:
            if change.change_type == "create":
                impact_lines.append(f"✨ New file: {change.file_path}")
            elif change.change_type == "modify":
                # Try to find affected symbols
                file_path = Path(change.file_path)
                if file_path.exists():
                    impact_lines.append(f"📝 Modified: {change.file_path}")
                    # Could add more detailed analysis here
            elif change.change_type == "delete":
                impact_lines.append(f"🗑️  Deleted: {change.file_path}")
        
        # Add recommendations
        impact_lines.append("")
        impact_lines.append("💡 Recommendations:")
        impact_lines.append("  - Review generated code for correctness")
        impact_lines.append("  - Run tests after applying changes")
        impact_lines.append("  - Backup is created automatically")
        
        return "\n".join(impact_lines)
    
    def apply_changes(
        self,
        proposal: CodeProposal,
        backup: bool = True
    ) -> ApplyResult:
        """Apply proposed changes to filesystem.
        
        Args:
            proposal: Proposal to apply
            backup: Whether to create backup
            
        Returns:
            Result of applying changes
        """
        return self.diff_engine.apply_changes(proposal, backup=backup)
    
    def _gather_context(
        self,
        prompt: str,
        context_file: Optional[str],
        max_files: int
    ) -> dict:
        """Gather relevant context from codebase.
        
        Args:
            prompt: User's request
            context_file: Specific file to include
            max_files: Max number of files to include
            
        Returns:
            Context dictionary
        """
        context = {"files": [], "snippets": []}
        
        # If specific file provided, use it
        if context_file:
            file_path = Path(context_file)
            if file_path.exists():
                context["files"].append({
                    "path": str(file_path),
                    "content": file_path.read_text()
                })
        
        # Search for relevant code
        try:
            search_results = self.orchestrator.search(prompt, top_k=max_files)
            for result in search_results[:max_files]:
                context["snippets"].append({
                    "qualname": result.qualname,
                    "file": result.file_path,
                    "code": result.code
                })
        except Exception:
            pass  # Continue even if search fails
        
        return context
    
    def _build_generation_prompt(self, user_prompt: str, context: dict) -> str:
        """Build prompt for LLM code generation.
        
        Args:
            user_prompt: User's request
            context: Gathered context
            
        Returns:
            Formatted prompt for LLM
        """
        prompt_parts = []
        
        prompt_parts.append("You are a code generation assistant. Generate clean, working code based on the user's request.")
        prompt_parts.append("")
        prompt_parts.append(f"User Request: {user_prompt}")
        prompt_parts.append("")
        
        # Add context if available
        if context.get("snippets"):
            prompt_parts.append("Relevant existing code:")
            for snippet in context["snippets"][:3]:
                prompt_parts.append(f"\n# {snippet['qualname']} ({snippet['file']})")
                prompt_parts.append(snippet["code"][:500])  # Limit length
            prompt_parts.append("")
        
        prompt_parts.append("Generate the requested code. Include:")
        prompt_parts.append("1. Complete, working implementation")
        prompt_parts.append("2. Proper error handling")
        prompt_parts.append("3. Clear docstrings")
        prompt_parts.append("4. Type hints where appropriate")
        prompt_parts.append("")
        prompt_parts.append("Output only the code, no explanations.")
        
        return "\n".join(prompt_parts)
    
    def _parse_generated_code(self, generated: str, prompt: str) -> list[FileChange]:
        """Parse LLM output into file changes.
        
        Args:
            generated: Generated code from LLM
            prompt: Original user prompt
            
        Returns:
            List of FileChange objects
        """
        # Simple implementation: treat as new file
        # In production, would parse file markers, detect modifications, etc.
        
        # Try to infer filename from prompt
        filename = self._infer_filename(prompt)
        
        return [
            FileChange(
                file_path=filename,
                change_type="create",
                new_content=generated.strip()
            )
        ]
    
    def _infer_filename(self, prompt: str) -> str:
        """Infer filename from user prompt.
        
        Args:
            prompt: User's request
            
        Returns:
            Suggested filename (relative to project source if context available)
        """
        # Simple heuristic: look for common patterns
        prompt_lower = prompt.lower()
        
        if "test" in prompt_lower:
            basename = "test_generated.py"
        elif "api" in prompt_lower or "endpoint" in prompt_lower:
            basename = "api_generated.py"
        elif "model" in prompt_lower:
            basename = "models_generated.py"
        else:
            basename = "generated.py"
        
        # If we have project context, prepend source path
        if self.project_context and self.project_context.has_source_access:
            return str(self.project_context.source_path / basename)
        else:
            return basename
