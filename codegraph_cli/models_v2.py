"""Data models for v2.0 code generation features."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class Location:
    """Represents a location in source code."""
    file_path: str
    line: int
    column: int = 0
    
    def __str__(self) -> str:
        return f"{self.file_path}:{self.line}"


@dataclass
class Range:
    """Represents a range in source code."""
    start: Location
    end: Location
    
    def __str__(self) -> str:
        if self.start.file_path == self.end.file_path:
            return f"{self.start.file_path}:{self.start.line}-{self.end.line}"
        return f"{self.start} to {self.end}"


@dataclass
class FileChange:
    """Represents changes to a single file."""
    file_path: str
    change_type: Literal["create", "modify", "delete"]
    original_content: Optional[str] = None
    new_content: Optional[str] = None
    diff: str = ""
    
    def __post_init__(self):
        """Validate change type constraints."""
        if self.change_type == "create" and self.original_content is not None:
            raise ValueError("Create changes should not have original_content")
        if self.change_type == "delete" and self.new_content is not None:
            raise ValueError("Delete changes should not have new_content")
        if self.change_type == "modify" and (self.original_content is None or self.new_content is None):
            raise ValueError("Modify changes must have both original and new content")


@dataclass
class CodeProposal:
    """Represents proposed code changes."""
    id: str
    description: str
    changes: List[FileChange]
    impact_summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_files_changed(self) -> int:
        """Number of files affected."""
        return len(self.changes)
    
    @property
    def num_files_created(self) -> int:
        """Number of new files."""
        return sum(1 for c in self.changes if c.change_type == "create")
    
    @property
    def num_files_modified(self) -> int:
        """Number of modified files."""
        return sum(1 for c in self.changes if c.change_type == "modify")
    
    @property
    def num_files_deleted(self) -> int:
        """Number of deleted files."""
        return sum(1 for c in self.changes if c.change_type == "delete")


@dataclass
class RefactorPlan:
    """Represents a refactoring operation."""
    refactor_type: str  # "extract-function", "rename", "extract-service"
    description: str
    source_locations: List[Location]
    target_location: Location
    call_sites: List[Location]
    changes: List[FileChange]
    
    @property
    def num_call_sites(self) -> int:
        """Number of call sites that will be updated."""
        return len(self.call_sites)


@dataclass
class TestCase:
    """Represents a generated test."""
    name: str
    target_function: str
    test_code: str
    description: str
    test_type: Literal["unit", "integration"] = "unit"
    coverage_impact: float = 0.0
    
    def __str__(self) -> str:
        return f"Test: {self.name} for {self.target_function}"


@dataclass
class ApplyResult:
    """Result of applying changes."""
    success: bool
    files_changed: List[str]
    backup_id: Optional[str] = None
    error: Optional[str] = None
    
    def __str__(self) -> str:
        if self.success:
            return f"✅ Applied changes to {len(self.files_changed)} files"
        return f"❌ Failed: {self.error}"


@dataclass
class ValidationResult:
    """Result of validating code changes."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        if self.valid:
            return "✅ Validation passed"
        return f"❌ Validation failed: {', '.join(self.errors)}"


@dataclass
class ChatMessage:
    """Represents a single message in a chat conversation."""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"[{self.role}] {self.content[:100]}"


@dataclass
class ChatSession:
    """Represents a chat session with conversation history."""
    id: str
    project_name: str
    messages: List[ChatMessage] = field(default_factory=list)
    pending_proposals: List[CodeProposal] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    
    def add_message(self, role: str, content: str, timestamp: str, metadata: Optional[Dict] = None):
        """Add a message to the conversation."""
        self.messages.append(ChatMessage(
            role=role,
            content=content,
            timestamp=timestamp,
            metadata=metadata or {}
        ))
        self.updated_at = timestamp
    
    def clear_history(self):
        """Clear all messages from this session."""
        self.messages.clear()
    
    def clear_proposals(self):
        """Clear all pending proposals."""
        self.pending_proposals.clear()
    
    @property
    def message_count(self) -> int:
        """Number of messages in conversation."""
        return len(self.messages)
    
    def get_recent_messages(self, n: int = 3) -> List[ChatMessage]:
        """Get the N most recent messages."""
        return self.messages[-n:] if len(self.messages) >= n else self.messages
