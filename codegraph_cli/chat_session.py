"""Chat session management with persistence."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .models_v2 import ChatMessage, ChatSession, CodeProposal


class SessionManager:
    """Manages chat session persistence and loading."""
    
    def __init__(self, sessions_dir: Optional[Path] = None):
        """Initialize session manager.
        
        Args:
            sessions_dir: Directory to store sessions (default: ~/.codegraph/chat_sessions/)
        """
        if sessions_dir is None:
            home = Path.home()
            sessions_dir = home / ".codegraph" / "chat_sessions"
        
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
    
    def create_session(self, project_name: str) -> ChatSession:
        """Create a new chat session.
        
        Args:
            project_name: Name of the project this session is for
            
        Returns:
            New ChatSession instance
        """
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        session = ChatSession(
            id=session_id,
            project_name=project_name,
            created_at=timestamp,
            updated_at=timestamp
        )
        
        return session
    
    def save_session(self, session: ChatSession) -> None:
        """Save session to disk.
        
        Args:
            session: Session to save
        """
        session_file = self.sessions_dir / f"{session.id}.json"
        
        # Convert to dict
        data = {
            "id": session.id,
            "project_name": session.project_name,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "metadata": msg.metadata
                }
                for msg in session.messages
            ],
            "pending_proposals": [
                {
                    "id": p.id,
                    "description": p.description,
                    "impact_summary": p.impact_summary,
                    "metadata": p.metadata,
                    "changes": [
                        {
                            "file_path": c.file_path,
                            "change_type": c.change_type,
                            "original_content": c.original_content,
                            "new_content": c.new_content,
                            "diff": c.diff
                        }
                        for c in p.changes
                    ]
                }
                for p in session.pending_proposals
            ]
        }
        
        session_file.write_text(json.dumps(data, indent=2))
    
    def load_session(self, session_id: str) -> Optional[ChatSession]:
        """Load session from disk.
        
        Args:
            session_id: ID of session to load
            
        Returns:
            ChatSession if found, None otherwise
        """
        session_file = self.sessions_dir / f"{session_id}.json"
        
        if not session_file.exists():
            return None
        
        data = json.loads(session_file.read_text())
        
        # Reconstruct session
        session = ChatSession(
            id=data["id"],
            project_name=data["project_name"],
            created_at=data["created_at"],
            updated_at=data["updated_at"]
        )
        
        # Reconstruct messages
        for msg_data in data["messages"]:
            session.messages.append(ChatMessage(
                role=msg_data["role"],
                content=msg_data["content"],
                timestamp=msg_data["timestamp"],
                metadata=msg_data.get("metadata", {})
            ))
        
        # Reconstruct pending proposals
        from .models_v2 import FileChange
        for prop_data in data.get("pending_proposals", []):
            changes = [
                FileChange(
                    file_path=c["file_path"],
                    change_type=c["change_type"],
                    original_content=c.get("original_content"),
                    new_content=c.get("new_content"),
                    diff=c.get("diff", "")
                )
                for c in prop_data["changes"]
            ]
            
            session.pending_proposals.append(CodeProposal(
                id=prop_data["id"],
                description=prop_data["description"],
                changes=changes,
                impact_summary=prop_data.get("impact_summary", ""),
                metadata=prop_data.get("metadata", {})
            ))
        
        return session
    
    def list_sessions(self, project_name: Optional[str] = None) -> List[dict]:
        """List all saved sessions.
        
        Args:
            project_name: Optional filter by project name
            
        Returns:
            List of session metadata dicts
        """
        sessions = []
        
        for session_file in self.sessions_dir.glob("*.json"):
            try:
                data = json.loads(session_file.read_text())
                
                # Filter by project if specified
                if project_name and data.get("project_name") != project_name:
                    continue
                
                sessions.append({
                    "id": data["id"],
                    "project_name": data["project_name"],
                    "created_at": data["created_at"],
                    "updated_at": data["updated_at"],
                    "message_count": len(data.get("messages", []))
                })
            except Exception:
                # Skip corrupted files
                continue
        
        # Sort by updated_at (most recent first)
        sessions.sort(key=lambda s: s["updated_at"], reverse=True)
        
        return sessions
    
    def get_latest_session(self, project_name: str) -> Optional[str]:
        """Get the most recent session ID for a project.
        
        Args:
            project_name: Project to find session for
            
        Returns:
            Session ID if found, None otherwise
        """
        sessions = self.list_sessions(project_name=project_name)
        
        if sessions:
            return sessions[0]["id"]
        
        return None
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session.
        
        Args:
            session_id: ID of session to delete
            
        Returns:
            True if deleted, False if not found
        """
        session_file = self.sessions_dir / f"{session_id}.json"
        
        if session_file.exists():
            session_file.unlink()
            return True
        
        return False
