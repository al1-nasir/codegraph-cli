"""DiffEngine for previewing and applying code changes."""

from __future__ import annotations

import difflib
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models_v2 import ApplyResult, CodeProposal, FileChange


class DiffEngine:
    """Handles previewing and applying code changes safely."""
    
    def __init__(self, backup_dir: Optional[Path] = None):
        """Initialize DiffEngine.
        
        Args:
            backup_dir: Directory to store backups. Defaults to .codegraph/backups/
        """
        self.backup_dir = backup_dir or Path.home() / ".codegraph" / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_diff(self, original: str, modified: str, filename: str = "file") -> str:
        """Create unified diff between two versions.
        
        Args:
            original: Original content
            modified: Modified content
            filename: Name of file for diff header
            
        Returns:
            Unified diff string
        """
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            lineterm=""
        )
        
        return "".join(diff)
    
    def preview_changes(self, proposal: CodeProposal) -> str:
        """Generate preview of all changes in a proposal.
        
        Args:
            proposal: Code proposal to preview
            
        Returns:
            Formatted preview string
        """
        lines = []
        lines.append(f"📝 Proposed Changes: {proposal.description}")
        lines.append(f"   ID: {proposal.id}")
        lines.append("")
        
        # Summary
        if proposal.num_files_created > 0:
            lines.append(f"   [NEW] {proposal.num_files_created} file(s)")
        if proposal.num_files_modified > 0:
            lines.append(f"   [MODIFY] {proposal.num_files_modified} file(s)")
        if proposal.num_files_deleted > 0:
            lines.append(f"   [DELETE] {proposal.num_files_deleted} file(s)")
        lines.append("")
        
        # Detailed changes
        for change in proposal.changes:
            lines.append(f"{'='*60}")
            lines.append(f"[{change.change_type.upper()}] {change.file_path}")
            lines.append(f"{'='*60}")
            
            if change.change_type == "create":
                lines.append(change.new_content or "")
            elif change.change_type == "delete":
                lines.append(f"File will be deleted")
            elif change.change_type == "modify":
                if change.diff:
                    lines.append(change.diff)
                else:
                    # Generate diff if not provided
                    diff = self.create_diff(
                        change.original_content or "",
                        change.new_content or "",
                        change.file_path
                    )
                    lines.append(diff)
            lines.append("")
        
        # Impact summary
        if proposal.impact_summary:
            lines.append(f"{'='*60}")
            lines.append("📊 Impact Analysis")
            lines.append(f"{'='*60}")
            lines.append(proposal.impact_summary)
        
        return "\n".join(lines)
    
    def apply_changes(
        self,
        proposal: CodeProposal,
        backup: bool = True,
        dry_run: bool = False
    ) -> ApplyResult:
        """Apply changes from a proposal to the filesystem.
        
        Args:
            proposal: Code proposal to apply
            backup: Whether to create backups before applying
            dry_run: If True, don't actually apply changes
            
        Returns:
            ApplyResult with success status and details
        """
        if dry_run:
            return ApplyResult(
                success=True,
                files_changed=[c.file_path for c in proposal.changes],
                backup_id=None
            )
        
        backup_id = None
        if backup:
            backup_id = self._create_backup(proposal)
        
        files_changed = []
        
        try:
            for change in proposal.changes:
                file_path = Path(change.file_path)
                
                if change.change_type == "create":
                    # Create new file
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(change.new_content or "")
                    files_changed.append(str(file_path))
                
                elif change.change_type == "modify":
                    # Modify existing file
                    if not file_path.exists():
                        raise FileNotFoundError(f"File not found: {file_path}")
                    file_path.write_text(change.new_content or "")
                    files_changed.append(str(file_path))
                
                elif change.change_type == "delete":
                    # Delete file
                    if file_path.exists():
                        file_path.unlink()
                        files_changed.append(str(file_path))
            
            return ApplyResult(
                success=True,
                files_changed=files_changed,
                backup_id=backup_id
            )
        
        except Exception as e:
            # Rollback if backup exists
            if backup_id:
                self.rollback(backup_id)
            
            return ApplyResult(
                success=False,
                files_changed=[],
                error=str(e)
            )
    
    def _create_backup(self, proposal: CodeProposal) -> str:
        """Create backup of files before applying changes.
        
        Args:
            proposal: Proposal containing files to backup
            
        Returns:
            Backup ID for rollback
        """
        backup_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        backup_path = self.backup_dir / backup_id
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = {
            "proposal_id": proposal.id,
            "description": proposal.description,
            "timestamp": datetime.now().isoformat(),
            "files": []
        }
        
        # Backup each file
        for change in proposal.changes:
            file_path = Path(change.file_path)
            
            if change.change_type in ["modify", "delete"] and file_path.exists():
                # Copy original file to backup
                backup_file = backup_path / file_path.name
                shutil.copy2(file_path, backup_file)
                metadata["files"].append({
                    "original": str(file_path),
                    "backup": str(backup_file),
                    "change_type": change.change_type
                })
        
        # Save metadata
        import json
        (backup_path / "metadata.json").write_text(json.dumps(metadata, indent=2))
        
        return backup_id
    
    def rollback(self, backup_id: str) -> bool:
        """Rollback changes using a backup.
        
        Args:
            backup_id: ID of backup to restore
            
        Returns:
            True if successful, False otherwise
        """
        backup_path = self.backup_dir / backup_id
        
        if not backup_path.exists():
            return False
        
        try:
            # Load metadata
            import json
            metadata = json.loads((backup_path / "metadata.json").read_text())
            
            # Restore each file
            for file_info in metadata["files"]:
                original_path = Path(file_info["original"])
                backup_file = Path(file_info["backup"])
                
                if backup_file.exists():
                    shutil.copy2(backup_file, original_path)
            
            return True
        
        except Exception:
            return False
    
    def list_backups(self) -> list[dict]:
        """List all available backups.
        
        Returns:
            List of backup metadata
        """
        backups = []
        
        for backup_dir in self.backup_dir.iterdir():
            if backup_dir.is_dir():
                metadata_file = backup_dir / "metadata.json"
                if metadata_file.exists():
                    import json
                    metadata = json.loads(metadata_file.read_text())
                    metadata["backup_id"] = backup_dir.name
                    backups.append(metadata)
        
        return sorted(backups, key=lambda x: x["timestamp"], reverse=True)
