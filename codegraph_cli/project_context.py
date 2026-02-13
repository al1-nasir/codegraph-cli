"""Project context manager with real file system access."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .storage import GraphStore, ProjectManager


class ProjectContext:
    """Unified context for project with real file access and code graph."""
    
    def __init__(self, project_name: str, project_manager: ProjectManager):
        """Initialize project context.
        
        Args:
            project_name: Name of the project
            project_manager: ProjectManager instance
        """
        self.project_name = project_name
        self.project_manager = project_manager
        self.project_dir = project_manager.project_dir(project_name)
        self.store = GraphStore(self.project_dir)
        self.metadata = self.store.get_metadata()
        
        # Get source path from metadata
        source_path_str = self.metadata.get("source_path")
        if source_path_str:
            self.source_path = Path(source_path_str)
            if not self.source_path.exists():
                # Source path moved or deleted
                self.source_path = None
        else:
            # Old project without source_path
            self.source_path = None
    
    @property
    def has_source_access(self) -> bool:
        """Check if we have access to the original source directory."""
        return self.source_path is not None and self.source_path.exists()
    
    # File System Operations
    
    def list_directory(self, rel_path: str = ".") -> List[Dict]:
        """List files and directories in project.
        
        Args:
            rel_path: Relative path from project root
            
        Returns:
            List of file/directory info dicts
            
        Raises:
            RuntimeError: If source path not available
        """
        if not self.has_source_access:
            raise RuntimeError(
                f"Source path not available for project '{self.project_name}'. "
                "Re-index the project to enable file access."
            )
        
        full_path = self.source_path / rel_path
        if not full_path.exists():
            return []
        
        if not full_path.is_dir():
            raise ValueError(f"Path is not a directory: {rel_path}")
        
        items = []
        for item in sorted(full_path.iterdir()):
            try:
                stat = item.stat()
                items.append({
                    "name": item.name,
                    "type": "dir" if item.is_dir() else "file",
                    "size": stat.st_size if item.is_file() else None,
                    "path": str(item.relative_to(self.source_path)),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            except (OSError, PermissionError):
                # Skip files we can't access
                continue
        
        return items
    
    def read_file(self, rel_path: str) -> str:
        """Read file contents from project.
        
        Args:
            rel_path: Relative path to file
            
        Returns:
            File contents as string
            
        Raises:
            RuntimeError: If source path not available
            FileNotFoundError: If file doesn't exist
        """
        if not self.has_source_access:
            raise RuntimeError(
                f"Source path not available for project '{self.project_name}'. "
                "Re-index the project to enable file access."
            )
        
        full_path = self.source_path / rel_path
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {rel_path}")
        
        if not full_path.is_file():
            raise ValueError(f"Path is not a file: {rel_path}")
        
        try:
            return full_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Try reading as binary for non-text files
            return f"[Binary file: {full_path.suffix}]"
    
    def write_file(self, rel_path: str, content: str, create_dirs: bool = True) -> bool:
        """Write or create a file in the project.
        
        Args:
            rel_path: Relative path to file
            content: File content
            create_dirs: Whether to create parent directories
            
        Returns:
            True if successful
            
        Raises:
            RuntimeError: If source path not available
        """
        if not self.has_source_access:
            raise RuntimeError(
                f"Source path not available for project '{self.project_name}'. "
                "Re-index the project to enable file access."
            )
        
        full_path = self.source_path / rel_path
        
        if create_dirs:
            full_path.parent.mkdir(parents=True, exist_ok=True)
        
        full_path.write_text(content, encoding="utf-8")
        return True
    
    def file_exists(self, rel_path: str) -> bool:
        """Check if a file exists in the project.
        
        Args:
            rel_path: Relative path to file
            
        Returns:
            True if file exists
        """
        if not self.has_source_access:
            return False
        
        return (self.source_path / rel_path).exists()
    
    def get_file_info(self, rel_path: str) -> Optional[Dict]:
        """Get information about a file.
        
        Args:
            rel_path: Relative path to file
            
        Returns:
            File info dict or None if not found
        """
        if not self.has_source_access:
            return None
        
        full_path = self.source_path / rel_path
        if not full_path.exists():
            return None
        
        try:
            stat = full_path.stat()
            return {
                "name": full_path.name,
                "path": rel_path,
                "type": "dir" if full_path.is_dir() else "file",
                "size": stat.st_size if full_path.is_file() else None,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
            }
        except (OSError, PermissionError):
            return None
    
    # Code Graph Operations
    
    def get_indexed_files(self) -> List[str]:
        """Get list of indexed Python files.
        
        Returns:
            List of file paths that were indexed
        """
        nodes = self.store.get_nodes()
        return sorted(set(node["file_path"] for node in nodes))
    
    def get_project_summary(self) -> Dict:
        """Get summary of project.
        
        Returns:
            Dictionary with project statistics
        """
        nodes = self.store.get_nodes()
        edges = self.store.get_edges()
        
        # Count by type
        node_types = {}
        for node in nodes:
            node_type = node["node_type"]
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        return {
            "project_name": self.project_name,
            "source_path": str(self.source_path) if self.source_path else None,
            "has_source_access": self.has_source_access,
            "indexed_at": self.metadata.get("indexed_at"),
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "node_types": node_types,
            "indexed_files": len(self.get_indexed_files())
        }
    
    def close(self):
        """Close the graph store connection."""
        self.store.close()
