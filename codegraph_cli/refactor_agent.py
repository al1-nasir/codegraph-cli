"""RefactorAgent for safe, dependency-aware code refactoring."""

from __future__ import annotations

import ast
import uuid
from pathlib import Path
from typing import List, Optional, Set

from .diff_engine import DiffEngine
from .models_v2 import FileChange, Location, Range, RefactorPlan
from .storage import GraphStore


class RefactorAgent:
    """Performs safe refactoring with automatic dependency tracking."""
    
    def __init__(self, store: GraphStore, diff_engine: Optional[DiffEngine] = None):
        """Initialize RefactorAgent.
        
        Args:
            store: Graph store for dependency tracking
            diff_engine: Engine for managing diffs (optional)
        """
        self.store = store
        self.diff_engine = diff_engine or DiffEngine()
    
    def rename_symbol(self, old_name: str, new_name: str) -> RefactorPlan:
        """Rename a symbol and update all references.
        
        Args:
            old_name: Current symbol name
            new_name: New symbol name
            
        Returns:
            RefactorPlan with all necessary changes
        """
        # Find the symbol in the graph
        node = self.store.get_node(old_name)
        if not node:
            raise ValueError(f"Symbol '{old_name}' not found in project")
        
        # Find all call sites
        call_sites = self.find_call_sites(old_name)
        
        # Create changes for renaming
        changes = []
        files_to_update = set()
        
        # Add the definition file
        files_to_update.add(node["file_path"])
        
        # Add all call site files
        for location in call_sites:
            files_to_update.add(location.file_path)
        
        # Generate changes for each file
        # Get project root from metadata
        metadata = self.store.get_metadata()
        project_root = Path(metadata.get("project_root", "."))
        
        for file_path in files_to_update:
            # Make path absolute
            abs_path = project_root / file_path if not Path(file_path).is_absolute() else Path(file_path)
            
            if not abs_path.exists():
                continue  # Skip non-existent files
                
            original_content = abs_path.read_text()
            new_content = self._rename_in_file(original_content, old_name, new_name)
            
            if original_content != new_content:
                changes.append(FileChange(
                    file_path=str(abs_path),
                    change_type="modify",
                    original_content=original_content,
                    new_content=new_content,
                    diff=self.diff_engine.create_diff(original_content, new_content, str(abs_path))
                ))
        
        return RefactorPlan(
            refactor_type="rename",
            description=f"Rename '{old_name}' to '{new_name}'",
            source_locations=[Location(node["file_path"], node["start_line"])],
            target_location=Location(node["file_path"], node["start_line"]),
            call_sites=call_sites,
            changes=changes
        )
    
    def extract_function(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
        function_name: str
    ) -> RefactorPlan:
        """Extract code range into a new function.
        
        Args:
            file_path: File containing code to extract
            start_line: Start line of code to extract
            end_line: End line of code to extract
            function_name: Name for the new function
            
        Returns:
            RefactorPlan with extraction changes
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise ValueError(f"File not found: {file_path}")
        
        original_content = file_path_obj.read_text()
        lines = original_content.splitlines(keepends=True)
        
        # Extract the code block
        extracted_lines = lines[start_line - 1:end_line]
        extracted_code = "".join(extracted_lines)
        
        # Analyze variables and detect parameters
        indent = self._get_indent(extracted_lines[0]) if extracted_lines else "    "
        params = self._detect_parameters(extracted_code, original_content, start_line, end_line)
        has_return = self._has_return_statement(extracted_code)
        
        # Create new function with detected parameters
        param_str = ", ".join(params) if params else ""
        new_function = f"def {function_name}({param_str}):\n"
        new_function += f"{indent}\"\"\"Extracted function.\"\"\"\n"
        new_function += extracted_code
        
        # Only add return None if no return statements found
        if not has_return:
            new_function += f"{indent}return None\n"
        new_function += "\n\n"
        
        # Replace extracted code with function call
        call_args = ", ".join(params) if params else ""
        if has_return:
            replacement_line = f"{indent}result = {function_name}({call_args})\n"
            replacement_line += f"{indent}if result:\n"
            replacement_line += f"{indent}    return result\n"
        else:
            replacement_line = f"{indent}{function_name}({call_args})\n"
        
        # Find insertion point (before containing function)
        insertion_line = self._find_function_start(lines, start_line)
        
        # Build new content
        new_lines = lines[:insertion_line]
        new_lines.append(new_function)
        new_lines.extend(lines[insertion_line:start_line - 1])
        new_lines.append(replacement_line)
        new_lines.extend(lines[end_line:])
        
        new_content = "".join(new_lines)
        
        changes = [FileChange(
            file_path=file_path,
            change_type="modify",
            original_content=original_content,
            new_content=new_content,
            diff=self.diff_engine.create_diff(original_content, new_content, file_path)
        )]
        
        return RefactorPlan(
            refactor_type="extract-function",
            description=f"Extract lines {start_line}-{end_line} to function '{function_name}'",
            source_locations=[Location(file_path, start_line)],
            target_location=Location(file_path, start_line - 1),
            call_sites=[],
            changes=changes
        )
    
    def extract_service(
        self,
        symbols: List[str],
        target_file: str
    ) -> RefactorPlan:
        """Extract multiple functions to a new service file.
        
        Args:
            symbols: List of function names to extract
            target_file: Path to new service file
            
        Returns:
            RefactorPlan with extraction changes
        """
        changes = []
        source_locations = []
        all_call_sites = []
        
        # Collect all functions to extract
        functions_code = []
        source_files = set()
        
        for symbol in symbols:
            node = self.store.get_node(symbol)
            if not node:
                raise ValueError(f"Symbol '{symbol}' not found")
            
            source_locations.append(Location(node["file_path"], node["start_line"]))
            source_files.add(node["file_path"])
            
            # Get function code
            functions_code.append(node["code"])
            
            # Find call sites
            call_sites = self.find_call_sites(symbol)
            all_call_sites.extend(call_sites)
        
        # Create new service file
        new_service_content = '"""Extracted service module."""\n\n'
        new_service_content += "\n\n".join(functions_code)
        
        changes.append(FileChange(
            file_path=target_file,
            change_type="create",
            new_content=new_service_content
        ))
        
        # Update source files to remove extracted functions and add imports
        # Get project root from metadata
        metadata = self.store.get_metadata()
        project_root = Path(metadata.get("project_root", "."))
        
        for source_file in source_files:
            # Make path absolute
            abs_path = project_root / source_file if not Path(source_file).is_absolute() else Path(source_file)
            
            if not abs_path.exists():
                continue
                
            original_content = abs_path.read_text()
            new_content = self._remove_functions_and_add_import(
                original_content,
                symbols,
                target_file
            )
            
            if original_content != new_content:
                changes.append(FileChange(
                    file_path=source_file,
                    change_type="modify",
                    original_content=original_content,
                    new_content=new_content,
                    diff=self.diff_engine.create_diff(original_content, new_content, source_file)
                ))
        
        # Update call sites to use new import
        call_site_files = {loc.file_path for loc in all_call_sites}
        for call_site_file in call_site_files:
            if call_site_file not in source_files:
                # Make path absolute
                abs_call_path = project_root / call_site_file if not Path(call_site_file).is_absolute() else Path(call_site_file)
                
                if not abs_call_path.exists():
                    continue
                    
                original_content = abs_call_path.read_text()
                new_content = self._add_import(original_content, symbols, target_file)
                
                if original_content != new_content:
                    changes.append(FileChange(
                        file_path=call_site_file,
                        change_type="modify",
                        original_content=original_content,
                        new_content=new_content,
                        diff=self.diff_engine.create_diff(original_content, new_content, call_site_file)
                    ))
        
        return RefactorPlan(
            refactor_type="extract-service",
            description=f"Extract {len(symbols)} function(s) to {target_file}",
            source_locations=source_locations,
            target_location=Location(target_file, 1),
            call_sites=all_call_sites,
            changes=changes
        )
    
    def find_call_sites(self, symbol: str) -> List[Location]:
        """Find all locations where a symbol is called.
        
        Args:
            symbol: Symbol name to find
            
        Returns:
            List of locations where symbol is used
        """
        call_sites = []
        
        # Use graph to find reverse dependencies
        node = self.store.get_node(symbol)
        if not node:
            return call_sites
        
        node_id = node["node_id"]
        
        # Find all edges pointing to this node
        # (This is a simplified implementation - would need reverse edge lookup)
        all_nodes = self.store.get_nodes()
        
        for other_node in all_nodes:
            # Check if this node has edges to our target
            edges = self.store.neighbors(other_node["node_id"])
            for edge in edges:
                if edge["dst"] == node_id:
                    call_sites.append(Location(
                        other_node["file_path"],
                        other_node["start_line"]
                    ))
        
        return call_sites
    
    def _rename_in_file(self, content: str, old_name: str, new_name: str) -> str:
        """Rename symbol in file content.
        
        Args:
            content: File content
            old_name: Old symbol name
            new_name: New symbol name
            
        Returns:
            Updated content
        """
        # Simple implementation: replace whole words only
        # In production, would use AST-based renaming
        import re
        pattern = r'\b' + re.escape(old_name) + r'\b'
        return re.sub(pattern, new_name, content)
    
    def _get_indent(self, line: str) -> str:
        """Get indentation from a line."""
        return line[:len(line) - len(line.lstrip())]
    
    def _remove_functions_and_add_import(
        self,
        content: str,
        symbols: List[str],
        target_file: str
    ) -> str:
        """Remove functions from content and add import."""
        # Simple implementation
        # In production, would use AST manipulation
        
        # Add import at top
        module_name = Path(target_file).stem
        import_line = f"from .{module_name} import {', '.join(symbols)}\n"
        
        lines = content.splitlines(keepends=True)
        
        # Find first non-import line
        insert_pos = 0
        for i, line in enumerate(lines):
            if not line.strip().startswith(('import ', 'from ', '#', '"""', "'''")):
                insert_pos = i
                break
        
        lines.insert(insert_pos, import_line)
        
        # Remove function definitions (simplified)
        # Would need proper AST-based removal in production
        
        return "".join(lines)
    
    
    def _detect_parameters(self, extracted_code: str, full_content: str, start_line: int, end_line: int) -> List[str]:
        """Detect variables that should be parameters for extracted function.
        
        Args:
            extracted_code: Code being extracted
            full_content: Full file content
            start_line: Start line of extraction
            end_line: End line of extraction
            
        Returns:
            List of parameter names
        """
        try:
            # Parse extracted code to find used variables
            tree = ast.parse(extracted_code)
            used_names = set()
            defined_names = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    used_names.add(node.id)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    defined_names.add(node.id)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    defined_names.add(node.name)
            
            # Parameters are variables used but not defined in extracted code
            # Filter out built-ins and common globals
            builtins = {'True', 'False', 'None', 'print', 'len', 'range', 'str', 'int', 'list', 'dict', 'set'}
            params = used_names - defined_names - builtins
            
            return sorted(list(params))
        except SyntaxError:
            # If parsing fails, return empty list
            return []
    
    def _has_return_statement(self, code: str) -> bool:
        """Check if code contains return statements.
        
        Args:
            code: Code to check
            
        Returns:
            True if code has return statements
        """
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Return):
                    return True
            return False
        except SyntaxError:
            return False
    
    def _find_function_start(self, lines: List[str], current_line: int) -> int:
        """Find the start of the containing function.
        
        Args:
            lines: All lines in file
            current_line: Current line number (1-indexed)
            
        Returns:
            Line number where containing function starts (0-indexed)
        """
        # Search backwards for function definition
        for i in range(current_line - 2, -1, -1):
            line = lines[i].strip()
            if line.startswith('def ') or line.startswith('async def '):
                return i
        
        # If no function found, insert at beginning
        return 0
    
    def _add_import(self, content: str, symbols: List[str], target_file: str) -> str:
        """Add import statement to content."""
        module_name = Path(target_file).stem
        import_line = f"from .{module_name} import {', '.join(symbols)}\n"
        
        lines = content.splitlines(keepends=True)
        
        # Find appropriate position for import
        insert_pos = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('from '):
                insert_pos = i + 1
        
        lines.insert(insert_pos, import_line)
        return "".join(lines)
