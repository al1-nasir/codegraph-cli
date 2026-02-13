"""ValidationEngine for detecting and fixing code errors."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List, Optional, Tuple

from .models_v2 import FileChange, ValidationResult


class ValidationEngine:
    """Validates code and suggests/applies fixes for common errors."""
    
    def __init__(self):
        """Initialize ValidationEngine."""
        pass
    
    def diagnose_project(self, project_path: Path) -> List[dict]:
        """Find all syntax errors in a project.
        
        Args:
            project_path: Path to project root
            
        Returns:
            List of error dictionaries with file, line, error info
        """
        errors = []
        
        for py_file in project_path.rglob("*.py"):
            file_errors = self.check_file(py_file)
            errors.extend(file_errors)
        
        return errors
    
    def check_file(self, file_path: Path) -> List[dict]:
        """Check a single file for syntax errors.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            List of errors found
        """
        errors = []
        
        try:
            content = file_path.read_text()
            ast.parse(content)
        except SyntaxError as e:
            errors.append({
                "file": str(file_path),
                "line": e.lineno,
                "column": e.offset,
                "error": str(e.msg),
                "type": "SyntaxError"
            })
        except IndentationError as e:
            errors.append({
                "file": str(file_path),
                "line": e.lineno,
                "column": e.offset,
                "error": str(e.msg),
                "type": "IndentationError"
            })
        except Exception as e:
            errors.append({
                "file": str(file_path),
                "line": 0,
                "column": 0,
                "error": str(e),
                "type": type(e).__name__
            })
        
        return errors
    
    def fix_common_errors(self, file_path: Path) -> Optional[FileChange]:
        """Attempt to fix common syntax errors automatically.
        
        Args:
            file_path: Path to file with errors
            
        Returns:
            FileChange if fixes were applied, None otherwise
        """
        original_content = file_path.read_text()
        fixed_content = original_content
        fixes_applied = []
        
        # Check for errors
        errors = self.check_file(file_path)
        if not errors:
            return None
        
        # Try common fixes
        for error in errors:
            if "invalid syntax" in error["error"].lower():
                # Try fixing missing parentheses, brackets, quotes
                fixed_content, fix_msg = self._fix_missing_delimiters(fixed_content, error)
                if fix_msg:
                    fixes_applied.append(fix_msg)
            
            elif "indentation" in error["error"].lower():
                # Fix indentation issues
                fixed_content, fix_msg = self._fix_indentation(fixed_content)
                if fix_msg:
                    fixes_applied.append(fix_msg)
            
            elif "eol while scanning" in error["error"].lower():
                # Fix unclosed strings
                fixed_content, fix_msg = self._fix_unclosed_strings(fixed_content, error)
                if fix_msg:
                    fixes_applied.append(fix_msg)
        
        # Verify fix worked
        try:
            ast.parse(fixed_content)
            # Success!
            return FileChange(
                file_path=str(file_path),
                change_type="modify",
                original_content=original_content,
                new_content=fixed_content
            )
        except:
            # Fix didn't work, return None
            return None
    
    def _fix_missing_delimiters(self, content: str, error: dict) -> Tuple[str, Optional[str]]:
        """Try to fix missing parentheses, brackets, or braces.
        
        Args:
            content: File content
            error: Error information
            
        Returns:
            (fixed_content, fix_message)
        """
        lines = content.splitlines(keepends=True)
        line_num = error["line"] - 1
        
        if line_num >= len(lines):
            return content, None
        
        line = lines[line_num]
        
        # Count delimiters
        open_parens = line.count('(') - line.count(')')
        open_brackets = line.count('[') - line.count(']')
        open_braces = line.count('{') - line.count('}')
        
        fixed_line = line
        fix_msg = None
        
        if open_parens > 0:
            fixed_line = fixed_line.rstrip() + ')' * open_parens + '\n'
            fix_msg = f"Added {open_parens} closing parenthesis"
        elif open_brackets > 0:
            fixed_line = fixed_line.rstrip() + ']' * open_brackets + '\n'
            fix_msg = f"Added {open_brackets} closing bracket"
        elif open_braces > 0:
            fixed_line = fixed_line.rstrip() + '}' * open_braces + '\n'
            fix_msg = f"Added {open_braces} closing brace"
        
        if fix_msg:
            lines[line_num] = fixed_line
            return "".join(lines), fix_msg
        
        return content, None
    
    def _fix_indentation(self, content: str) -> Tuple[str, Optional[str]]:
        """Fix indentation issues (tabs vs spaces).
        
        Args:
            content: File content
            
        Returns:
            (fixed_content, fix_message)
        """
        # Convert all tabs to 4 spaces
        if '\t' in content:
            fixed = content.replace('\t', '    ')
            return fixed, "Converted tabs to 4 spaces"
        
        return content, None
    
    def _fix_unclosed_strings(self, content: str, error: dict) -> Tuple[str, Optional[str]]:
        """Fix unclosed string literals.
        
        Args:
            content: File content
            error: Error information
            
        Returns:
            (fixed_content, fix_message)
        """
        lines = content.splitlines(keepends=True)
        line_num = error["line"] - 1
        
        if line_num >= len(lines):
            return content, None
        
        line = lines[line_num]
        
        # Count quotes
        single_quotes = line.count("'") - line.count("\\'")
        double_quotes = line.count('"') - line.count('\\"')
        
        fixed_line = line
        fix_msg = None
        
        if single_quotes % 2 == 1:
            # Odd number of single quotes
            fixed_line = fixed_line.rstrip() + "'\n"
            fix_msg = "Added closing single quote"
        elif double_quotes % 2 == 1:
            # Odd number of double quotes
            fixed_line = fixed_line.rstrip() + '"\n'
            fix_msg = "Added closing double quote"
        
        if fix_msg:
            lines[line_num] = fixed_line
            return "".join(lines), fix_msg
        
        return content, None
    
    def validate_syntax(self, code: str) -> ValidationResult:
        """Check if code has valid Python syntax.
        
        Args:
            code: Python code to validate
            
        Returns:
            ValidationResult
        """
        try:
            ast.parse(code)
            return ValidationResult(valid=True)
        except SyntaxError as e:
            return ValidationResult(
                valid=False,
                errors=[f"SyntaxError at line {e.lineno}: {e.msg}"]
            )
        except Exception as e:
            return ValidationResult(
                valid=False,
                errors=[f"{type(e).__name__}: {str(e)}"]
            )
    
    def validate_imports(self, code: str) -> ValidationResult:
        """Check if all imports are available.
        
        Args:
            code: Python code to validate
            
        Returns:
            ValidationResult
        """
        warnings = []
        
        try:
            tree = ast.parse(code)
        except:
            return ValidationResult(valid=False, errors=["Cannot parse code"])
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    try:
                        __import__(alias.name)
                    except ImportError:
                        warnings.append(f"Import '{alias.name}' may not be available")
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    try:
                        __import__(node.module)
                    except ImportError:
                        warnings.append(f"Module '{node.module}' may not be available")
        
        return ValidationResult(
            valid=True,
            warnings=warnings
        )
