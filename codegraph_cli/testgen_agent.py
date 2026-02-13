"""TestGenAgent for graph-powered test generation."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Optional

from .llm import LocalLLM
from .models_v2 import TestCase
from .storage import GraphStore


class TestGenAgent:
    """Generates tests based on call graph analysis and usage patterns."""
    
    def __init__(self, store: GraphStore, llm: Optional[LocalLLM] = None):
        """Initialize TestGenAgent.
        
        Args:
            store: Graph store for analyzing code
            llm: Optional LLM for generating test code
        """
        self.store = store
        self.llm = llm
    
    def generate_unit_tests(self, symbol: str) -> List[TestCase]:
        """Generate unit tests for a function.
        
        Args:
            symbol: Function name to generate tests for
            
        Returns:
            List of generated test cases
        """
        # Get function info from graph
        node = self.store.get_node(symbol)
        if not node:
            raise ValueError(f"Symbol '{symbol}' not found")
        
        # Analyze function signature and dependencies
        test_cases = []
        
        # Parse function to understand parameters
        try:
            tree = ast.parse(node["code"])
            func_def = None
            for node_ast in ast.walk(tree):
                if isinstance(node_ast, ast.FunctionDef):
                    func_def = node_ast
                    break
            
            if func_def:
                # Generate tests based on function analysis
                test_cases.extend(self._generate_basic_tests(symbol, func_def, node))
                test_cases.extend(self._generate_edge_case_tests(symbol, func_def, node))
                test_cases.extend(self._generate_error_tests(symbol, func_def, node))
        
        except Exception:
            # Fallback: generate basic test template
            test_cases.append(self._generate_basic_test_template(symbol, node))
        
        return test_cases
    
    def generate_integration_tests(self, flow_description: str) -> List[TestCase]:
        """Generate integration tests for a user flow.
        
        Args:
            flow_description: Description of the flow to test
            
        Returns:
            List of integration test cases
        """
        # Use LLM to generate integration test if available
        if self.llm:
            prompt = self._build_integration_test_prompt(flow_description)
            test_code = self.llm.explain(prompt)
            
            return [TestCase(
                name=f"test_{flow_description.lower().replace(' ', '_')}",
                target_function=flow_description,
                test_code=test_code,
                description=f"Integration test for {flow_description}",
                test_type="integration"
            )]
        
        # Fallback: basic template
        return [self._generate_integration_template(flow_description)]
    
    def analyze_coverage_impact(self, tests: List[TestCase]) -> dict:
        """Predict coverage improvement from tests.
        
        Args:
            tests: List of test cases
            
        Returns:
            Coverage analysis dictionary
        """
        # Simple heuristic: estimate based on number of tests
        # In production, would use actual coverage tools
        
        total_functions = len(self.store.get_nodes())
        tested_functions = len(set(t.target_function for t in tests))
        
        current_coverage = 0.0  # Would get from coverage tool
        estimated_new_coverage = min(100.0, current_coverage + (tested_functions / max(total_functions, 1)) * 100)
        
        return {
            "current_coverage": current_coverage,
            "estimated_coverage": estimated_new_coverage,
            "coverage_increase": estimated_new_coverage - current_coverage,
            "tests_generated": len(tests),
            "functions_covered": tested_functions
        }
    
    def _generate_basic_tests(
        self,
        symbol: str,
        func_def: ast.FunctionDef,
        node: dict
    ) -> List[TestCase]:
        """Generate basic happy-path tests."""
        tests = []
        
        # Get parameter names
        params = [arg.arg for arg in func_def.args.args if arg.arg != 'self']
        
        # Generate basic test
        test_name = f"test_{symbol}_basic"
        test_code = self._generate_test_code(
            test_name,
            symbol,
            params,
            "Basic functionality test"
        )
        
        tests.append(TestCase(
            name=test_name,
            target_function=symbol,
            test_code=test_code,
            description=f"Test basic functionality of {symbol}",
            test_type="unit"
        ))
        
        return tests
    
    def _generate_edge_case_tests(
        self,
        symbol: str,
        func_def: ast.FunctionDef,
        node: dict
    ) -> List[TestCase]:
        """Generate edge case tests."""
        tests = []
        
        params = [arg.arg for arg in func_def.args.args if arg.arg != 'self']
        
        # Test with empty/None values
        if params:
            test_name = f"test_{symbol}_empty_input"
            test_code = self._generate_test_code(
                test_name,
                symbol,
                params,
                "Empty/None input test",
                use_empty=True
            )
            
            tests.append(TestCase(
                name=test_name,
                target_function=symbol,
                test_code=test_code,
                description=f"Test {symbol} with empty/None inputs",
                test_type="unit"
            ))
        
        return tests
    
    def _generate_error_tests(
        self,
        symbol: str,
        func_def: ast.FunctionDef,
        node: dict
    ) -> List[TestCase]:
        """Generate error handling tests."""
        tests = []
        
        # Check if function has error handling
        has_try_except = False
        for node_ast in ast.walk(func_def):
            if isinstance(node_ast, ast.Try):
                has_try_except = True
                break
        
        if has_try_except:
            test_name = f"test_{symbol}_error_handling"
            params = [arg.arg for arg in func_def.args.args if arg.arg != 'self']
            
            test_code = self._generate_test_code(
                test_name,
                symbol,
                params,
                "Error handling test",
                test_error=True
            )
            
            tests.append(TestCase(
                name=test_name,
                target_function=symbol,
                test_code=test_code,
                description=f"Test error handling in {symbol}",
                test_type="unit"
            ))
        
        return tests
    
    def _generate_test_code(
        self,
        test_name: str,
        function_name: str,
        params: List[str],
        description: str,
        use_empty: bool = False,
        test_error: bool = False
    ) -> str:
        """Generate actual test code."""
        lines = []
        
        lines.append(f"def {test_name}():")
        lines.append(f'    """{description}."""')
        
        # Generate test inputs
        if use_empty:
            args = ", ".join(["None" for _ in params])
        else:
            # Generate reasonable test values
            args = ", ".join([self._generate_test_value(p) for p in params])
        
        if test_error:
            lines.append("    with pytest.raises(Exception):")
            lines.append(f"        result = {function_name}({args})")
        else:
            lines.append(f"    result = {function_name}({args})")
            lines.append("    assert result is not None")
        
        return "\n".join(lines)
    
    def _generate_test_value(self, param_name: str) -> str:
        """Generate a test value based on parameter name."""
        param_lower = param_name.lower()
        
        if 'id' in param_lower:
            return "1"
        elif 'name' in param_lower or 'username' in param_lower:
            return '"test_user"'
        elif 'email' in param_lower:
            return '"test@example.com"'
        elif 'password' in param_lower:
            return '"password123"'
        elif 'count' in param_lower or 'num' in param_lower:
            return "10"
        elif 'data' in param_lower:
            return '{"key": "value"}'
        else:
            return '"test_value"'
    
    def _generate_basic_test_template(self, symbol: str, node: dict) -> TestCase:
        """Generate a basic test template."""
        test_code = f"""def test_{symbol}():
    \"\"\"Test {symbol} function.\"\"\"
    # TODO: Add test implementation
    result = {symbol}()
    assert result is not None
"""
        
        return TestCase(
            name=f"test_{symbol}",
            target_function=symbol,
            test_code=test_code,
            description=f"Basic test for {symbol}",
            test_type="unit"
        )
    
    def _generate_integration_template(self, flow: str) -> TestCase:
        """Generate integration test template."""
        test_name = f"test_{flow.lower().replace(' ', '_')}"
        
        test_code = f"""def {test_name}():
    \"\"\"Integration test for {flow}.\"\"\"
    # TODO: Implement integration test
    # 1. Setup test data
    # 2. Execute flow
    # 3. Verify results
    pass
"""
        
        return TestCase(
            name=test_name,
            target_function=flow,
            test_code=test_code,
            description=f"Integration test for {flow}",
            test_type="integration"
        )
    
    def _build_integration_test_prompt(self, flow: str) -> str:
        """Build prompt for LLM to generate integration test."""
        return f"""Generate a Python integration test for the following user flow: {flow}

Include:
1. Test setup (fixtures, test data)
2. Step-by-step flow execution
3. Assertions to verify each step
4. Cleanup

Use pytest framework. Output only the test code.
"""
