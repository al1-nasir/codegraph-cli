"""CLI commands for test generation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from . import config
from .llm import LocalLLM
from .storage import GraphStore, ProjectManager
from .testgen_agent import TestGenAgent

# Create sub-app for test generation
test_app = typer.Typer(help="Generate tests from code graph")


def _get_testgen_agent(pm: ProjectManager) -> TestGenAgent:
    """Get TestGenAgent with current project context."""
    project = pm.get_current_project()
    if not project:
        raise typer.BadParameter("No project loaded. Use 'cg load-project <name>' or run 'cg index <path>'.")
    
    project_dir = pm.project_dir(project)
    if not project_dir.exists():
        raise typer.BadParameter(f"Loaded project '{project}' does not exist in memory.")
    
    store = GraphStore(project_dir)
    
    # Create LLM if available
    llm = None
    if config.LLM_PROVIDER and config.LLM_MODEL:
        llm = LocalLLM(
            model=config.LLM_MODEL,
            provider=config.LLM_PROVIDER,
            api_key=config.LLM_API_KEY
        )
    
    return TestGenAgent(store, llm)


@test_app.command("unit")
def generate_unit_tests(
    symbol: str = typer.Argument(..., help="Function name to generate tests for"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output test file path"),
):
    """🧪 Generate unit tests for a function.

    Example:
      cg v2 test unit "calculate_total"
      cg v2 test unit "UserService.authenticate" --output tests/test_auth.py
    """
    pm = ProjectManager()
    agent = _get_testgen_agent(pm)
    
    typer.echo(f"🧪 Generating unit tests for '{symbol}'...")
    
    try:
        tests = agent.generate_unit_tests(symbol)
    except ValueError as e:
        typer.echo(f"❌ Error: {e}")
        raise typer.Exit(1)
    
    if not tests:
        typer.echo("❌ No tests generated")
        return
    
    typer.echo(f"\n✅ Generated {len(tests)} test(s):\n")
    
    # Show tests
    all_test_code = []
    for test in tests:
        typer.echo(f"📝 {test.name}")
        typer.echo(f"   {test.description}")
        typer.echo("")
        all_test_code.append(test.test_code)
    
    # Show coverage impact
    coverage = agent.analyze_coverage_impact(tests)
    typer.echo(f"📊 Coverage Impact:")
    typer.echo(f"   Estimated increase: +{coverage['coverage_increase']:.1f}%")
    typer.echo(f"   Functions covered: {coverage['functions_covered']}")
    typer.echo("")
    
    # Show full test code
    typer.echo("="*60)
    typer.echo("GENERATED TEST CODE")
    typer.echo("="*60)
    full_code = "\n\n".join(all_test_code)
    typer.echo(full_code)
    typer.echo("")
    
    # Write to file if requested
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add imports
        test_file_content = "import pytest\n\n"
        test_file_content += full_code
        
        output_path.write_text(test_file_content)
        typer.echo(f"✅ Tests written to {output}")
    else:
        typer.echo("💡 Use --output to save tests to a file")


@test_app.command("integration")
def generate_integration_tests(
    flow: str = typer.Argument(..., help="User flow description"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output test file path"),
):
    """🧪 Generate integration tests for a user flow.

    Example:
      cg v2 test integration "user login and token refresh"
      cg v2 test integration "payment processing" --output tests/test_payments.py
    """
    pm = ProjectManager()
    agent = _get_testgen_agent(pm)
    
    typer.echo(f"🧪 Generating integration test for '{flow}'...")
    
    tests = agent.generate_integration_tests(flow)
    
    if not tests:
        typer.echo("❌ No tests generated")
        return
    
    typer.echo(f"\n✅ Generated {len(tests)} test(s):\n")
    
    # Show tests
    all_test_code = []
    for test in tests:
        typer.echo(f"📝 {test.name}")
        typer.echo(f"   {test.description}")
        typer.echo("")
        all_test_code.append(test.test_code)
    
    # Show full test code
    typer.echo("="*60)
    typer.echo("GENERATED TEST CODE")
    typer.echo("="*60)
    full_code = "\n\n".join(all_test_code)
    typer.echo(full_code)
    typer.echo("")
    
    # Write to file if requested
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add imports
        test_file_content = "import pytest\n\n"
        test_file_content += full_code
        
        output_path.write_text(test_file_content)
        typer.echo(f"✅ Tests written to {output}")
    else:
        typer.echo("💡 Use --output to save tests to a file")


@test_app.command("coverage")
def show_coverage_prediction(
    symbol: str = typer.Argument(..., help="Function to analyze"),
):
    """📊 Show predicted coverage impact of generating tests.

    Example:
      cg v2 test coverage "process_payment"
    """
    pm = ProjectManager()
    agent = _get_testgen_agent(pm)
    
    typer.echo(f"📊 Analyzing coverage impact for '{symbol}'...")
    
    try:
        tests = agent.generate_unit_tests(symbol)
        coverage = agent.analyze_coverage_impact(tests)
        
        typer.echo(f"\n📈 Coverage Analysis:")
        typer.echo(f"   Current coverage: {coverage['current_coverage']:.1f}%")
        typer.echo(f"   Estimated after tests: {coverage['estimated_coverage']:.1f}%")
        typer.echo(f"   Increase: +{coverage['coverage_increase']:.1f}%")
        typer.echo(f"   Tests to generate: {coverage['tests_generated']}")
        typer.echo(f"   Functions covered: {coverage['functions_covered']}")
    
    except ValueError as e:
        typer.echo(f"❌ Error: {e}")
        raise typer.Exit(1)
