"""CLI command for AI-generated project onboarding README."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from . import config
from .llm import LocalLLM
from .storage import GraphStore, ProjectManager

console = Console()


def _collect_project_intel(store: GraphStore, source_path: Optional[Path]) -> dict:
    """Gather all graph data needed to generate the README.

    Returns a dict with keys:
        files, functions, classes, modules, edges, entry_points,
        top_callers, dependency_tree, docstrings, file_tree
    """
    nodes = store.get_nodes()
    edges = store.get_edges()

    files: set[str] = set()
    functions: list[dict] = []
    classes: list[dict] = []
    modules: list[dict] = []
    docstrings: dict[str, str] = {}  # qualname → docstring

    for n in nodes:
        files.add(n["file_path"])
        entry = {
            "name": n["name"],
            "qualname": n["qualname"],
            "file": n["file_path"],
            "lines": f"L{n['start_line']}-{n['end_line']}",
        }
        if n["node_type"] == "function":
            functions.append(entry)
        elif n["node_type"] == "class":
            classes.append(entry)
        elif n["node_type"] == "module":
            modules.append(entry)
        if n["docstring"] and n["docstring"].strip():
            docstrings[n["qualname"]] = n["docstring"].strip()

    # Build call count per function (who gets called the most?)
    call_count: dict[str, int] = {}
    callers_of: dict[str, list[str]] = {}
    imports: list[tuple[str, str]] = []

    for e in edges:
        if e["edge_type"] == "calls":
            call_count[e["dst"]] = call_count.get(e["dst"], 0) + 1
            callers_of.setdefault(e["dst"], []).append(e["src"])
        elif e["edge_type"] == "depends_on":
            imports.append((e["src"], e["dst"]))

    # Top-called functions (likely core logic)
    top_called = sorted(call_count.items(), key=lambda x: x[1], reverse=True)[:15]

    # Entry points: functions that are called by nothing (zero incoming call edges)
    all_dst = {e["dst"] for e in edges if e["edge_type"] == "calls"}
    all_func_ids = {f["qualname"] for f in functions}
    entry_points = sorted(all_func_ids - all_dst)[:20]

    # File tree (compact)
    sorted_files = sorted(files)
    file_tree = "\n".join(f"  {f}" for f in sorted_files[:60])
    if len(sorted_files) > 60:
        file_tree += f"\n  ... and {len(sorted_files) - 60} more files"

    # Detect likely config / entry files
    notable_files = []
    for f in sorted_files:
        name = Path(f).name.lower()
        if name in (
            "main.py", "app.py", "cli.py", "__main__.py", "server.py",
            "wsgi.py", "asgi.py", "manage.py", "setup.py", "conftest.py",
            "settings.py", "config.py", "urls.py", "routes.py",
            "index.js", "index.ts", "app.js", "app.ts", "server.js",
        ):
            notable_files.append(f)

    return {
        "file_count": len(files),
        "function_count": len(functions),
        "class_count": len(classes),
        "module_count": len(modules),
        "edge_count": len(edges),
        "files": sorted_files,
        "notable_files": notable_files,
        "functions": functions,
        "classes": classes,
        "entry_points": entry_points,
        "top_called": top_called,
        "imports": imports[:30],
        "docstrings": docstrings,
        "file_tree": file_tree,
    }


def _build_onboard_prompt(project_name: str, intel: dict, source_path: Optional[str]) -> str:
    """Build the LLM prompt from collected intelligence."""
    # Top-called functions with callers
    top_called_text = ""
    for qualname, count in intel["top_called"]:
        top_called_text += f"  - {qualname} (called {count} times)\n"

    # Entry points
    entry_text = "\n".join(f"  - {ep}" for ep in intel["entry_points"][:15])

    # Classes
    class_text = "\n".join(
        f"  - {c['qualname']} ({c['file']})" for c in intel["classes"][:20]
    )

    # Notable files
    notable_text = "\n".join(f"  - {f}" for f in intel["notable_files"])

    # Key docstrings (first 12)
    doc_text = ""
    for qualname, doc in list(intel["docstrings"].items())[:12]:
        first_line = doc.split("\n")[0][:120]
        doc_text += f"  - {qualname}: {first_line}\n"

    # Import relationships (module dependencies)
    import_text = "\n".join(
        f"  - {src} → {dst}" for src, dst in intel["imports"][:20]
    )

    prompt = f"""You are a senior developer writing a README.md for a project you just analyzed.
Generate a complete, professional README.md based on the code graph analysis below.

PROJECT: {project_name}
SOURCE: {source_path or 'unknown'}

STATISTICS:
  - {intel['file_count']} files
  - {intel['function_count']} functions
  - {intel['class_count']} classes
  - {intel['edge_count']} dependency edges

FILE STRUCTURE:
{intel['file_tree']}

NOTABLE FILES (likely entry points / config):
{notable_text or '  (none detected)'}

TOP ENTRY POINTS (functions not called by anything else — likely CLI/API handlers):
{entry_text or '  (none detected)'}

MOST-CALLED FUNCTIONS (core logic):
{top_called_text or '  (none detected)'}

CLASSES:
{class_text or '  (none)'}

KEY DOCSTRINGS:
{doc_text or '  (none found)'}

MODULE DEPENDENCIES:
{import_text or '  (none)'}

INSTRUCTIONS:
Write a complete README.md with these sections:
1. **Project title and one-line description** — infer the purpose from the code structure, docstrings, and function names.
2. **Overview** — 2-3 paragraphs explaining what this project does, its architecture, and key design decisions.
3. **Project Structure** — a clean tree view of the major directories/files with brief descriptions.
4. **Key Modules** — describe the 5-8 most important modules and what they do.
5. **Getting Started** — installation steps (infer from file structure: requirements.txt, pyproject.toml, package.json, etc.)
6. **Usage** — example commands or API usage (infer from entry points and CLI handlers).
7. **Architecture** — how the components connect. Use the dependency graph data.
8. **Contributing** — brief contributing guidelines.

RULES:
- Output ONLY the markdown content, no preamble or explanation.
- Be specific — use real function names, file paths, and class names from the analysis.
- Do NOT invent features not evidenced by the code graph.
- Keep it concise but comprehensive. Target ~200-400 lines.
- Use proper markdown formatting with headers, code blocks, and tables where appropriate.
"""
    return prompt


def _generate_fallback_readme(project_name: str, intel: dict, source_path: Optional[str]) -> str:
    """Generate a README without LLM — pure template from graph data."""
    lines = [
        f"# {project_name}",
        "",
    ]

    # One-liner from first docstring or generic
    first_doc = next(iter(intel["docstrings"].values()), None)
    if first_doc:
        lines.append(f"> {first_doc.split(chr(10))[0]}")
    else:
        lines.append(f"> Auto-generated documentation for **{project_name}**.")
    lines.append("")

    # Stats
    lines.extend([
        "## Overview",
        "",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Files | {intel['file_count']} |",
        f"| Functions | {intel['function_count']} |",
        f"| Classes | {intel['class_count']} |",
        f"| Dependencies | {intel['edge_count']} |",
        "",
    ])

    # Project structure
    lines.extend(["## Project Structure", "", "```"])
    for f in intel["files"][:40]:
        lines.append(f)
    if len(intel["files"]) > 40:
        lines.append(f"... and {len(intel['files']) - 40} more files")
    lines.extend(["```", ""])

    # Key classes
    if intel["classes"]:
        lines.extend(["## Key Classes", ""])
        for c in intel["classes"][:15]:
            doc = intel["docstrings"].get(c["qualname"], "")
            desc = f" — {doc.split(chr(10))[0]}" if doc else ""
            lines.append(f"- **`{c['qualname']}`** ({c['file']}){desc}")
        lines.append("")

    # Entry points
    if intel["entry_points"]:
        lines.extend(["## Entry Points", "", "Functions that are not called by any other indexed code:", ""])
        for ep in intel["entry_points"][:15]:
            lines.append(f"- `{ep}`")
        lines.append("")

    # Most-called functions
    if intel["top_called"]:
        lines.extend(["## Core Functions", "", "Most frequently called functions in the codebase:", ""])
        lines.append("| Function | Call Count |")
        lines.append("|----------|-----------|")
        for qualname, count in intel["top_called"]:
            lines.append(f"| `{qualname}` | {count} |")
        lines.append("")

    # Notable files
    if intel["notable_files"]:
        lines.extend(["## Notable Files", ""])
        for f in intel["notable_files"]:
            lines.append(f"- `{f}`")
        lines.append("")

    lines.extend([
        "---",
        "",
        f"*Generated by [CodeGraph CLI](https://github.com/al1-nasir/codegraph-cli) from code graph analysis.*",
    ])

    return "\n".join(lines)


def onboard(
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output file path (default: prints to stdout).",
    ),
    save: bool = typer.Option(
        False, "--save", "-s",
        help="Save as ONBOARD.md in the project source directory.",
    ),
    no_llm: bool = typer.Option(
        False, "--no-llm",
        help="Skip LLM and generate from template only.",
    ),
    llm_provider: str = typer.Option(config.LLM_PROVIDER, help="LLM provider."),
    llm_model: str = typer.Option(config.LLM_MODEL, help="LLM model."),
    llm_api_key: Optional[str] = typer.Option(config.LLM_API_KEY, help="API key."),
):
    """🚀 Auto-generate a project README from the code graph.

    Analyzes the indexed project's structure, dependencies, entry points,
    and docstrings to produce a comprehensive README.md — either via LLM
    or from a pure template.

    Examples:
      cg onboard                        # print to stdout
      cg onboard --save                 # save as ONBOARD.md in project dir
      cg onboard -o README.md           # save to specific file
      cg onboard --no-llm               # template only, no LLM call
    """
    pm = ProjectManager()
    project = pm.get_current_project()
    if not project:
        console.print("[red]❌ No project loaded.[/red]")
        console.print("[dim]Use: cg project index <path>  or  cg project load <name>[/dim]")
        raise typer.Exit(1)

    project_dir = pm.project_dir(project)
    if not project_dir.exists():
        console.print(f"[red]❌ Project '{project}' not found in memory.[/red]")
        raise typer.Exit(1)

    store = GraphStore(project_dir)
    metadata = store.get_metadata()
    source_path = metadata.get("source_path") or metadata.get("project_root")

    # ── Step 1: Collect intelligence from the graph ──────────────
    with Progress(
        SpinnerColumn(), TextColumn("[cyan]{task.description}[/cyan]"), console=console,
    ) as progress:
        task = progress.add_task("Analyzing project graph...", total=None)
        intel = _collect_project_intel(store, Path(source_path) if source_path else None)
        progress.update(task, description="[green]Graph analysis complete")

    console.print(
        f"  [dim]Analyzed[/dim] [white]{intel['file_count']} files, "
        f"{intel['function_count']} functions, "
        f"{intel['class_count']} classes[/white]"
    )

    # ── Step 2: Generate README ──────────────────────────────────
    if no_llm:
        console.print("  [dim]Generating template-based README (--no-llm)...[/dim]")
        readme_content = _generate_fallback_readme(project, intel, source_path)
    else:
        llm = LocalLLM(
            model=llm_model,
            provider=llm_provider,
            api_key=llm_api_key,
        )
        prompt = _build_onboard_prompt(project, intel, source_path)

        with Progress(
            SpinnerColumn(), TextColumn("[cyan]{task.description}[/cyan]"), console=console,
        ) as progress:
            task = progress.add_task("Generating README with LLM...", total=None)
            result = llm.explain(prompt)
            progress.update(task, description="[green]README generated")

        # If LLM returned something useful, use it; otherwise fall back
        if result and len(result) > 200 and not result.startswith("LLM provider"):
            readme_content = result
        else:
            console.print("  [yellow]⚠ LLM unavailable, using template fallback.[/yellow]")
            readme_content = _generate_fallback_readme(project, intel, source_path)

    store.close()

    # ── Step 3: Output ───────────────────────────────────────────
    if output:
        output.write_text(readme_content, encoding="utf-8")
        console.print(f"\n[green]✅ Saved to {output}[/green]")
    elif save:
        if source_path:
            dest = Path(source_path) / "ONBOARD.md"
        else:
            dest = Path.cwd() / "ONBOARD.md"
        dest.write_text(readme_content, encoding="utf-8")
        console.print(f"\n[green]✅ Saved to {dest}[/green]")
    else:
        # Print to stdout with a panel
        console.print()
        console.print(Panel(
            readme_content,
            title=f"📄 Generated README for {project}",
            title_align="left",
            border_style="cyan",
            expand=False,
        ))

    console.print(
        f"\n[dim]Tip: Review and edit the output — AI-generated docs are a starting point, not final.[/dim]"
    )
