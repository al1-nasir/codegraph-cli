"""Command hierarchy groups for organized CLI experience.

Provides logical grouping of commands under:
  cg config   — Configuration management
  cg project  — Project management
  cg analyze  — Code analysis
  cg chat     — Interactive AI chat
"""

from __future__ import annotations

import typer

# ── Configuration group ──────────────────────────────────────
config_grp = typer.Typer(
    help="⚙️  Configuration — LLM, embedding, and setup.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# ── Project management group ─────────────────────────────────
project_grp = typer.Typer(
    help="📂 Projects — index, load, and manage project memories.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# ── Analysis group ───────────────────────────────────────────
analyze_grp = typer.Typer(
    help="🔍 Analysis — search, impact, graph, and RAG context.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
