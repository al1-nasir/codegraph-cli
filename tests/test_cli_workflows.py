"""Integration tests for CLI enhancement modules."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from codegraph_cli.cli import app

runner = CliRunner()


# ── Config group tests ───────────────────────────────────────────


class TestConfigGroup:
    """Tests for 'cg config' group."""

    def test_config_help(self):
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0
        assert "setup" in result.stdout.lower()

    def test_config_show_llm(self):
        result = runner.invoke(app, ["config", "show-llm"])
        assert result.exit_code == 0

    def test_config_show_embedding(self):
        result = runner.invoke(app, ["config", "show-embedding"])
        assert result.exit_code == 0


# ── Project group tests ──────────────────────────────────────────


class TestProjectGroup:
    """Tests for 'cg project' group."""

    def test_project_help(self):
        result = runner.invoke(app, ["project", "--help"])
        assert result.exit_code == 0
        assert "index" in result.stdout.lower()

    def test_project_list(self, temp_project_manager):
        result = runner.invoke(app, ["project", "list"])
        assert result.exit_code == 0

    def test_project_current(self, temp_project_manager):
        result = runner.invoke(app, ["project", "current"])
        assert result.exit_code == 0

    def test_project_init_help(self):
        """Test 'cg project init' is registered."""
        result = runner.invoke(app, ["project", "init", "--help"])
        assert result.exit_code == 0

    def test_project_init_skip_all(self, temp_project_manager, sample_project_path):
        result = runner.invoke(
            app,
            ["project", "init", "run", str(sample_project_path), "--skip-setup", "--skip-index"],
        )
        assert result.exit_code == 0

    def test_project_watch_help(self):
        result = runner.invoke(app, ["project", "watch", "--help"])
        assert result.exit_code == 0


# ── Analyze group tests ──────────────────────────────────────────


class TestAnalyzeGroup:
    """Tests for 'cg analyze' group."""

    def test_analyze_help(self):
        result = runner.invoke(app, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "search" in result.stdout.lower()

    def test_analyze_health_help(self):
        result = runner.invoke(app, ["analyze", "health", "--help"])
        assert result.exit_code == 0

    def test_analyze_health_no_project(self, temp_project_manager):
        result = runner.invoke(app, ["analyze", "health", "dashboard"])
        assert result.exit_code != 0 or "No project" in result.stdout


# ── Chat group tests ─────────────────────────────────────────────


class TestChatGroup:
    """Tests for 'cg chat' group."""

    def test_chat_help(self):
        result = runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0
        assert "start" in result.stdout.lower()


# ── No top-level clutter tests ───────────────────────────────────


class TestNoTopLevelClutter:
    """Verify removed commands no longer appear at the top level."""

    def test_no_top_level_setup(self):
        result = runner.invoke(app, ["setup"])
        assert result.exit_code != 0  # should fail — not top-level

    def test_no_top_level_search(self):
        result = runner.invoke(app, ["search", "test"])
        assert result.exit_code != 0

    def test_no_top_level_undo(self):
        result = runner.invoke(app, ["undo"])
        assert result.exit_code != 0

    def test_no_top_level_aliases(self):
        for alias in ["find", "ask", "gen", "fix"]:
            result = runner.invoke(app, [alias, "--help"])
            assert result.exit_code != 0, f"Alias '{alias}' should NOT be top-level"

    def test_top_level_only_groups(self):
        """Top-level --help should show only the 6 groups."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        stdout = result.stdout
        # Groups must be visible
        for grp in ["config", "project", "analyze", "chat", "explore", "export"]:
            assert grp in stdout, f"Group '{grp}' missing from top-level help"
        # Removed groups must NOT be visible
        for removed in ["improve", "admin"]:
            assert removed not in stdout.split(), f"Removed group '{removed}' still at top-level"
        # Old flat commands must NOT be visible
        for old in ["setup", "set-llm", "index", "search", "debug-embed", "cheatsheet", "undo"]:
            assert old not in stdout.split(), f"Old command '{old}' still at top-level"


# ── Unit tests (modules) ────────────────────────────────────────


class TestProjectDetection:
    """Tests for project type detection."""

    def test_detect_python(self, temp_dir, monkeypatch):
        (temp_dir / "pyproject.toml").write_text("[project]\nname='test'\n")
        monkeypatch.chdir(temp_dir)
        from codegraph_cli.cli_quickstart import detect_project_type
        assert detect_project_type() == "Python"

    def test_detect_javascript(self, temp_dir, monkeypatch):
        (temp_dir / "package.json").write_text('{"name": "test"}')
        monkeypatch.chdir(temp_dir)
        from codegraph_cli.cli_quickstart import detect_project_type
        assert detect_project_type() == "JavaScript/TypeScript"

    def test_detect_unknown(self, temp_dir, monkeypatch):
        monkeypatch.chdir(temp_dir)
        from codegraph_cli.cli_quickstart import detect_project_type
        assert detect_project_type() == "Unknown"


class TestSuggestions:
    """Tests for contextual suggestions module."""

    def test_next_steps_known_command(self):
        from codegraph_cli.cli_suggestions import NEXT_STEPS
        assert "index" in NEXT_STEPS
        assert "search" in NEXT_STEPS
        assert len(NEXT_STEPS["index"]) > 0

    def test_next_steps_unknown_command(self):
        from codegraph_cli.cli_suggestions import NEXT_STEPS
        assert "nonexistent_command" not in NEXT_STEPS


class TestAutoContext:
    """Tests for auto-context gathering in chat sessions."""

    def test_gather_auto_context_no_source(self):
        from codegraph_cli.cli_chat import _gather_auto_context

        mock_ctx = MagicMock()
        mock_ctx.has_source_access = False
        mock_ctx.get_project_summary.return_value = {
            "project_name": "test", "indexed_files": 5,
            "total_nodes": 20, "node_types": {"function": 10, "class": 3},
        }
        result = _gather_auto_context(mock_ctx)
        assert result["summary"]["indexed_files"] == 5
        assert result["recent_files"] == []

    def test_gather_auto_context_with_source(self):
        from codegraph_cli.cli_chat import _gather_auto_context

        mock_ctx = MagicMock()
        mock_ctx.has_source_access = True
        mock_ctx.get_project_summary.return_value = {
            "project_name": "test", "indexed_files": 3,
            "total_nodes": 15, "node_types": {"function": 8, "class": 2},
        }
        mock_ctx.list_directory.return_value = [
            {"name": "app.py", "type": "file", "modified": "2026-02-14T10:00:00"},
            {"name": "utils.py", "type": "file", "modified": "2026-02-13T08:00:00"},
            {"name": "tests", "type": "dir", "modified": "2026-02-12T06:00:00"},
        ]
        result = _gather_auto_context(mock_ctx)
        assert "app.py" in result["recent_files"]
        assert "tests" not in result["recent_files"]

    def test_gather_auto_context_exception_handling(self):
        from codegraph_cli.cli_chat import _gather_auto_context

        mock_ctx = MagicMock()
        mock_ctx.get_project_summary.side_effect = RuntimeError("DB error")
        mock_ctx.has_source_access = True
        mock_ctx.list_directory.side_effect = OSError("Permission denied")
        result = _gather_auto_context(mock_ctx)
        assert result["summary"] == {}
        assert result["recent_files"] == []


class TestFuzzyMatching:
    """Tests for fuzzy command matching infrastructure."""

    def test_get_all_command_names(self):
        from codegraph_cli.cli import _get_all_command_names
        names = _get_all_command_names()
        for grp in ["config", "project", "analyze", "chat", "explore", "export"]:
            assert grp in names, f"Group '{grp}' missing from command names"
        # Removed groups must not appear
        for removed in ["improve", "admin"]:
            assert removed not in names, f"Removed group '{removed}' still in command names"

    def test_fuzzy_suggestions_typo(self):
        from difflib import get_close_matches
        from codegraph_cli.cli import _get_all_command_names
        all_commands = _get_all_command_names()
        suggestions = get_close_matches("analze", all_commands, n=3, cutoff=0.5)
        assert "analyze" in suggestions
