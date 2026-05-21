"""Tests for NexusMemo guidelines multi-target writer."""

import tempfile
from pathlib import Path

import pytest

from nexusmemo.guidelines import (
    TARGET_INFO,
    VALID_TARGETS,
    write_guidelines,
)


class TestGuidelinesTargets:
    def test_valid_targets_list(self):
        """All expected targets should be present."""
        expected = {"claude", "cursor", "windsurf", "copilot", "agents", "all"}
        assert expected == set(VALID_TARGETS)

    def test_target_info_has_required_keys(self):
        for target, info in TARGET_INFO.items():
            assert "file" in info, f"{target} missing 'file'"
            assert "label" in info, f"{target} missing 'label'"
            assert "format" in info, f"{target} missing 'format'"
            assert info["format"] in ("markdown", "mdc"), f"{target} has unknown format"

    def test_unknown_target_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = write_guidelines("unknown_tool", cwd=tmp)
        assert len(results) == 1
        assert "Unknown target" in results[0]


class TestGuidelinesWrite:
    def test_claude_creates_claude_md(self, tmp_path):
        results = write_guidelines("claude", cwd=str(tmp_path))
        assert len(results) == 1
        assert "Created" in results[0]
        claude_md = tmp_path / "CLAUDE.md"
        assert claude_md.exists()
        content = claude_md.read_text()
        assert "NexusMemo" in content

    def test_agents_creates_agents_md(self, tmp_path):
        results = write_guidelines("agents", cwd=str(tmp_path))
        assert (tmp_path / "AGENTS.md").exists()
        assert "Created" in results[0]

    def test_cursor_creates_mdc_with_frontmatter(self, tmp_path):
        results = write_guidelines("cursor", cwd=str(tmp_path))
        mdc = tmp_path / ".cursor" / "rules" / "nexusmemo.mdc"
        assert mdc.exists(), ".cursor/rules/nexusmemo.mdc should be created"
        content = mdc.read_text()
        assert content.startswith("---"), "MDC file should start with frontmatter"
        assert "alwaysApply: true" in content
        assert "NexusMemo" in content

    def test_windsurf_creates_windsurfrules(self, tmp_path):
        results = write_guidelines("windsurf", cwd=str(tmp_path))
        assert (tmp_path / ".windsurfrules").exists()

    def test_copilot_creates_nested_dir(self, tmp_path):
        results = write_guidelines("copilot", cwd=str(tmp_path))
        copilot_file = tmp_path / ".github" / "copilot-instructions.md"
        assert copilot_file.exists(), ".github/ directory should be created"
        assert "NexusMemo" in copilot_file.read_text()

    def test_all_target_creates_all_files(self, tmp_path):
        results = write_guidelines("all", cwd=str(tmp_path))
        assert len(results) == len(TARGET_INFO)

        expected_files = [info["file"] for info in TARGET_INFO.values()]
        for rel_path in expected_files:
            assert (tmp_path / rel_path).exists(), f"{rel_path} should exist"

    def test_skip_if_already_configured(self, tmp_path):
        # Write first time
        write_guidelines("claude", cwd=str(tmp_path))
        # Write second time — should skip
        results = write_guidelines("claude", cwd=str(tmp_path))
        assert "skipped" in results[0].lower() or "already" in results[0].lower()

    def test_append_to_existing_file_without_nexusmemo(self, tmp_path):
        existing = tmp_path / "CLAUDE.md"
        existing.write_text("# My Project\n\nSome existing content.\n")

        results = write_guidelines("claude", cwd=str(tmp_path))
        content = existing.read_text()

        assert "My Project" in content, "Existing content should be preserved"
        assert "NexusMemo" in content, "Guidelines should be appended"
        assert "Appended" in results[0]

    def test_all_files_contain_mcp_tool_list(self, tmp_path):
        """Every generated file should mention the core MCP tools."""
        write_guidelines("all", cwd=str(tmp_path))
        for target, info in TARGET_INFO.items():
            content = (tmp_path / info["file"]).read_text()
            assert "add_memory" in content, f"{target}: add_memory tool missing"
            assert "search_memory" in content, f"{target}: search_memory tool missing"
