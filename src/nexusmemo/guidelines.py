"""Multi-target AI guidelines writer for NexusMemo.

Supported targets:
  claude   → CLAUDE.md               (Claude Code)
  cursor   → .cursor/rules/nexusmemo.mdc  (Cursor)
  windsurf → .windsurfrules          (Windsurf)
  copilot  → .github/copilot-instructions.md (GitHub Copilot)
  agents   → AGENTS.md               (Antigravity, Codex, other agents)
  all      → writes all of the above
"""

from __future__ import annotations

import os
from pathlib import Path

# ─── Target Registry ─────────────────────────────────────────────

TARGET_INFO: dict[str, dict] = {
    "claude": {
        "file": "CLAUDE.md",
        "label": "Claude Code",
        "format": "markdown",
    },
    "cursor": {
        "file": ".cursor/rules/nexusmemo.mdc",
        "label": "Cursor",
        "format": "mdc",
    },
    "windsurf": {
        "file": ".windsurfrules",
        "label": "Windsurf",
        "format": "markdown",
    },
    "copilot": {
        "file": ".github/copilot-instructions.md",
        "label": "GitHub Copilot",
        "format": "markdown",
    },
    "agents": {
        "file": "AGENTS.md",
        "label": "AI Agents (Antigravity, Codex, etc.)",
        "format": "markdown",
    },
}

VALID_TARGETS = list(TARGET_INFO.keys()) + ["all"]

# ─── Templates ───────────────────────────────────────────────────

_GUIDELINES_BODY = """
# NexusMemo (Persistent Memory) Usage Guidelines

As an AI assistant in this project, you are expected to actively and proactively use the `nexusmemo` MCP plugin to maintain long-term context.

1. **Recall Context Before Acting:** Before starting a complex task, making architectural changes, or fixing a deep bug, use the `search_memory` or `get_project_context` tools to retrieve historical decisions, known issues, and architectural rules.
2. **Proactively Save Important Decisions:** Whenever we make an architectural decision, add a new library, resolve a critical bug, or define a new business rule, immediately use the `add_memory` tool to save this information to the persistent memory without asking for permission.
3. **Maintain a Rich Knowledge Graph:** When adding memories, write clear and descriptive sentences so the underlying system can successfully extract specific Entities and Relations to build a highly connected knowledge graph.

## Available NexusMemo MCP Tools

- `add_memory` — Store project context, decisions, and architectural notes
- `search_memory` — Query past memories using natural language
- `get_project_context` — Get rich context about a topic from memories + knowledge graph
- `find_related` — Explore knowledge graph relationships for an entity
- `get_decisions` — Review past architectural decisions
- `memory_status` — Check system stats
""".lstrip()

_MDC_FRONTMATTER = """\
---
description: NexusMemo persistent memory usage guidelines
globs: "**/*"
alwaysApply: true
---

"""

# ─── Template Builder ─────────────────────────────────────────────

def _get_template(fmt: str) -> str:
    """Return the full file content for a given format."""
    if fmt == "mdc":
        return _MDC_FRONTMATTER + _GUIDELINES_BODY
    return _GUIDELINES_BODY


# ─── Public API ───────────────────────────────────────────────────

def write_guidelines(target: str, cwd: str | None = None) -> list[str]:
    """Write NexusMemo guidelines for the given target(s).

    Args:
        target: One of the VALID_TARGETS ('claude', 'cursor', 'windsurf',
                'copilot', 'agents', 'all').
        cwd:    Project root directory. Defaults to os.getcwd().

    Returns:
        List of human-readable result messages (one per file written).
    """
    if target not in VALID_TARGETS:
        return [
            f"Unknown target '{target}'. "
            f"Valid targets: {', '.join(VALID_TARGETS)}"
        ]

    cwd = cwd or os.getcwd()
    base = Path(cwd)
    targets = list(TARGET_INFO.keys()) if target == "all" else [target]

    results: list[str] = []
    for t in targets:
        info = TARGET_INFO[t]
        file_path = base / info["file"]
        template = _get_template(info["format"])
        label = info["label"]

        # Ensure parent directories exist (e.g. .cursor/rules/, .github/)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.exists():
            existing = file_path.read_text(encoding="utf-8")
            if "NexusMemo" in existing:
                results.append(
                    f"✓ {label} ({info['file']}): Already configured, skipped."
                )
                continue
            # Append to existing file (e.g. CLAUDE.md with other content)
            file_path.write_text(existing.rstrip() + "\n\n" + template, encoding="utf-8")
            results.append(
                f"✓ {label} ({info['file']}): Appended NexusMemo guidelines."
            )
        else:
            file_path.write_text(template, encoding="utf-8")
            results.append(
                f"✓ {label} ({info['file']}): Created with NexusMemo guidelines."
            )

    return results
