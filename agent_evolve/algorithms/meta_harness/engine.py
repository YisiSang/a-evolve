"""MetaHarnessEngine -- evolution via Claude Code as proposer.

Faithful to the Meta-Harness paper (Lee et al., 2026):
  - Proposer is Claude Code CLI with Opus 4.6 via Bedrock
  - Claude Code gets full filesystem access to the workspace
    (including execution traces in evolution/observations/)
  - A minimal "skill" prompt steers the search
  - Claude Code decides what to inspect and how to mutate
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from ...config import EvolveConfig
from ...contract.workspace import AgentWorkspace
from ...engine.base import EvolutionEngine
from ...engine.history import EvolutionHistory
from ...engine.trial import TrialRunner
from ...types import Observation, StepResult
from .prompts import PROPOSER_SYSTEM_PROMPT, build_proposer_prompt

logger = logging.getLogger(__name__)

# Default model: Bedrock Opus 4.6 (same as the paper)
DEFAULT_MODEL = "bedrock:us.anthropic.claude-opus-4-6-v1"


class MetaHarnessEngine(EvolutionEngine):
    """Evolution engine that uses Claude Code CLI as the proposer.

    Claude Code browses the workspace filesystem — including raw
    execution traces in ``evolution/observations/`` — and mutates
    workspace files (prompts, skills, memory, tools, harness.py).
    """

    def __init__(self, config: EvolveConfig):
        self.config = config
        self.harness_enabled: bool = config.extra.get("harness_enabled", False)
        self.model: str = config.extra.get("proposer_model", DEFAULT_MODEL)
        self.max_turns: int = config.extra.get("proposer_max_turns", 50)
        self.timeout_sec: int = config.extra.get("proposer_timeout_sec", 600)

    def step(
        self,
        workspace: AgentWorkspace,
        observations: list[Observation],
        history: EvolutionHistory,
        trial: TrialRunner,
    ) -> StepResult:
        """Run one Meta-Harness evolution step.

        Observations are already persisted as JSONL files in
        ``evolution/observations/`` by the loop's Observer.  We tell
        Claude Code where to find them via the proposer prompt; it
        decides what to read.
        """
        cycle_num = history.latest_cycle + 1
        score_curve = history.get_score_curve()

        # Snapshot workspace state before mutation
        skills_before = {s.name for s in workspace.list_skills()}
        prompt_before = workspace.read_prompt()
        harness_before = _read_harness(workspace) if self.harness_enabled else None

        # Build minimal proposer prompt
        prompt = build_proposer_prompt(
            workspace,
            cycle_num,
            score_curve,
            harness_enabled=self.harness_enabled,
        )

        # Run Claude Code as the proposer
        result = self._run_claude_code(prompt, workspace.root)

        # Detect what changed
        skills_after = {s.name for s in workspace.list_skills()}
        prompt_after = workspace.read_prompt()
        harness_after = _read_harness(workspace) if self.harness_enabled else None

        changes = []
        new_skills = skills_after - skills_before
        removed_skills = skills_before - skills_after
        if new_skills:
            changes.append(f"+{len(new_skills)} skills")
        if removed_skills:
            changes.append(f"-{len(removed_skills)} skills")
        if prompt_after != prompt_before:
            changes.append("prompt modified")
        if self.harness_enabled and harness_after != harness_before:
            changes.append("harness.py modified")

        mutated = bool(changes)
        summary = (
            f"MetaHarness cycle {cycle_num}: {', '.join(changes)}"
            if changes
            else f"MetaHarness cycle {cycle_num}: no mutation"
        )

        return StepResult(
            mutated=mutated,
            summary=summary,
            metadata={
                "cycle": cycle_num,
                "changes": changes,
                "skills_before": len(skills_before),
                "skills_after": len(skills_after),
                "harness_enabled": self.harness_enabled,
                "proposer_model": self.model,
                "proposer_exit_code": result.get("exit_code"),
                "proposer_output_chars": len(result.get("output", "")),
            },
        )

    def _run_claude_code(self, prompt: str, workspace_root: Path) -> dict[str, Any]:
        """Invoke Claude Code CLI as the proposer.

        Runs in non-interactive mode (-p) with:
          - --model: Bedrock Opus 4.6
          - --system-prompt: minimal Meta-Harness skill
          - --dangerously-skip-permissions: no interactive approval
          - cwd: workspace root (Claude Code sees the full tree)
        """
        cmd = [
            "claude",
            "-p", prompt,
            "--model", self.model,
            "--system-prompt", PROPOSER_SYSTEM_PROMPT,
            "--dangerously-skip-permissions",
            "--output-format", "json",
            "--no-session-persistence",
            "--bare",
        ]

        logger.info(
            "Running Claude Code proposer (model=%s, cwd=%s)",
            self.model,
            workspace_root,
        )

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                cwd=str(workspace_root),
            )

            output = proc.stdout.strip()
            stderr = proc.stderr.strip()

            if proc.returncode != 0:
                logger.warning(
                    "Claude Code exited with code %d: %s",
                    proc.returncode,
                    stderr[:500],
                )

            # Try to parse JSON output for structured result
            result_text = output
            try:
                parsed = json.loads(output)
                result_text = parsed.get("result", output)
            except (json.JSONDecodeError, TypeError):
                pass

            logger.info(
                "Claude Code finished (exit=%d, output=%d chars)",
                proc.returncode,
                len(output),
            )

            return {
                "output": result_text,
                "stderr": stderr,
                "exit_code": proc.returncode,
            }

        except subprocess.TimeoutExpired:
            logger.error(
                "Claude Code timed out after %ds", self.timeout_sec
            )
            return {
                "output": "",
                "stderr": "TIMEOUT",
                "exit_code": -1,
            }
        except FileNotFoundError:
            logger.error(
                "Claude Code CLI not found. Install it: "
                "https://docs.anthropic.com/en/docs/claude-code"
            )
            return {
                "output": "",
                "stderr": "claude CLI not found",
                "exit_code": -1,
            }


def _read_harness(workspace: AgentWorkspace) -> str | None:
    """Read harness.py from the workspace root, or None if absent."""
    path = workspace.root / "harness.py"
    return path.read_text() if path.exists() else None
