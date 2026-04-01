"""MetaHarnessEngine -- evolution via full-trace filesystem access.

Core differences from AdaptiveSkillEngine:
  1. Full traces: observations are NOT compressed into the prompt.
     They live on the filesystem; the proposer browses them via bash.
  2. All history: no last_n_cycles limit.  The proposer sees every
     prior cycle's traces and can grep/cat selectively.
  3. Minimal prompt: the proposer receives only the directory layout,
     permissions, and objective.  Diagnosis strategy is its own.
  4. Harness mutation (optional): when enabled, the proposer can also
     modify a harness.py file containing agent scaffolding logic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ...config import EvolveConfig
from ...contract.workspace import AgentWorkspace
from ...engine.base import EvolutionEngine
from ...engine.history import EvolutionHistory
from ...engine.trial import TrialRunner
from ...llm.base import LLMMessage, LLMProvider
from ...types import Observation, StepResult
from .prompts import PROPOSER_SYSTEM_PROMPT, build_proposer_prompt

logger = logging.getLogger(__name__)


class MetaHarnessEngine(EvolutionEngine):
    """LLM-driven evolution with full-trace filesystem access.

    The proposer LLM gets bash access to the entire workspace including
    the ``evolution/observations/`` directory containing raw execution
    traces from all prior cycles.  It decides what to read, how to
    diagnose failures, and what to mutate.
    """

    def __init__(self, config: EvolveConfig, llm: LLMProvider | None = None):
        self.config = config
        self._llm = llm
        self.harness_enabled: bool = config.extra.get("harness_enabled", False)

    @property
    def llm(self) -> LLMProvider:
        if self._llm is None:
            from ..adaptive_skill.tools import create_default_llm
            self._llm = create_default_llm(self.config)
        return self._llm

    def step(
        self,
        workspace: AgentWorkspace,
        observations: list[Observation],
        history: EvolutionHistory,
        trial: TrialRunner,
    ) -> StepResult:
        """Run one Meta-Harness evolution step.

        Unlike other engines, we do NOT inject observation data into the
        prompt.  The observations are already persisted as JSONL files in
        ``evolution/observations/`` by the loop's Observer.  We simply
        tell the proposer where to find them.
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

        # Run proposer with bash tool access
        response = self._run_proposer(prompt, workspace.root)

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
        summary = f"MetaHarness cycle {cycle_num}: {', '.join(changes)}" if changes else f"MetaHarness cycle {cycle_num}: no mutation"

        return StepResult(
            mutated=mutated,
            summary=summary,
            metadata={
                "cycle": cycle_num,
                "changes": changes,
                "skills_before": len(skills_before),
                "skills_after": len(skills_after),
                "harness_enabled": self.harness_enabled,
                "usage": response.get("usage", {}),
            },
        )

    def _run_proposer(self, prompt: str, workspace_root: Path) -> dict[str, Any]:
        """Run the proposer LLM with bash access to the full workspace."""
        from ..adaptive_skill.tools import BASH_TOOL_SPEC, make_workspace_bash

        bash_fn = make_workspace_bash(workspace_root)

        try:
            from ...llm.bedrock import BedrockProvider

            if isinstance(self.llm, BedrockProvider):
                response = self.llm.converse_loop(
                    system_prompt=PROPOSER_SYSTEM_PROMPT,
                    user_message=prompt,
                    tools=[BASH_TOOL_SPEC],
                    tool_executor={"workspace_bash": lambda command: bash_fn(command)},
                    max_tokens=self.config.evolver_max_tokens,
                )
                return {
                    "content": response.content,
                    "usage": response.usage,
                }
        except ImportError:
            pass

        # Fallback: non-Bedrock providers (no tool loop, single completion)
        messages = [
            LLMMessage(role="system", content=PROPOSER_SYSTEM_PROMPT),
            LLMMessage(role="user", content=prompt),
        ]
        response = self.llm.complete(
            messages, max_tokens=self.config.evolver_max_tokens
        )
        return {
            "content": response.content,
            "usage": response.usage,
        }


def _read_harness(workspace: AgentWorkspace) -> str | None:
    """Read harness.py from the workspace root, or None if absent."""
    path = workspace.root / "harness.py"
    return path.read_text() if path.exists() else None
