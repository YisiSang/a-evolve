"""Personalized MCP-Atlas benchmark adapter.

Extends McpAtlasBenchmark with persona-aware evaluation. Loads open-ended
tasks from personalized_tasks.json and evaluates using persona-dependent
claims instead of (or in addition to) fixed factual claims.

Does NOT modify the shared McpAtlasBenchmark — all persona logic lives here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .mcp_atlas import McpAtlasBenchmark
from .personas import Persona, PERSONAS, get_persona
from ...types import Feedback, Task, Trajectory

logger = logging.getLogger(__name__)

_TASKS_FILE = Path(__file__).parent / "personalized_tasks.json"


class PersonalizedMcpBenchmark(McpAtlasBenchmark):
    """MCP-Atlas benchmark with persona-aware evaluation.

    Supports two modes:
    1. Standard tasks (from HuggingFace) — evaluated with original claims.
    2. Personalized tasks (from personalized_tasks.json) — evaluated with
       base_claims + persona-specific claims.

    Usage:
        bench = PersonalizedMcpBenchmark(persona_id="exec")
        tasks = bench.get_personalized_tasks()
        # ... run agent ...
        feedback = bench.evaluate(task, trajectory)
    """

    def __init__(self, persona_id: str, **kwargs):
        super().__init__(**kwargs)
        self.persona_id = persona_id
        self.persona: Persona = get_persona(persona_id)
        self._personalized_tasks: list[dict] | None = None

    # ── Load personalized tasks ─────────────────────────────────────

    def _load_personalized_tasks(self) -> list[dict]:
        """Load tasks from personalized_tasks.json (cached)."""
        if self._personalized_tasks is None:
            with open(_TASKS_FILE) as f:
                data = json.load(f)
            self._personalized_tasks = data["tasks"]
        return self._personalized_tasks

    def get_personalized_tasks(self, limit: int = 100) -> list[Task]:
        """Return personalized Task objects with persona claims injected."""
        raw_tasks = self._load_personalized_tasks()
        tasks: list[Task] = []
        for t in raw_tasks[:limit]:
            persona_claims = t.get("persona_claims", {}).get(self.persona_id, [])
            base_claims = t.get("base_claims", [])
            tasks.append(Task(
                id=t["id"],
                input=t["prompt"],
                metadata={
                    "task_id": t["id"],
                    "enabled_tools": t.get("enabled_tools", []),
                    "mcp_server_names": t.get("mcp_server_names", []),
                    "mcp_server_config": {},
                    "category": t.get("category", ""),
                    "difficulty": "",
                    # Persona evaluation fields
                    "active_persona": self.persona_id,
                    "base_claims": base_claims,
                    "persona_claims": t.get("persona_claims", {}),
                    # Combine base + persona claims as expected_output
                    # so the parent evaluate() can use them directly.
                    "expected_output": json.dumps(base_claims + persona_claims),
                },
            ))
        return tasks

    # ── Override evaluate to add persona context to judge ────────────

    def _get_evaluation_prompt(self, claim: str, response: str) -> str:
        """Augment the judge prompt with persona context.

        This helps the LLM judge evaluate style/format claims more
        accurately by understanding the persona's expectations.
        """
        persona_context = (
            f"EVALUATOR CONTEXT:\n"
            f"You are evaluating this response from the perspective of a "
            f"'{self.persona.name}' user ({self.persona.role}). "
            f"{self.persona.description}\n"
            f"User profile: expertise={self.persona.expertise}, "
            f"scope={self.persona.scope}, task_stage={self.persona.task_stage}, "
            f"register={self.persona.register}. "
            f"Output preferences: verbosity={self.persona.verbosity}, "
            f"format={self.persona.format}, "
            f"reasoning_display={self.persona.reasoning_display}.\n"
            f"Evaluate whether the claim is met given these expectations.\n\n"
        )
        base_prompt = super()._get_evaluation_prompt(claim, response)
        return persona_context + base_prompt
