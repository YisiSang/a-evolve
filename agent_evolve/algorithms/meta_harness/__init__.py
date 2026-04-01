"""MetaHarness -- evolution via full-trace filesystem access.

Inspired by Meta-Harness (Lee et al., 2026): give the proposer LLM
unrestricted filesystem access to all prior candidates' source code,
execution traces, and scores.  The proposer decides what to inspect
and how to mutate the workspace — including an optional harness.py
that contains agent scaffolding logic.
"""

from .engine import MetaHarnessEngine

__all__ = ["MetaHarnessEngine"]
