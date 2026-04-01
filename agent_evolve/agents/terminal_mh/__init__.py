"""Terminal-Bench agent with Meta-Harness scaffolding support.

Extends TerminalAgent with dynamic harness.py loading — the evolver
can modify prompt assembly, pre-solve setup, and user prompt construction
by writing Python functions into workspace/harness.py.
"""

from .agent import TerminalMHAgent

__all__ = ["TerminalMHAgent"]
