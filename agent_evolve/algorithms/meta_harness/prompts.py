"""Proposer prompts for MetaHarness.

The key design principle (from the Meta-Harness paper, Appendix D):
  "The skill should constrain outputs and safety-relevant behavior,
   not the proposer's diagnosis procedure: it should specify what is
   forbidden, what artifacts to produce, and what objectives to optimize,
   while leaving the model free to inspect scores, traces, and prior
   code as needed."
"""

from __future__ import annotations

from typing import Any

from ...contract.workspace import AgentWorkspace

# Minimal system prompt — role + constraints only.
# Diagnosis strategy is left entirely to the proposer.
PROPOSER_SYSTEM_PROMPT = """\
You are a harness optimizer.  Your job is to improve an AI agent's \
performance on a benchmark by mutating its workspace files and \
optional scaffolding code.

You have bash access to the workspace directory.  Use grep, cat, ls, \
and any standard CLI tools to inspect files.  Use file writes (cat <<'EOF', \
sed, etc.) to mutate them.
"""


def build_proposer_prompt(
    workspace: AgentWorkspace,
    cycle: int,
    score_curve: list[float],
    *,
    harness_enabled: bool = False,
) -> str:
    """Build the user-message prompt for one Meta-Harness evolution step.

    Intentionally minimal.  Tells the proposer:
      1. Directory layout (what lives where)
      2. What it CAN modify
      3. What it MUST NOT do
      4. The optimization objective
      5. Where to find traces and scores

    Everything else — what to read, how to diagnose, what to change —
    is left to the proposer.
    """
    skills = workspace.list_skills()
    skill_names = [s.name for s in skills]

    # Score history
    if score_curve:
        scores_str = " → ".join(f"{s:.3f}" for s in score_curve)
        latest = score_curve[-1]
    else:
        scores_str = "(no prior cycles)"
        latest = 0.0

    # Harness section
    harness_section = ""
    if harness_enabled:
        harness_path = workspace.root / "harness.py"
        harness_exists = harness_path.exists()
        harness_section = f"""
### Harness Code
- `harness.py` — agent scaffolding logic (prompt assembly, tool orchestration, etc.)
- Status: {"exists (" + str(harness_path.stat().st_size) + " bytes)" if harness_exists else "does not exist yet — you may create it"}
- The agent dynamically loads this file at runtime.  Changes take effect on next solve cycle.
- You may add functions, modify control flow, change prompt construction logic.
"""

    return f"""\
## Meta-Harness Evolution — Cycle {cycle}

### Objective
Improve the agent's benchmark score.  Current: {latest:.3f}.
Score history: {scores_str}

### Workspace Layout
```
{workspace.root}/
├── prompts/system.md        — agent system prompt
├── skills/*/SKILL.md        — on-demand skill library ({len(skill_names)} skills)
├── memory/*.jsonl           — episodic memory
├── tools/                   — tool implementations
├── evolution/
│   └── observations/        — batch_XXXX.jsonl files with FULL execution traces
│       Each JSONL record contains:
│         task_id, task_input, success, score, feedback_detail,
│         conversation (complete agent trace: every message, tool call, and output)
```
{harness_section}
### What You CAN Modify
- prompts/system.md
- skills/ (create, update, delete SKILL.md files)
- memory/*.jsonl (add insights, prune noise)
- tools/ (modify tool implementations)
{("- harness.py (scaffolding code)" if harness_enabled else "")}

### What You MUST NOT Do
- Do not hardcode task-specific answers or task IDs into any file.
- Do not delete or corrupt evolution/observations/ (read-only history).

### How to Work
1. Browse `evolution/observations/` — these contain **full execution traces** \
(every tool call, every output, every error).  Use `grep`, `cat`, `jq` to \
selectively inspect what you need.  You do NOT need to read everything.
2. Form hypotheses about what causes failures.
3. Make targeted changes to workspace files.
4. Verify your changes with `git diff` before finishing.

### Current Skills
{chr(10).join(f"- {s}" for s in skill_names) if skill_names else "None yet."}

When done, summarize what you changed and why in 2-3 sentences.
"""
