"""Starter harness — scaffolding hooks for McpMHAgent.

The MetaHarness evolver can modify this file to change how the agent
assembles prompts, selects tools, and constructs user messages.

Available hooks (all optional — delete or leave unimplemented to use defaults):

  build_system_prompt(base_prompt: str, skills: list[SkillMeta], task_prompt: str | None) -> str
  build_user_prompt(task_id: str, task_input: str) -> str | None
  pre_solve(task_metadata: dict) -> dict
"""


def build_system_prompt(base_prompt: str, skills: list, task_prompt: str | None = None) -> str:
    """Assemble system prompt with skills.

    Default implementation: append skill content inline.
    The evolver may rewrite this to change skill injection strategy,
    add chain-of-thought instructions, etc.
    """
    parts = [base_prompt]

    if skills and task_prompt:
        # Simple keyword-based skill selection (matching McpAgent default)
        task_lower = task_prompt.lower()
        selected = []
        for skill in skills:
            keywords = skill.name.replace("-", " ").split()
            keywords += skill.description.lower().split()
            score = sum(1 for kw in keywords if kw in task_lower and len(kw) > 3)
            if score >= 2:
                selected.append((score, skill))
        selected.sort(key=lambda x: x[0], reverse=True)
        selected = [s for _, s in selected[:3]]

        if selected:
            parts.append("\n\n## Available Skills\n")
            for skill in selected:
                parts.append(f"- **{skill.name}**: {skill.description}")
    elif skills:
        parts.append("\n\n## Available Skills\n")
        for skill in skills:
            parts.append(f"- **{skill.name}**: {skill.description}")

    return "\n".join(parts)
