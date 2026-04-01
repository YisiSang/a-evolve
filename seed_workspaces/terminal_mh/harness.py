"""Starter harness — scaffolding hooks for TerminalMHAgent.

The MetaHarness evolver can modify this file to change how the agent
assembles prompts, prepares the environment, and constructs user messages.

Available hooks (all optional — delete or leave unimplemented to use defaults):

  build_system_prompt(base_prompt: str, skills: list[SkillMeta]) -> str
  build_user_prompt(task_name: str, task_input: str) -> str
  pre_solve(container_name: str) -> str
"""

import subprocess


def pre_solve(container_name: str) -> str:
    """Gather an environment snapshot before the agent loop starts.

    Returns a text block that gets appended to the user prompt,
    giving the agent immediate awareness of the sandbox environment
    without wasting turns on exploratory commands.
    """
    cmd = (
        "echo '@@PWD@@' && pwd && "
        "echo '@@LS@@' && ls -la /app/ 2>/dev/null && "
        "echo '@@LANG@@' && "
        "(python3 --version 2>&1 || echo 'python3: not found') && "
        "(gcc --version 2>&1 | head -1 || echo 'gcc: not found') && "
        "(node --version 2>&1 || echo 'node: not found') && "
        "(rustc --version 2>&1 || echo 'rustc: not found') && "
        "(go version 2>&1 || echo 'go: not found') && "
        "echo '@@PKG@@' && "
        "(pip3 --version 2>&1 || echo 'pip3: not found') && "
        "(apt-get --version 2>&1 | head -1 || echo 'apt-get: not found') && "
        "echo '@@MEM@@' && free -h 2>/dev/null | head -2 || true"
    )

    try:
        result = subprocess.run(
            ["docker", "exec", container_name, "bash", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=15,
        )
        stdout = result.stdout.strip()
    except Exception:
        return ""

    if not stdout:
        return ""

    # Parse sections
    sections = {}
    current_key = None
    current_lines = []
    for line in stdout.split("\n"):
        if line.startswith("@@") and line.endswith("@@"):
            if current_key:
                sections[current_key] = "\n".join(current_lines)
            current_key = line.strip("@")
            current_lines = []
        else:
            current_lines.append(line)
    if current_key:
        sections[current_key] = "\n".join(current_lines)

    parts = []
    if "PWD" in sections:
        parts.append(f"Working directory: {sections['PWD'].strip()}")
    if "LS" in sections:
        ls_lines = sections["LS"].strip().split("\n")
        if len(ls_lines) > 25:
            parts.append(
                f"/app contents ({len(ls_lines)} entries):\n"
                + "\n".join(ls_lines[:20])
                + f"\n... ({len(ls_lines) - 20} more files)"
            )
        else:
            parts.append(f"/app contents:\n{sections['LS'].strip()}")
    if "LANG" in sections:
        lang_lines = [l.strip() for l in sections["LANG"].strip().split("\n") if l.strip()]
        parts.append("Available languages/tools: " + "; ".join(lang_lines))
    if "PKG" in sections:
        pkg_lines = [l.strip() for l in sections["PKG"].strip().split("\n") if l.strip()]
        parts.append("Package managers: " + "; ".join(pkg_lines))
    if "MEM" in sections:
        mem = sections["MEM"].strip()
        if mem:
            parts.append(f"Memory: {mem}")

    if not parts:
        return ""

    return "[Environment Snapshot]\n" + "\n".join(parts)
