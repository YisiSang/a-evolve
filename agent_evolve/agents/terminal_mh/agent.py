"""TerminalMHAgent -- Terminal-Bench agent with Meta-Harness hooks.

Inherits all solving logic from TerminalAgent.  Adds three hook points
that harness.py can override:

  build_system_prompt(base_prompt: str, skills: list[SkillMeta]) -> str
      Override how the system prompt is assembled.

  build_user_prompt(task_name: str, task_input: str) -> str
      Override how the user prompt is built.

  pre_solve(container_name: str) -> str
      Run before the agent loop.  Returns extra context to append
      to the user prompt (e.g. environment bootstrapping).
      Return "" for no extra context.

If harness.py does not exist or a hook is not defined, the default
TerminalAgent behavior is used.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..terminal.agent import TerminalAgent
from ..terminal.tools import set_container_name, reset_submit_flag, reset_tool_counter
from ..terminal.docker_env import TB2Container, pull_image
from ...types import Task, Trajectory

logger = logging.getLogger(__name__)


class TerminalMHAgent(TerminalAgent):
    """Terminal-Bench agent with dynamic harness.py hook support."""

    def _build_system_prompt(self) -> str:
        hook = self.harness_hook("build_system_prompt")
        if hook:
            try:
                return hook(self.system_prompt, self.skills)
            except Exception as e:
                logger.warning("harness build_system_prompt failed: %s", e)
        return super()._build_system_prompt()

    def _build_user_prompt(self, task_name: str, prompt: str) -> str:
        hook = self.harness_hook("build_user_prompt")
        if hook:
            try:
                return hook(task_name, prompt)
            except Exception as e:
                logger.warning("harness build_user_prompt failed: %s", e)
        return super()._build_user_prompt(task_name, prompt)

    def solve(self, task: Task) -> Trajectory:
        """Solve with optional pre_solve harness hook.

        If harness.py defines pre_solve(container_name) -> str, its
        return value is appended to the user prompt as extra context
        (e.g. environment snapshot, pre-installed packages, etc.).
        """
        pre_solve_hook = self.harness_hook("pre_solve")
        if pre_solve_hook is None:
            # No hook — use parent solve directly
            return super().solve(task)

        # We need to intercept the solve flow to inject pre_solve output.
        # Replicate the parent's solve but insert the hook after container start.
        docker_image = task.metadata.get("docker_image", "")
        task_name = task.metadata.get("task_name", task.id)
        test_sh_path = task.metadata.get("test_sh_path", "")
        test_py_path = task.metadata.get("test_py_path")
        timeout_sec = task.metadata.get("agent_timeout_sec", 900)

        if not docker_image:
            raise ValueError(
                f"Task {task.id} missing 'docker_image' in metadata. "
                "TerminalMHAgent requires a Docker image."
            )

        pull_image(docker_image)
        container = TB2Container(docker_image)
        steps: list[dict] = []

        with container:
            set_container_name(container.container_name)
            reset_submit_flag()
            reset_tool_counter()

            # Run pre_solve hook
            extra_context = ""
            try:
                extra_context = pre_solve_hook(container.container_name) or ""
                if extra_context:
                    logger.info("pre_solve hook returned %d chars", len(extra_context))
            except Exception as e:
                logger.warning("harness pre_solve failed: %s", e)

            agent = self._build_strands_agent()
            user_prompt = self._build_user_prompt(task_name, task.input)
            if extra_context:
                user_prompt = f"{user_prompt}\n\n{extra_context}"

            logger.info(
                "Solving %s with image %s (timeout=%ds, harness=%s)",
                task_name, docker_image, timeout_sec,
                "loaded" if self.harness else "none",
            )

            import time as _time
            t0 = _time.time()
            response = self._run_with_timeout(agent, user_prompt, timeout_sec)
            solve_elapsed = _time.time() - t0
            logger.info("Agent finished in %.1fs", solve_elapsed)

            usage = {}
            if response:
                try:
                    u = response.metrics.accumulated_usage
                    usage = {
                        "input_tokens": u.get("inputTokens", 0),
                        "output_tokens": u.get("outputTokens", 0),
                        "total_tokens": u.get("totalTokens", 0),
                    }
                except Exception:
                    pass

            passed = False
            eval_output = ""
            if test_sh_path and __import__("os").path.exists(test_sh_path):
                self._copy_test_files(container, test_sh_path, test_py_path)
                logger.info("Running evaluation...")
                verifier_timeout = task.metadata.get("verifier_timeout_sec", 900)
                passed, eval_output = container.run_tests_with_retry(
                    test_sh_path, timeout=verifier_timeout, max_retries=3
                )
                logger.info("Evaluation: %s", "PASS" if passed else "FAIL")
            else:
                logger.warning("No test.sh found, skipping evaluation")

            conversation = []
            try:
                from ..terminal.agent import _extract_conversation
                conversation = _extract_conversation(agent.messages)
            except Exception:
                logger.debug("Could not extract conversation")

            steps.append({
                "llm_output": str(response)[:2000] if response else "(timeout)",
                "usage": usage,
                "passed": passed,
                "eval_output": eval_output[-2000:] if len(eval_output) > 2000 else eval_output,
                "conversation": conversation,
            })

            self.remember(
                f"Solved {task_name}: passed={passed}, "
                f"tokens={usage.get('input_tokens', 0) + usage.get('output_tokens', 0)}",
                category="episodic",
                task_id=task_name,
            )

        output = f"passed={passed}\n{eval_output}"
        return Trajectory(task_id=task.id, output=output, steps=steps)
