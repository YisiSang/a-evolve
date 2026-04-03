#!/usr/bin/env python3
"""Quick test: run a few personalized tasks under different personas.

Usage:
    uv run python examples/mcp_examples/test_personalized.py \
        --task personal_001 --personas exec,ds \
        --workers 1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.setrecursionlimit(4000)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("BYPASS_TOOL_CONSENT", "true")

# Load .env if present
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

from agent_evolve.benchmarks.mcp_atlas.personalized_benchmark import PersonalizedMcpBenchmark
from agent_evolve.benchmarks.mcp_atlas.personas import list_personas, get_persona
from agent_evolve.agents.mcp.docker_env import McpAtlasContainer, pull_image
from agent_evolve.agents.mcp.mcp_client import McpClientWrapper
from agent_evolve.agents.mcp.key_registry import KeyRegistry
from agent_evolve.agents.mcp_mh.agent import McpMHAgent
from agent_evolve.types import Task

log = logging.getLogger("test_personalized")


def run_task_with_persona(
    agent: McpMHAgent,
    benchmark: PersonalizedMcpBenchmark,
    task: Task,
    client: McpClientWrapper,
) -> dict:
    """Solve one task and evaluate it."""
    t0 = time.time()
    try:
        trajectory = agent.solve(task, shared_client=client)
        feedback = benchmark.evaluate(task, trajectory)
        elapsed = time.time() - t0
        return {
            "task_id": task.id,
            "persona": benchmark.persona_id,
            "score": feedback.score,
            "success": feedback.success,
            "detail": feedback.detail,
            "output_preview": str(trajectory.output or "")[:500],
            "elapsed_sec": round(elapsed, 1),
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "task_id": task.id,
            "persona": benchmark.persona_id,
            "score": 0.0,
            "success": False,
            "detail": f"Error: {e}",
            "output_preview": "",
            "elapsed_sec": round(elapsed, 1),
        }


def main():
    p = argparse.ArgumentParser(description="Test personalized evaluation")
    p.add_argument("--task", type=str, default=None,
                   help="Task ID to run (default: first task)")
    p.add_argument("--personas", type=str, default="exec,ds",
                   help="Comma-separated persona IDs (default: exec,ds)")
    p.add_argument("--model", type=str, default="us.anthropic.claude-opus-4-6-v1")
    p.add_argument("--region", type=str, default="us-west-2")
    p.add_argument("--docker-image", type=str,
                   default="ghcr.io/scaleapi/mcp-atlas:latest")
    p.add_argument("--workspace", type=str, default="seed_workspaces/mcp_mh",
                   help="Agent workspace (default: seed_workspaces/mcp_mh)")
    p.add_argument("--eval-model", type=str, default="us.anthropic.claude-opus-4-6-v1")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("test_personalized").setLevel(logging.INFO)

    persona_ids = [x.strip() for x in args.personas.split(",")]
    log.info("Personas: %s", persona_ids)
    log.info("Available personas: %s", list_personas())

    # Collect env vars for MCP servers
    key_registry = KeyRegistry()

    # Start Docker container
    env_vars = {}
    if key_registry:
        # Get keys for all possible servers
        all_servers = set()
        for pid in persona_ids:
            bench = PersonalizedMcpBenchmark(
                persona_id=pid, eval_model_id=args.eval_model, use_litellm=False,
            )
            for t in bench.get_personalized_tasks():
                all_servers.update(t.metadata.get("mcp_server_names", []))
        env_vars = key_registry.get_keys_for_servers(list(all_servers))

    pull_image(args.docker_image)
    container = McpAtlasContainer(args.docker_image, env_vars=env_vars)
    container.start()
    client = McpClientWrapper(base_url=container.base_url)
    log.info("Container started at %s", container.base_url)

    try:
        results = []
        for pid in persona_ids:
            persona = get_persona(pid)
            log.info("\n%s", "=" * 60)
            log.info("  Persona: %s (%s)", persona.name, persona.role)
            log.info("  Verbosity: %s, Format: %s, Depth: %s",
                     persona.verbosity, persona.format, persona.depth)
            log.info("%s", "=" * 60)

            bench = PersonalizedMcpBenchmark(
                persona_id=pid, eval_model_id=args.eval_model, use_litellm=False,
            )
            tasks = bench.get_personalized_tasks()

            # Filter to specific task if requested
            if args.task:
                tasks = [t for t in tasks if t.id == args.task]
                if not tasks:
                    log.error("Task %s not found", args.task)
                    continue

            # Just run the first task for quick testing
            task = tasks[0]
            log.info("Running task: %s", task.id)
            log.info("Prompt: %s", task.input[:200])

            agent = McpMHAgent(
                workspace_dir=Path(args.workspace),
                model_id=args.model,
                region=args.region,
                max_tokens=16384,
                docker_image=args.docker_image,
                key_registry=key_registry,
            )

            result = run_task_with_persona(agent, bench, task, client)
            results.append(result)

            log.info("Score: %.3f | Success: %s | Time: %.1fs",
                     result["score"], result["success"], result["elapsed_sec"])
            log.info("Detail: %s", result["detail"])
            log.info("Output preview:\n%s", result["output_preview"])

        # Summary
        print(f"\n{'=' * 60}")
        print(f"  RESULTS SUMMARY")
        print(f"{'=' * 60}")
        for r in results:
            print(f"  [{r['persona']:10s}] {r['task_id']}: "
                  f"score={r['score']:.3f} success={r['success']} "
                  f"time={r['elapsed_sec']}s")
        print(f"{'=' * 60}")

        # Save results
        out_path = Path("personalized_test_results.json")
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {out_path}")

    finally:
        client.close()
        try:
            container.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
