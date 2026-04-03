#!/usr/bin/env python3
"""MetaHarness experiment on Terminal-Bench 2.0.

Implements the Meta-Harness search loop (Lee et al., 2026, arXiv:2603.28052)
using a-evolve's infrastructure:

  Phase 0 — Baseline: solve all tasks with seed workspace (no evolution)
  Phase 1 — Evolution: 10 cycles of MetaHarness (proposer → validate → eval → archive)
  Phase 2 — Final eval: solve all tasks with evolved workspace

Usage:
    # Full experiment (all 3 phases)
    uv run python examples/tb_examples/run_metaharness.py \\
        --config examples/configs/metaharness_tbench2.yaml \\
        --run-name opus_v1

    # Quick smoke test (2 tasks, 1 cycle)
    uv run python examples/tb_examples/run_metaharness.py \\
        --config examples/configs/metaharness_tbench2.yaml \\
        --run-name test --eval-sample-size 2 --max-cycles 1 --workers 1

    # Budget run (20-task subsample during search)
    uv run python examples/tb_examples/run_metaharness.py \\
        --config examples/configs/metaharness_tbench2.yaml \\
        --run-name budget_v1 --eval-sample-size 20

    # Run specific phase
    uv run python examples/tb_examples/run_metaharness.py \\
        --config examples/configs/metaharness_tbench2.yaml \\
        --run-name opus_v1 --phase 1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time

sys.setrecursionlimit(4000)
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("BYPASS_TOOL_CONSENT", "true")

from agent_evolve.agents.terminal_mh.agent import TerminalMHAgent
from agent_evolve.algorithms.meta_harness import MetaHarnessEngine
from agent_evolve.benchmarks.tb2.terminal2 import Terminal2Benchmark
from agent_evolve.config import EvolveConfig
from agent_evolve.engine.history import EvolutionHistory
from agent_evolve.engine.observer import Observer
from agent_evolve.engine.trial import TrialRunner
from agent_evolve.engine.versioning import VersionControl
from agent_evolve.types import CycleRecord, Feedback, Observation, Task, Trajectory

log = logging.getLogger("metaharness")


# ---------------------------------------------------------------------------
# Parallel TrialRunner — wraps TrialRunner with ThreadPoolExecutor
# ---------------------------------------------------------------------------

class ParallelTrialRunner(TrialRunner):
    """TrialRunner that evaluates tasks in parallel using threads.

    The base TrialRunner.run_tasks() is sequential.  For 89 TB2 tasks
    at ~5-15 min each, parallel execution is essential.
    """

    def __init__(self, agent, benchmark, max_workers: int = 6):
        super().__init__(agent, benchmark)
        self.max_workers = max_workers

    def run_tasks(self, tasks: list[Task]) -> list[Observation]:
        if len(tasks) <= 1 or self.max_workers <= 1:
            return super().run_tasks(tasks)

        results: list[Observation] = []
        errors = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._run_one, task): task for task in tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    obs = future.result()
                    if obs is not None:
                        results.append(obs)
                except Exception as e:
                    errors += 1
                    log.error("ParallelTrialRunner: task %s failed: %s", task.id, e)

        if errors:
            log.warning("%d/%d tasks failed during parallel eval", errors, len(tasks))
        return results

    def _run_one(self, task: Task) -> Observation | None:
        try:
            trajectory = self._agent.solve(task)
            feedback = self._benchmark.evaluate(task, trajectory)
            return Observation(task=task, trajectory=trajectory, feedback=feedback)
        except Exception as e:
            log.error("Task %s error: %s", task.id, e)
            return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(observations: list[Observation]) -> dict:
    total = len(observations)
    passed = sum(1 for o in observations if o.feedback.success)
    by_difficulty = defaultdict(lambda: {"total": 0, "passed": 0})
    by_category = defaultdict(lambda: {"total": 0, "passed": 0})

    for o in observations:
        diff = o.task.metadata.get("difficulty", "unknown")
        cat = o.task.metadata.get("category", "unknown")
        by_difficulty[diff]["total"] += 1
        by_category[cat]["total"] += 1
        if o.feedback.success:
            by_difficulty[diff]["passed"] += 1
            by_category[cat]["passed"] += 1

    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total > 0 else 0.0,
        "by_difficulty": dict(by_difficulty),
        "by_category": dict(by_category),
    }


def print_metrics(label: str, metrics: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Total: {metrics['total']}  Passed: {metrics['passed']}  "
          f"Rate: {metrics['pass_rate']:.1%}")

    if metrics["by_difficulty"]:
        print(f"\n  By Difficulty:")
        for diff in ["easy", "medium", "hard"]:
            d = metrics["by_difficulty"].get(diff)
            if d:
                rate = d["passed"] / d["total"] if d["total"] > 0 else 0.0
                print(f"    {diff:<10} {d['passed']}/{d['total']} ({rate:.1%})")

    if metrics["by_category"]:
        print(f"\n  By Category:")
        cats = sorted(metrics["by_category"].items(),
                      key=lambda x: x[1]["passed"] / max(x[1]["total"], 1),
                      reverse=True)
        for cat, d in cats:
            rate = d["passed"] / d["total"] if d["total"] > 0 else 0.0
            print(f"    {cat:<35} {d['passed']}/{d['total']} ({rate:.1%})")
    print()


# ---------------------------------------------------------------------------
# Phase 0 — Baseline evaluation
# ---------------------------------------------------------------------------

def run_baseline(
    agent: TerminalMHAgent,
    benchmark: Terminal2Benchmark,
    trial: ParallelTrialRunner,
    observer: Observer,
    versioning: VersionControl,
    history: EvolutionHistory,
    tasks: list[Task],
) -> list[Observation]:
    """Solve all tasks with the seed workspace. No evolution."""
    print(f"\n{'=' * 60}")
    print(f"  PHASE 0: Baseline Evaluation ({len(tasks)} tasks)")
    print(f"{'=' * 60}\n")

    t0 = time.time()
    observations = trial.run_tasks(tasks)
    elapsed = time.time() - t0

    # Collect observations
    batch_path = observer.collect(observations)

    score = sum(o.feedback.score for o in observations) / len(observations) if observations else 0.0
    print(f"Baseline: {score:.1%} ({sum(1 for o in observations if o.feedback.success)}"
          f"/{len(observations)} passed) in {elapsed:.0f}s")

    # Commit baseline state
    versioning.commit(
        message=f"baseline: score={score:.3f}",
        tag="baseline",
    )

    # Record as cycle 0
    record = CycleRecord(
        cycle=0,
        score=score,
        mutated=False,
        engine_name="baseline",
        summary=f"Baseline evaluation: {score:.3f}",
        observation_batch=batch_path.name,
    )
    history.record_cycle(record)

    metrics = compute_metrics(observations)
    print_metrics("Baseline Results", metrics)

    return observations


# ---------------------------------------------------------------------------
# Phase 1 — MetaHarness evolution
# ---------------------------------------------------------------------------

def run_evolution(
    engine: MetaHarnessEngine,
    agent: TerminalMHAgent,
    trial: ParallelTrialRunner,
    observer: Observer,
    versioning: VersionControl,
    history: EvolutionHistory,
    observations: list[Observation],
    max_cycles: int,
    config: EvolveConfig,
    tasks: list[Task] | None = None,
    eval_factory: Callable | None = None,
) -> float:
    """Run the MetaHarness search loop for max_cycles iterations."""
    print(f"\n{'=' * 60}")
    print(f"  PHASE 1: MetaHarness Evolution ({max_cycles} cycles, "
          f"k={engine.num_candidates})")
    print(f"{'=' * 60}\n")

    score_history = history.get_score_curve()
    best_score = max(score_history) if score_history else 0.0

    for cycle in range(1, max_cycles + 1):
        cycle_t0 = time.time()
        print(f"\n--- Cycle {cycle}/{max_cycles} "
              f"(best so far: {best_score:.1%}) ---")

        # Engine step: proposer → validate → eval → archive → select
        step_result = engine.step(
            workspace=agent.workspace,
            observations=observations,
            history=history,
            trial=trial,
            tasks=tasks,
            eval_factory=eval_factory,
        )

        cycle_elapsed = time.time() - cycle_t0

        # Extract score from metadata
        cycle_score = step_result.metadata.get("best_score", 0.0)
        best_score = max(best_score, cycle_score)

        # Post-step bookkeeping
        tag = f"evo-{cycle}"
        if step_result.mutated:
            versioning.commit(
                message=f"evo-{cycle}: {step_result.summary}",
                tag=tag,
            )
        else:
            versioning.commit(
                message=f"evo-{cycle}: no mutation",
                tag=tag,
            )

        record = CycleRecord(
            cycle=cycle,
            score=cycle_score,
            mutated=step_result.mutated,
            engine_name="MetaHarnessEngine",
            summary=step_result.summary,
            metadata=step_result.metadata,
        )
        history.record_cycle(record)

        # Reload workspace (picks up evolved harness.py, prompts, etc.)
        agent.reload_from_fs()
        engine.on_cycle_end(accepted=step_result.mutated, score=cycle_score)

        # Log
        _append_history(agent.workspace.root / "evolution", cycle, cycle_score, step_result.mutated)
        _write_metrics(agent.workspace.root / "evolution", history.get_score_curve())

        print(f"Cycle {cycle}: score={cycle_score:.1%} mutated={step_result.mutated} "
              f"({cycle_elapsed:.0f}s) | {step_result.summary}")

    return best_score


# ---------------------------------------------------------------------------
# Phase 2 — Final evaluation
# ---------------------------------------------------------------------------

def _restore_best_candidate(work_dir: Path, agent: TerminalMHAgent) -> None:
    """Select the best candidate from the archive and restore its snapshot.

    Implements the paper's "return Pareto frontier" step: scan all candidates,
    compute the Pareto frontier across (score↑, cost↓), select the highest-
    scoring candidate on the frontier, and copy its snapshot/ files into the
    workspace so Phase 2 evaluates the best harness rather than the last one.
    """
    candidates_dir = work_dir / "evolution" / "candidates"
    if not candidates_dir.exists():
        log.warning("No candidates directory found at %s — skipping archive selection", candidates_dir)
        return

    # Load all candidate scores
    candidates = []
    for scores_path in sorted(candidates_dir.glob("*/scores.json")):
        try:
            data = json.loads(scores_path.read_text())
            if not data.get("valid", True):
                continue
            candidates.append({
                "label": scores_path.parent.name,
                "score": data.get("score", 0.0),
                "cost": data.get("cost", 0),
                "snapshot_dir": scores_path.parent / "snapshot",
            })
        except (json.JSONDecodeError, KeyError) as exc:
            log.warning("Skipping %s: %s", scores_path, exc)

    if not candidates:
        log.warning("No valid candidates found in archive — skipping")
        return

    # Compute Pareto frontier (maximize score, minimize cost)
    frontier = []
    for c in candidates:
        dominated = False
        for other in candidates:
            if other is c:
                continue
            if (other["score"] >= c["score"]
                    and other["cost"] <= c["cost"]
                    and (other["score"] > c["score"]
                         or other["cost"] < c["cost"])):
                dominated = True
                break
        if not dominated:
            frontier.append(c)

    best = max(frontier, key=lambda c: c["score"])
    snapshot_dir = best["snapshot_dir"]

    print(f"\n  Archive selection: {best['label']} "
          f"(score={best['score']:.3f}, cost={best['cost']}) "
          f"from {len(candidates)} candidates ({len(frontier)} on Pareto frontier)")

    if not snapshot_dir.exists():
        log.error("Snapshot directory %s does not exist", snapshot_dir)
        return

    # Restore snapshot files into workspace
    workspace_root = agent.workspace.root
    for item in snapshot_dir.iterdir():
        dest = workspace_root / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    log.info("Restored snapshot from %s to %s", snapshot_dir, workspace_root)

    # Reload agent state from the restored workspace
    agent.reload_from_fs()
    print(f"  Agent reloaded from best candidate's snapshot\n")


def run_final_eval(
    agent: TerminalMHAgent,
    trial: ParallelTrialRunner,
    observer: Observer,
    tasks: list[Task],
) -> list[Observation]:
    """Solve all tasks with the evolved workspace."""
    print(f"\n{'=' * 60}")
    print(f"  PHASE 2: Final Evaluation ({len(tasks)} tasks)")
    print(f"{'=' * 60}\n")

    t0 = time.time()
    observations = trial.run_tasks(tasks)
    elapsed = time.time() - t0

    observer.collect(observations)

    score = sum(o.feedback.score for o in observations) / len(observations) if observations else 0.0
    print(f"Final: {score:.1%} ({sum(1 for o in observations if o.feedback.success)}"
          f"/{len(observations)} passed) in {elapsed:.0f}s")

    metrics = compute_metrics(observations)
    print_metrics("Final Evolved Results", metrics)

    return observations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _append_history(evolution_dir: Path, cycle: int, score: float, mutated: bool) -> None:
    history_file = evolution_dir / "history.jsonl"
    entry = {
        "cycle": cycle,
        "score": score,
        "mutated": mutated,
        "timestamp": datetime.now().isoformat(),
    }
    with open(history_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _write_metrics(evolution_dir: Path, scores: list[float]) -> None:
    metrics_file = evolution_dir / "metrics.json"
    metrics = {
        "cycles_completed": len(scores),
        "latest_score": scores[-1] if scores else 0.0,
        "best_score": max(scores) if scores else 0.0,
        "avg_score": sum(scores) / len(scores) if scores else 0.0,
    }
    metrics_file.write_text(json.dumps(metrics, indent=2))


def ensure_challenges(challenges_dir: str) -> None:
    """Download TB2 challenges if the directory is empty."""
    cdir = Path(challenges_dir)
    if cdir.exists() and any(cdir.iterdir()):
        return
    print("Downloading Terminal-Bench 2.0 challenges...")
    script = Path(__file__).resolve().parent.parent.parent / "agent_evolve" / "benchmarks" / "tb2" / "download_challenges.sh"
    subprocess.run(["bash", str(script), str(cdir)], check=True)


def cleanup_containers() -> None:
    """Kill any leaked tb2 Docker containers."""
    try:
        result = subprocess.run(
            ["docker", "ps", "-q", "--filter", "name=tb2-"],
            capture_output=True, text=True, timeout=10,
        )
        for cid in result.stdout.strip().split():
            if cid:
                subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=10)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="MetaHarness experiment on Terminal-Bench 2.0"
    )
    p.add_argument("--config", type=str, required=True,
                   help="Path to YAML config (e.g. examples/configs/metaharness_tbench2.yaml)")
    p.add_argument("--run-name", type=str, required=True,
                   help="Experiment run name (used for work-dir naming)")
    p.add_argument("--phase", type=str, default="all",
                   choices=["0", "1", "2", "all"],
                   help="Which phase to run (default: all)")
    p.add_argument("--seed-workspace", type=str, default="seed_workspaces/terminal_mh",
                   help="Seed workspace to copy")
    p.add_argument("--work-dir", type=str, default=None,
                   help="Workspace directory (default: evolution_workdir/metaharness_<run-name>)")

    # Overrides
    p.add_argument("--workers", type=int, default=None,
                   help="Override solve_workers from config")
    p.add_argument("--max-cycles", type=int, default=None,
                   help="Override max_cycles from config")
    p.add_argument("--eval-sample-size", type=int, default=None,
                   help="Override eval_sample_size (0=all tasks, >0=subsample)")
    p.add_argument("--task-limit", type=int, default=None,
                   help="Limit total number of tasks loaded (for smoke tests)")
    p.add_argument("--solver-model", type=str, default=None,
                   help="Override solver model (e.g. us.anthropic.claude-haiku-4-5-20251001)")
    p.add_argument("--skip-tasks", type=str, default=None,
                   help="Comma-separated task IDs to skip (e.g. 'build-pov-ray,qemu-startup')")
    args = p.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for n in ("botocore", "urllib3", "httpcore", "httpx",
              "strands.models", "strands.tools", "strands.telemetry"):
        logging.getLogger(n).setLevel(logging.WARNING)

    # Load config
    config = EvolveConfig.from_yaml(args.config)

    # Apply CLI overrides
    workers = args.workers or config.extra.get("solve_workers", 6)
    max_cycles = args.max_cycles or config.max_cycles
    solver_model = args.solver_model or config.extra.get("solver_model", "us.anthropic.claude-opus-4-6-v1")
    solver_region = config.extra.get("solver_region", "us-west-2")
    solver_max_tokens = config.extra.get("solver_max_tokens", 16384)

    if args.eval_sample_size is not None:
        config.extra["eval_sample_size"] = args.eval_sample_size

    # Work directory
    work_dir = Path(args.work_dir) if args.work_dir else Path(f"evolution_workdir/metaharness_{args.run_name}")
    seed_dir = Path(args.seed_workspace)

    if not work_dir.exists() and seed_dir.exists():
        work_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(seed_dir, work_dir)
        log.info("Copied seed workspace %s -> %s", seed_dir, work_dir)

    # Ensure TB2 challenges are downloaded
    from agent_evolve.benchmarks.tb2.terminal2 import DEFAULT_CHALLENGES_DIR
    ensure_challenges(DEFAULT_CHALLENGES_DIR)

    # Initialize components
    agent = TerminalMHAgent(
        workspace_dir=work_dir,
        model_id=solver_model,
        region=solver_region,
        max_tokens=solver_max_tokens,
    )

    benchmark = Terminal2Benchmark(
        holdout_ratio=0.0,
        shuffle=False,
    )

    trial = ParallelTrialRunner(agent, benchmark, max_workers=workers)

    engine = MetaHarnessEngine(config)

    evolution_dir = work_dir / "evolution"
    evolution_dir.mkdir(parents=True, exist_ok=True)
    observer = Observer(evolution_dir)
    versioning = VersionControl(work_dir)
    versioning.init()
    history = EvolutionHistory(observer, versioning)

    # Load tasks (consistent across phases)
    task_limit = args.task_limit or 10000
    all_tasks = benchmark.get_tasks(split="test", limit=task_limit)

    # Skip tasks that provide no signal (always pass or always fail)
    skip_set = set()
    if args.skip_tasks:
        skip_set = {t.strip() for t in args.skip_tasks.split(",")}
        before = len(all_tasks)
        all_tasks = [t for t in all_tasks if t.id not in skip_set]
        log.info("Skipped %d tasks: %s", before - len(all_tasks), skip_set)

    log.info("Loaded %d tasks", len(all_tasks))

    print(f"\n{'=' * 60}")
    print(f"  MetaHarness Experiment: {args.run_name}")
    print(f"  Workspace:    {work_dir}")
    print(f"  Tasks:        {len(all_tasks)}")
    print(f"  Max cycles:   {max_cycles}")
    print(f"  Candidates/cycle: {engine.num_candidates}")
    print(f"  Eval sample:  {config.extra.get('eval_sample_size', 0) or 'all'}")
    print(f"  Solver:       {solver_model}")
    print(f"  Proposer:     {engine.model}")
    print(f"  Workers:      {workers}")
    print(f"  Phase:        {args.phase}")
    print(f"{'=' * 60}")

    run_phases = args.phase
    global_t0 = time.time()
    observations: list[Observation] = []

    try:
        # Phase 0: Baseline
        if run_phases in ("0", "all"):
            observations = run_baseline(
                agent, benchmark, trial, observer, versioning, history, all_tasks,
            )

        # Phase 1: Evolution
        if run_phases in ("1", "all"):
            # Factory for parallel candidate evaluation: creates an
            # independent (agent, trial_runner) pair for a workspace copy.
            def _eval_factory(workspace_path: Path) -> ParallelTrialRunner:
                eval_agent = TerminalMHAgent(
                    workspace_dir=workspace_path,
                    model_id=solver_model,
                    region=solver_region,
                    max_tokens=solver_max_tokens,
                )
                return ParallelTrialRunner(eval_agent, benchmark, max_workers=workers)

            best_score = run_evolution(
                engine, agent, trial, observer, versioning, history,
                observations, max_cycles, config, tasks=all_tasks,
                eval_factory=_eval_factory,
            )

        # Select best candidate from archive before final eval (paper: "return Pareto frontier")
        if run_phases in ("2", "all"):
            _restore_best_candidate(work_dir, agent)

            final_obs = run_final_eval(agent, trial, observer, all_tasks)

            # Compare baseline vs evolved
            baseline_scores = history.get_score_curve()
            if baseline_scores:
                baseline = baseline_scores[0]
                final = sum(o.feedback.score for o in final_obs) / len(final_obs) if final_obs else 0.0
                print(f"\n  Baseline: {baseline:.1%} -> Final: {final:.1%} "
                      f"(delta: {final - baseline:+.1%})")

    except KeyboardInterrupt:
        print("\n\nInterrupted! Cleaning up...")
        cleanup_containers()

    total_elapsed = time.time() - global_t0
    print(f"\nTotal wall time: {total_elapsed / 3600:.1f} hours")

    # Write final summary
    summary_path = evolution_dir / "experiment_summary.json"
    summary = {
        "run_name": args.run_name,
        "config": args.config,
        "solver_model": solver_model,
        "proposer_model": engine.model,
        "max_cycles": max_cycles,
        "num_candidates": engine.num_candidates,
        "eval_sample_size": config.extra.get("eval_sample_size", 0),
        "score_history": history.get_score_curve(),
        "total_wall_time_sec": total_elapsed,
        "timestamp": datetime.now().isoformat(),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
