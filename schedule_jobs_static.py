#!/usr/bin/env python3
"""
Sequential Static Job Scheduler

Runs training jobs (rewards x constraints x seeds) sequentially on a single GPU.
Features:
- Progress tracking with JSON file for resume support
- Timestamped logging to global log file
- Individual job logs for debugging
- Failure handling with continuation to next job

Usage:
  # Start/resume all jobs
  uv run python schedule_jobs_static.py --episodes 200

  # Check what would run (dry run)
  uv run python schedule_jobs_static.py --dry-run

  # Run in background with nohup
  nohup uv run python schedule_jobs_static.py --episodes 200 > scheduler_output.log 2>&1 &
"""

import argparse
import os
import subprocess
from datetime import datetime

from loan_simulator.scheduler import get_all_jobs, get_job_id, run_scheduler

# ---------------------------------------------------------------------------
# Job matrix — edit to select which combinations to run
# ---------------------------------------------------------------------------
JOB_MATRIX = {
    "rewards": [
        "social_welfare",
        "rawlsian_maximin",
        "fairness_lagrangian",
        "utilitarian_profit",
    ],
    "constraints": ["approval_rate", "wealth", "both"],
    "seeds": range(1, 4),
}


def run_job(
    reward: str,
    constraint: str,
    seed: int,
    episodes: int,
    test_episodes: int,
    results_dir: str,
    logs_dir: str,
) -> bool:
    """Run a single static training + testing job. Returns True on success."""
    job_id = get_job_id(reward, constraint, seed)
    log_file = os.path.join(logs_dir, f"{job_id}.log")

    train_cmd = [
        "uv", "run", "python", "-u",
        "main.py",
        "--seed", str(seed),
        "--single",
        "--reward", reward,
        "--constraint", constraint,
        "--episodes", str(episodes),
        "--weights-dir", results_dir,
    ]

    test_cmd = [
        "uv", "run", "python", "-u",
        "test_main.py",
        "--seed", str(seed + 1),
        "--episodes", str(test_episodes),
        "--results-dir", results_dir,
        "--weights", os.path.join(
            results_dir, f"{reward}_{constraint}_seed{seed}.pt"
        ),
    ]

    cwd = os.path.dirname(os.path.abspath(__file__)) or "."

    try:
        with open(log_file, "w") as f:
            f.write(f"Job: {job_id}\n")
            f.write(f"Train command: {' '.join(train_cmd)}\n")
            f.write(f"Test command:  {' '.join(test_cmd)}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            f.flush()

            train_result = subprocess.run(
                train_cmd, stdout=f, stderr=subprocess.STDOUT,
                cwd=cwd, timeout=None,
            )

            test_result = subprocess.run(
                test_cmd, stdout=f, stderr=subprocess.STDOUT,
                cwd=cwd, timeout=None,
            )

            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Finished: {datetime.now().isoformat()}\n")
            f.write(f"Train exit code: {train_result.returncode}\n")
            f.write(f"Test exit code:  {test_result.returncode}\n")

        return train_result.returncode == 0 and test_result.returncode == 0

    except subprocess.TimeoutExpired:
        with open(log_file, "a") as f:
            f.write("\nERROR: Job timed out\n")
        return False
    except Exception as e:
        with open(log_file, "a") as f:
            f.write(f"\nERROR: {e}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Sequential Static Job Scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--episodes", type=int, default=200,
        help="Number of training episodes per job (default: 200)",
    )
    parser.add_argument(
        "--test-episodes", type=int, default=100,
        help="Number of testing episodes per job (default: 100)",
    )
    parser.add_argument(
        "--results-dir", type=str, default="./static_sequential_results",
        help="Directory for training results (default: ./static_sequential_results)",
    )
    parser.add_argument(
        "--logs-dir", type=str, default="./job_logs_static",
        help="Directory for job logs (default: ./job_logs_static)",
    )
    parser.add_argument(
        "--progress-file", type=str, default="./job_progress_static.json",
        help="Progress tracking file (default: ./job_progress_static.json)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print jobs without executing",
    )

    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    all_jobs = get_all_jobs(JOB_MATRIX)

    run_scheduler(
        all_jobs=all_jobs,
        run_job_fn=run_job,
        progress_file=args.progress_file,
        logs_dir=args.logs_dir,
        episodes=args.episodes,
        test_episodes=args.test_episodes,
        results_dir=args.results_dir,
        scheduler_label="Sequential Static Job Scheduler",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
