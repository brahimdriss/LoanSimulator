#!/usr/bin/env python3
"""
Sequential PePG Job Scheduler

Runs 60 training jobs (4 rewards x 3 constraints x 5 seeds) sequentially on a single GPU.
Features:
- Progress tracking with JSON file for resume support
- Timestamped logging to global log file
- Individual job logs for debugging
- Failure handling with continuation to next job
"""

import argparse
import json
import os
import subprocess
from datetime import datetime
from typing import Dict, List

# Job matrix configuration
# JOB_MATRIX = {
#     'rewards': ['social_welfare', 'rawlsian_maximin', 'fairness_lagrangian', 'utilitarian_profit'],
#     'constraints': ['approval_rate', 'wealth', 'both'],
#     'seeds': range(1, 6)
# }
# JOB_MATRIX = {
#     "rewards": [
#         "social_welfare",
#         "rawlsian_maximin",
#         "fairness_lagrangian",
#         "utilitarian_profit",
#     ],
#     "constraints": ["wealth"],
#     "seeds": range(1, 6),
# }

JOB_MATRIX = {
    "rewards": [
        "rawlsian_maximin",
    ],
    "constraints": ["wealth"],
    "seeds": range(1, 2),
}

# "constraints": ["approval_rate", "wealth", "both"],

def get_job_id(reward: str, constraint: str, seed: int) -> str:
    """Generate a unique job identifier."""
    return f"{reward}_{constraint}_seed{seed}"


def get_all_jobs() -> List[Dict[str, any]]:
    """Generate list of all jobs in the matrix."""
    jobs = []
    for reward in JOB_MATRIX["rewards"]:
        for constraint in JOB_MATRIX["constraints"]:
            for seed in JOB_MATRIX["seeds"]:
                jobs.append(
                    {
                        "reward": reward,
                        "constraint": constraint,
                        "seed": seed,
                        "id": get_job_id(reward, constraint, seed),
                    }
                )
    return jobs


def load_progress(progress_file: str) -> Dict:
    """Load progress from JSON file."""
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load progress file: {e}")
    return {"completed": [], "failed": [], "in_progress": None}


def save_progress(progress_file: str, progress: Dict) -> None:
    """Save progress to JSON file."""
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


def log_message(log_file: str, level: str, message: str) -> None:
    """Write a timestamped message to the log file and stdout."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"{timestamp} | {level:5} | {message}"
    print(log_line)

    with open(log_file, "a") as f:
        f.write(log_line + "\n")


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes >= 60:
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s"


def run_job(
    reward: str,
    constraint: str,
    seed: int,
    episodes: int,
    test_episodes: int,
    results_dir: str,
    logs_dir: str,
) -> bool:
    """
    Run a single training job.

    Returns True if successful, False otherwise.
    """
    job_id = get_job_id(reward, constraint, seed)
    log_file = os.path.join(logs_dir, f"{job_id}.log")
    
    # Build command using uv
    cmd = [
        "uv",
        "run",
        "python",
        "-u",
        "par_static.py",
        "--seeds",
        "1",
        "--seed",
        str(seed),
        "--reward",
        reward,
        "--constraint",
        constraint,
        "--episodes",
        str(episodes),
        "--workers",
        "1",
        "--weights-dir",
        results_dir,
    ]

    # Save model

    job_id = get_job_id(reward, constraint, seed)

    weights_filename = f"pepg_{reward}_{constraint}_seed{seed}.pt"
    weights_path = os.path.join(results_dir, weights_filename)

    test_cmd = [
        "uv",
        "run",
        "python",
        "-u",
        "par_static_testing.py",
        "--seeds",
        "1",
        "--seed",
        str(seed),
        "--reward",
        reward,
        "--constraint",
        constraint,
        "--episodes",
        str(test_episodes),
        "--workers",
        "1",
        "--weights-dir",
        results_dir+"_test",
        "--load",
        weights_path,
    ]

    try:
        with open(log_file, "w") as f:
            f.write(f"Job: {job_id}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            f.flush()

            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
                timeout=None,  # No timeout - let jobs run to completion
            )

            test = subprocess.run(
                test_cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
                timeout=None,  # No timeout - let jobs run to completion
            )

            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Finished: {datetime.now().isoformat()}\n")
            f.write(f"Exit code: {result.returncode}\n")
            f.write(f"Test Exit code: {test.returncode}\n")

        return result.returncode == 0 and test.returncode == 0

    except subprocess.TimeoutExpired:
        with open(log_file, "a") as f:
            f.write(f"\nERROR: Job timed out\n")
        return False
    except Exception as e:
        with open(log_file, "a") as f:
            f.write(f"\nERROR: {str(e)}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Sequential PePG Job Scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start/resume all jobs
  uv run python schedule_jobs.py --episodes 200

  # Check what would run (dry run)
  uv run python schedule_jobs.py --dry-run

  # Run in background with nohup
  nohup uv run python schedule_jobs.py --episodes 200 > scheduler_output.log 2>&1 &
        """,
    )

    # parser.add_argument(
    #     "--episodes",
    #     type=int,
    #     default=300,
    #     help="Number of training episodes per job (default: 200)",
    # )
    # parser.add_argument(
    #     "--test-episodes",
    #     type=int,
    #     default=100,
    #     help="Number of testing episodes per job (default: 100)",
    # )



    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of training episodes per job (default: 200)",
    )
    parser.add_argument(
        "--test-episodes",
        type=int,
        default=10,
        help="Number of testing episodes per job (default: 100)",
    )

    parser.add_argument(
        "--machine",
        type=str,
        required=True,
        help="Name of the machine running this scheduler (used to separate results, logs, and progress)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory for training results (default: ./pg_sequential_results_<machine>)",
    )

    parser.add_argument(
        "--logs-dir",
        type=str,
        default=None,
        help="Directory for job logs (default: ./job_logs_pg_<machine>)",
    )
    parser.add_argument(
        "--progress-file",
        type=str,
        default=None,
        help="Progress tracking file (default: ./job_progress_pg_<machine>.json)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print jobs without executing"
    )

    args = parser.parse_args()

    # Apply machine-name defaults
    if args.results_dir is None:
        args.results_dir = f"./pg_sequential_results_{args.machine}"
    if args.logs_dir is None:
        args.logs_dir = f"./job_logs_pg_{args.machine}"
    if args.progress_file is None:
        args.progress_file = f"./job_progress_pg_{args.machine}.json"

    # Create directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.results_dir+"_test", exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    # Global log file
    global_log = os.path.join(args.logs_dir, "job_scheduler.log")

    # Get all jobs
    all_jobs = get_all_jobs()
    total_jobs = len(all_jobs)

    # Load progress
    progress = load_progress(args.progress_file)

    # Handle interrupted job (was in_progress when scheduler was killed)
    if progress["in_progress"] is not None:
        interrupted_job = progress["in_progress"]
        log_message(
            global_log, "WARN", f"Found interrupted job: {interrupted_job} - will retry"
        )
        progress["in_progress"] = None
        save_progress(args.progress_file, progress)

    # Filter pending jobs
    completed_set = set(progress["completed"])
    failed_set = set(progress["failed"])
    pending_jobs = [j for j in all_jobs if j["id"] not in completed_set]

    # Dry run mode
    if args.dry_run:
        print(f"\n{'=' * 60}")
        print("DRY RUN - Jobs that would be executed:")
        print(f"{'=' * 60}")
        print(f"Total jobs in matrix: {total_jobs}")
        print(f"Already completed: {len(progress['completed'])}")
        print(f"Previously failed: {len(progress['failed'])}")
        print(f"Pending jobs: {len(pending_jobs)}")
        print(f"\nSettings:")
        print(f"  Episodes: {args.episodes}")
        print(f"  Results dir: {args.results_dir}")
        print(f"  Logs dir: {args.logs_dir}")
        print(f"  Progress file: {args.progress_file}")
        print(f"\nPending jobs:")
        for i, job in enumerate(pending_jobs, 1):
            status = " (previously failed)" if job["id"] in failed_set else ""
            print(f"  {i:2}. {job['id']}{status}")
        return

    # Start scheduler
    log_message(global_log, "INFO", "=" * 50)
    log_message(global_log, "INFO", "Sequential PG Job Scheduler Started")
    log_message(
        global_log,
        "INFO",
        f"Total jobs: {total_jobs}, Pending: {len(pending_jobs)}, "
        f"Completed: {len(completed_set)}, Failed: {len(failed_set)}",
    )
    log_message(global_log, "INFO", f"Episodes per job: {args.episodes}")
    log_message(global_log, "INFO", "=" * 50)

    if not pending_jobs:
        log_message(global_log, "INFO", "All jobs already completed!")
        return

    # Process jobs
    scheduler_start = datetime.now()

    for job_num, job in enumerate(pending_jobs, 1):
        job_id = job["id"]
        job_index = all_jobs.index(job) + 1  # 1-indexed position in full matrix

        # Mark as in progress
        progress["in_progress"] = job_id
        # Remove from failed if retrying
        if job_id in progress["failed"]:
            progress["failed"].remove(job_id)
        save_progress(args.progress_file, progress)

        log_message(global_log, "START", f"Job {job_index}/{total_jobs}: {job_id}")

        job_start = datetime.now()

        success = run_job(
            reward=job["reward"],
            constraint=job["constraint"],
            seed=job["seed"],
            episodes=args.episodes,
            test_episodes=args.test_episodes,
            results_dir=args.results_dir,
            logs_dir=args.logs_dir,
        )

        job_duration = (datetime.now() - job_start).total_seconds()
        duration_str = format_duration(job_duration)

        # Update progress
        progress["in_progress"] = None
        if success:
            progress["completed"].append(job_id)
            log_message(
                global_log,
                "DONE",
                f"Job {job_index}/{total_jobs}: {job_id} ({duration_str})",
            )
        else:
            progress["failed"].append(job_id)
            log_message(
                global_log,
                "FAIL",
                f"Job {job_index}/{total_jobs}: {job_id} ({duration_str}) - "
                f"See {args.logs_dir}/{job_id}.log",
            )

        save_progress(args.progress_file, progress)

        # Progress summary every 10 jobs
        if job_num % 10 == 0:
            completed = len(progress["completed"])
            failed = len(progress["failed"])
            remaining = total_jobs - completed - failed
            log_message(
                global_log,
                "INFO",
                f"Progress: {completed} completed, {failed} failed, {remaining} remaining",
            )

    # Final summary
    total_duration = (datetime.now() - scheduler_start).total_seconds()
    log_message(global_log, "INFO", "=" * 50)
    log_message(global_log, "INFO", "Scheduler Finished")
    log_message(global_log, "INFO", f"Total time: {format_duration(total_duration)}")
    log_message(
        global_log, "INFO", f"Completed: {len(progress['completed'])}/{total_jobs}"
    )
    log_message(global_log, "INFO", f"Failed: {len(progress['failed'])}")

    if progress["failed"]:
        log_message(global_log, "INFO", "Failed jobs:")
        for failed_job in progress["failed"]:
            log_message(global_log, "INFO", f"  - {failed_job}")

    log_message(global_log, "INFO", "=" * 50)


if __name__ == "__main__":
    main()
