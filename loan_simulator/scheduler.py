import json
import os
from datetime import datetime
from typing import Callable, Dict, List


def get_job_id(reward: str, constraint: str, seed: int) -> str:
    """Generate a unique job identifier."""
    return f"{reward}_{constraint}_seed{seed}"


def get_all_jobs(job_matrix: dict) -> List[Dict]:
    """Generate list of all jobs in the matrix."""
    jobs = []
    for reward in job_matrix["rewards"]:
        for constraint in job_matrix["constraints"]:
            for seed in job_matrix["seeds"]:
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


def run_scheduler(
    all_jobs: List[Dict],
    run_job_fn: Callable,
    progress_file: str,
    logs_dir: str,
    episodes: int,
    test_episodes: int,
    results_dir: str,
    scheduler_label: str,
    dry_run: bool = False,
) -> None:
    """
    Core scheduling loop shared by all schedulers.

    Args:
        all_jobs: Full list of job dicts (from get_all_jobs).
        run_job_fn: Callable(reward, constraint, seed, episodes, test_episodes,
                             results_dir, logs_dir) -> bool
        progress_file: Path to the JSON progress file.
        logs_dir: Directory for per-job log files.
        episodes: Training episodes per job.
        test_episodes: Testing episodes per job.
        results_dir: Output directory for trained weights / results.
        scheduler_label: Human-readable label for log messages.
        dry_run: If True, print plan and exit without running anything.
    """
    global_log = os.path.join(logs_dir, "job_scheduler.log")
    total_jobs = len(all_jobs)

    progress = load_progress(progress_file)

    # Handle interrupted job
    if progress["in_progress"] is not None:
        interrupted_job = progress["in_progress"]
        log_message(
            global_log, "WARN", f"Found interrupted job: {interrupted_job} - will retry"
        )
        progress["in_progress"] = None
        save_progress(progress_file, progress)

    completed_set = set(progress["completed"])
    failed_set = set(progress["failed"])
    pending_jobs = [j for j in all_jobs if j["id"] not in completed_set]

    if dry_run:
        print(f"\n{'=' * 60}")
        print("DRY RUN - Jobs that would be executed:")
        print(f"{'=' * 60}")
        print(f"Total jobs in matrix: {total_jobs}")
        print(f"Already completed:    {len(progress['completed'])}")
        print(f"Previously failed:    {len(progress['failed'])}")
        print(f"Pending jobs:         {len(pending_jobs)}")
        print(f"\nSettings:")
        print(f"  Episodes:      {episodes}")
        print(f"  Results dir:   {results_dir}")
        print(f"  Logs dir:      {logs_dir}")
        print(f"  Progress file: {progress_file}")
        print(f"\nPending jobs:")
        for i, job in enumerate(pending_jobs, 1):
            status = " (previously failed)" if job["id"] in failed_set else ""
            print(f"  {i:2}. {job['id']}{status}")
        return

    # Start scheduler
    log_message(global_log, "INFO", "=" * 50)
    log_message(global_log, "INFO", f"{scheduler_label} Started")
    log_message(
        global_log,
        "INFO",
        f"Total jobs: {total_jobs}, Pending: {len(pending_jobs)}, "
        f"Completed: {len(completed_set)}, Failed: {len(failed_set)}",
    )
    log_message(global_log, "INFO", f"Episodes per job: {episodes}")
    log_message(global_log, "INFO", "=" * 50)

    if not pending_jobs:
        log_message(global_log, "INFO", "All jobs already completed!")
        return

    scheduler_start = datetime.now()

    for job_num, job in enumerate(pending_jobs, 1):
        job_id = job["id"]
        job_index = all_jobs.index(job) + 1

        progress["in_progress"] = job_id
        if job_id in progress["failed"]:
            progress["failed"].remove(job_id)
        save_progress(progress_file, progress)

        log_message(global_log, "START", f"Job {job_index}/{total_jobs}: {job_id}")

        job_start = datetime.now()

        success = run_job_fn(
            reward=job["reward"],
            constraint=job["constraint"],
            seed=job["seed"],
            episodes=episodes,
            test_episodes=test_episodes,
            results_dir=results_dir,
            logs_dir=logs_dir,
        )

        job_duration = (datetime.now() - job_start).total_seconds()
        duration_str = format_duration(job_duration)

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
                f"See {logs_dir}/{job_id}.log",
            )

        save_progress(progress_file, progress)

        if job_num % 10 == 0:
            completed = len(progress["completed"])
            failed = len(progress["failed"])
            remaining = total_jobs - completed - failed
            log_message(
                global_log,
                "INFO",
                f"Progress: {completed} completed, {failed} failed, {remaining} remaining",
            )

    total_duration = (datetime.now() - scheduler_start).total_seconds()
    log_message(global_log, "INFO", "=" * 50)
    log_message(global_log, "INFO", f"{scheduler_label} Finished")
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
