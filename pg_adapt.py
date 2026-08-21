"""Entry point. See cluster/ for HTCondor submission."""

# --- BLAS thread guard -----------------------------------------------------
# MUST run before numpy/torch/matplotlib are imported: the BLAS backend reads
# these at load time and cannot be reconfigured afterwards.
#
# OpenBLAS defaults to one thread per visible core (32 on the cluster nodes).
# With multiprocessing's "spawn" start method every worker re-imports this
# module and does the same, so N workers try to create 32*N threads. On the
# MPI-IS login node -- capped at ~100 processes/threads per user -- that fails
# immediately with "blas_thread_init: pthread_create failed ... Resource
# temporarily unavailable". On an exec node it silently oversubscribes the
# slot, and CPU limits there are HARD-enforced, so it runs slower rather than
# faster.
#
# One BLAS thread per worker is right for this workload: parallelism comes
# from the (seed, combo) process pool, and the per-cohort tensors are small.
# Set these in the environment beforehand to override.
import os as _os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import multiprocessing as mp
import os
import random
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch

# Cap torch intra-op threads too, honouring the guard set at the top.
torch.set_num_threads(int(_os.environ["OMP_NUM_THREADS"]))
from tqdm import tqdm

from loan_simulator.data_loader import AdultIncomeDataLoader
from loan_simulator.environment import IncomeEnvironment
from loan_simulator.testing.data_loader import TestingAdultIncomeDataLoader
from loan_simulator.testing.environment import TestingIncomeEnvironment
from loan_simulator.transition_learner import TransitionParameterLearner
from loan_simulator.agent import PolicyGradientAgent
# See SAMPLE_SIZE note in pepg_adapt.py -- 20000 rows gives only 6439
# female records, far short of --N-female=12000.
SAMPLE_SIZE = 100000

AGENT_TAG = "pg"

from run_multi_seed import add_derived_columns, aggregate_across_seeds
from pg_run import (
    VALID_COMBOS,
    _combo_key,
    print_summary_table,
    plot_comparison_agg,
    plot_wealth_agg,
    plot_social_welfare_agg,
    plot_inequality_agg,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_lambdas(reward, constraint):
    lw = 0.5 if constraint == "two_sided" else (
        0.0 if reward == "utilitarian_profit" else
        2.0 if reward == "social_welfare" else
        5.0 if reward == "rawlsian_maximin" else
        10.0  # fairness_lagrangian
    )
    la = (
        0.0 if reward == "utilitarian_profit" else
        2.0 if reward == "social_welfare" else
        5.0 if reward == "rawlsian_maximin" else
        10.0
    )
    return lw, la


# ---------------------------------------------------------------------------
# Phase 1 worker — static IncomeEnvironment
# ---------------------------------------------------------------------------

def _train_worker(cfg):
    seed       = cfg["seed"]
    reward     = cfg["reward_function"]
    constraint = cfg["constraint_type"]
    run_id     = cfg.get("run_id", 0)
    total      = cfg.get("total_runs", 1)

    try:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        # Skip if weights already exist (allows partial restart / a fast
        # aggregate-only pass) -- mirrors pepg_adapt.py's _train_worker.
        weights_path = cfg["weights_path"]
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        if os.path.exists(weights_path):
            print(f"  [{run_id:3d}/{total}] TRAIN SKIP (weights exist)  seed={seed}  {reward}/{constraint}")
            return {"success": True, "seed": seed, "reward": reward,
                    "constraint": constraint, "weights_path": weights_path}

        loader = AdultIncomeDataLoader(
            filepath=cfg["data_filepath"], sample_size=SAMPLE_SIZE
        )
        loader.load_data()
        loader.preprocess()

        theta = TransitionParameterLearner(
            default_rate_min=0.02, default_rate_max=0.15
        )
        theta.fit(loader.data)

        env = IncomeEnvironment(
            theta_params=theta,
            initial_wealth_male=loader.male_data["X"].values,
            initial_wealth_female=loader.female_data["X"].values,
            N_male=cfg["N_male"],
            N_female=cfg["N_female"],
            T=cfg["T"],
            dt=cfg["dt"],
            seed=seed,
        )

        lw, la = _default_lambdas(reward, constraint)
        agent = PolicyGradientAgent(
            env,
            hidden_dim=cfg.get("hidden_dim", 128),
            lr=cfg.get("lr", 1e-3),
            reward_function=reward,
            constraint_type=constraint,
            lambda_wealth=cfg.get("lambda_wealth", lw),
            lambda_approval=cfg.get("lambda_approval", la),
            lambda_lr=cfg.get("lambda_lr", 1e-3),
            alpha_lr=cfg.get("alpha_lr", None),
            entropy_coef=cfg.get("entropy_coef", 0.01),
        )

        for _ in range(cfg.get("warmup_episodes", 0)):
            obs, _ = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                obs, _, term, trunc, _ = env.step(action)
                done = term or trunc

        agent.train(num_episodes=cfg["train_episodes"])

        agent.save_model(weights_path)

        print(
            f"  [{run_id:3d}/{total}] TRAIN OK  "
            f"seed={seed}  {reward}/{constraint}  "
            f"eps={cfg['train_episodes']}"
        )
        return {"success": True, "seed": seed, "reward": reward,
                "constraint": constraint, "weights_path": weights_path}

    except Exception as exc:
        import traceback
        print(f"  [{run_id}/{total}] TRAIN FAIL  seed={seed}  {reward}/{constraint}: {exc}")
        traceback.print_exc()
        return {"success": False, "seed": seed, "reward": reward, "constraint": constraint}


# ---------------------------------------------------------------------------
# Phase 2 worker — load weights, continue training on TestingIncomeEnvironment
# ---------------------------------------------------------------------------

def _deploy_worker(cfg):
    seed         = cfg["seed"]
    reward       = cfg["reward_function"]
    constraint   = cfg["constraint_type"]
    weights_path = cfg["weights_path"]
    run_id       = cfg.get("run_id", 0)
    total        = cfg.get("total_runs", 1)

    try:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        env = TestingIncomeEnvironment(
            theta_params=cfg["theta"],
            initial_wealth_male=cfg["male_X"],
            initial_wealth_female=cfg["female_X"],
            ground_truth_male=cfg["gt_male"],
            ground_truth_female=cfg["gt_female"],
            N_male=cfg["N_male"],
            N_female=cfg["N_female"],
            T=cfg["T"],
            dt=cfg["dt"],
            seed=seed,
        )

        lw, la = _default_lambdas(reward, constraint)
        agent = PolicyGradientAgent(
            env,
            hidden_dim=cfg.get("hidden_dim", 128),
            lr=cfg.get("lr", 1e-3),
            reward_function=reward,
            constraint_type=constraint,
            lambda_wealth=lw,
            lambda_approval=la,
            lambda_lr=cfg.get("lambda_lr", 1e-3),
            alpha_lr=cfg.get("alpha_lr", None),
            entropy_coef=cfg.get("entropy_coef", 0.01),
        )

        # Load pre-trained weights from static training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        saved = torch.load(weights_path, map_location=device, weights_only=False)
        agent.policy_net.load_state_dict(saved["policy_net_state_dict"])
        if "lambda_state_dict" in saved and agent.learnable_lambdas is not None:
            agent.learnable_lambdas.load_state_dict(saved["lambda_state_dict"])

        # Continue training (fine-tune) on the performative environment
        deploy_eps = cfg["deploy_episodes"]
        for ep in range(deploy_eps):
            agent.train_episode()
            if (ep + 1) % max(1, deploy_eps // 5) == 0 or ep + 1 == deploy_eps:
                app_M = env.episode_loans_M / max(env.episode_applications_M, 1)
                app_F = env.episode_loans_F / max(env.episode_applications_F, 1)
                print(
                    f"  [{run_id:3d}/{total}] DEPLOY  "
                    f"seed={seed}  {reward[:4]}/{constraint[:4]}  "
                    f"ep {ep+1}/{deploy_eps}  "
                    f"μ_M={env.mu_M:.1f}  μ_F={env.mu_F:.1f}  "
                    f"appM={app_M:.3f} appF={app_F:.3f}"
                )

        env.finalize_episode_metrics()
        df = env.get_episode_metrics_dataframe()
        key = _combo_key(reward, constraint)

        # Persist the POST-DEPLOY artefacts. Previously only the train phase
        # saved weights, so the policy that actually produced every reported
        # result -- after `deploy_episodes` further updates -- was discarded.
        deploy_dir = cfg.get("deploy_artifacts_dir")
        if deploy_dir:
            os.makedirs(deploy_dir, exist_ok=True)
            stem = f"{AGENT_TAG}_{reward}__{constraint}__seed{seed}"
            agent.save_model(os.path.join(deploy_dir, f"{stem}_deployed.pt"))
            df.to_csv(os.path.join(deploy_dir, f"{stem}_episodes.csv"), index=False)
            pd.DataFrame({
                "episode": range(1, len(agent.episode_rewards) + 1),
                "episode_reward": agent.episode_rewards,
                "lambda_wealth": agent.lambda_history["wealth"],
                "lambda_approval": agent.lambda_history["approval"],
            }).to_csv(os.path.join(deploy_dir, f"{stem}_training_trace.csv"), index=False)
            np.savez_compressed(
                os.path.join(deploy_dir, f"{stem}_population.npz"),
                X_male=env.current_X_male, X_female=env.current_X_female,
                loan_counts_M=env.loan_counts_M, loan_counts_F=env.loan_counts_F,
            )

        print(f"  [{run_id:3d}/{total}] DEPLOY OK  seed={seed}  {reward}/{constraint}")
        return seed, key, df

    except Exception as exc:
        import traceback
        print(f"  [{run_id}/{total}] DEPLOY FAIL  seed={seed}  {reward}/{constraint}: {exc}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Static pre-train PG → continued training on performative env"
    )

    # Seeds / combos
    parser.add_argument("--seeds",      type=int, default=5)  # paper: 5 seeds
    parser.add_argument("--seed-list",  type=int, nargs="+", default=None)
    parser.add_argument("--reward",     type=str, default="all",
                        choices=["all", "utilitarian_profit", "social_welfare",
                                 "rawlsian_maximin", "fairness_lagrangian"])
    parser.add_argument("--constraint", type=str, default="all",
                        choices=["all", "social", "dm", "two_sided"])

    # Training (static env)
    parser.add_argument("--train-episodes",  type=int,   default=500)
    parser.add_argument("--warmup",           type=int,   default=0)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--lambda-lr",        type=float, default=1e-3)
    parser.add_argument("--alpha-lr",         type=float, default=None,
                        help="LR for the two_sided alpha blend weight. "
                             "Defaults to lambda_lr/4 -- alpha is bounded in (0,1) "
                             "and takes a normalised signal, so the rate that suits "
                             "the unbounded lambdas saturates it.")
    parser.add_argument("--entropy-coef",     type=float, default=0.01)
    parser.add_argument("--lambda-wealth",    type=float, default=None,
                        help="Override default lambda_wealth (default: per reward function)")
    parser.add_argument("--lambda-approval",  type=float, default=None)

    # Deployment (performative env)
    parser.add_argument("--deploy-episodes",  type=int,   default=1000)  # paper: 1000-episode axis
    parser.add_argument("--credit-threshold", type=float, default=0.5)

    # Architecture
    parser.add_argument("--hidden-dim", type=int, default=128)

    # Environment
    parser.add_argument("--N-male",   type=int,   default=12000)
    parser.add_argument("--N-female", type=int,   default=12000)
    parser.add_argument("--T",        type=int,   default=100)
    parser.add_argument("--dt",       type=float, default=0.5)

    # I/O
    parser.add_argument("--data",              type=str, default=None)
    parser.add_argument("--weights-dir",       type=str, default="./weights_adapt")
    parser.add_argument("--results-dir",       type=str, default="./results_adapt")
    parser.add_argument("--checkpoint-dir",    type=str, default=None)
    parser.add_argument("--workers",           type=int, default=None)
    parser.add_argument("--no-plots",          action="store_true")
    parser.add_argument("--save-per-seed-csv", action="store_true")
    parser.add_argument("--skip-train",        action="store_true",
                        help="Skip Phase 1 — use existing weights in --weights-dir")

    args = parser.parse_args()

    seeds   = args.seed_list if args.seed_list is not None else list(range(args.seeds))
    n_seeds = len(seeds)

    combos = VALID_COMBOS
    if args.reward != "all":
        combos = [(r, c) for r, c in combos if r == args.reward]
    if args.constraint != "all":
        combos = [(r, c) for r, c in combos if c == args.constraint]

    if not combos:
        print("No valid combos match filter. Exiting.")
        return

    if args.workers is None:
        args.workers = min(int(mp.cpu_count() * 0.8), 20)

    os.makedirs(args.weights_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    checkpoint_dir = args.checkpoint_dir or os.path.join(args.results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("PG STATIC PRE-TRAIN → PERFORMATIVE DEPLOY")
    print("=" * 70)
    print(f"  Seeds           : {seeds}")
    print(f"  Combos          : {len(combos)}")
    print(f"  Train episodes  : {args.train_episodes}  (static env)")
    print(f"  Deploy episodes : {args.deploy_episodes} (performative env, keeps updating)")
    print(f"  Workers         : {args.workers}")
    print(f"  Population      : {args.N_male}M + {args.N_female}F")
    print(f"  Weights dir     : {args.weights_dir}")
    print(f"  Results dir     : {args.results_dir}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Phase 1: Train on static IncomeEnvironment
    # ------------------------------------------------------------------
    if not args.skip_train:
        print(f"\n[1/3] Training {len(seeds) * len(combos)} agents on static env…")
        train_configs = []
        for seed in seeds:
            for reward, constraint in combos:
                lw, la = _default_lambdas(reward, constraint)
                train_configs.append({
                    "seed":            seed,
                    "reward_function": reward,
                    "constraint_type": constraint,
                    "weights_path":    os.path.join(
                        args.weights_dir, f"{reward}__{constraint}__seed{seed}.pt"
                    ),
                    "train_episodes":  args.train_episodes,
                    "warmup_episodes": args.warmup,
                    "N_male":          args.N_male,
                    "N_female":        args.N_female,
                    "T":               args.T,
                    "dt":              args.dt,
                    "hidden_dim":      args.hidden_dim,
                    "lr":              args.lr,
                    "lambda_wealth":   args.lambda_wealth if args.lambda_wealth is not None else lw,
                    "lambda_approval": args.lambda_approval if args.lambda_approval is not None else la,
                    "lambda_lr":       args.lambda_lr,
                    "alpha_lr":        args.alpha_lr,
                    "entropy_coef":    args.entropy_coef,
                    "data_filepath":   args.data,
                    "run_id":          len(train_configs) + 1,
                    "total_runs":      len(seeds) * len(combos),
                })

        with mp.Pool(processes=args.workers) as pool:
            train_results = list(
                tqdm(
                    pool.imap(_train_worker, train_configs),
                    total=len(train_configs),
                    desc="Phase 1: training",
                    unit="run",
                )
            )

        n_ok = sum(1 for r in train_results if r["success"])
        print(f"\n  Training done: {n_ok}/{len(train_configs)} successful")
    else:
        print(f"\n[1/3] --skip-train: using existing weights in {args.weights_dir}")
        train_results = []
        for seed in seeds:
            for reward, constraint in combos:
                wp = os.path.join(args.weights_dir, f"{reward}__{constraint}__seed{seed}.pt")
                exists = os.path.exists(wp)
                if not exists:
                    print(f"  MISSING: {wp}")
                train_results.append({
                    "success":      exists,
                    "seed":         seed,
                    "reward":       reward,
                    "constraint":   constraint,
                    "weights_path": wp,
                })
        n_ok = sum(1 for r in train_results if r["success"])
        print(f"  Found {n_ok}/{len(train_results)} weight files")

    # ------------------------------------------------------------------
    # Phase 2: Deploy — continued training on TestingIncomeEnvironment
    # ------------------------------------------------------------------
    print(f"\n[2/3] Deploying {n_ok} agents on performative env…")

    print("  Loading test data (once)…")
    test_loader = TestingAdultIncomeDataLoader(
        filepath=args.data,
        sample_size=SAMPLE_SIZE,
        credit_threshold=args.credit_threshold,
    )
    test_loader.load_data()
    test_loader.preprocess()
    test_theta = TransitionParameterLearner(
        default_rate_min=0.05, default_rate_max=0.25
    )
    test_theta.fit(test_loader.data)
    _male_X    = test_loader.male_data["X"].values
    _female_X  = test_loader.female_data["X"].values
    _gt_male   = test_loader.male_data["ground_truth_approval"].values
    _gt_female = test_loader.female_data["ground_truth_approval"].values

    # Load existing deploy checkpoints
    seed_to_results = defaultdict(dict)
    n_loaded = 0
    for tr in train_results:
        if not tr["success"]:
            continue
        seed       = tr["seed"]
        key        = _combo_key(tr["reward"], tr["constraint"])
        ckpt_path  = os.path.join(checkpoint_dir, f"seed{seed}_{key}.csv")
        if os.path.exists(ckpt_path):
            try:
                df = add_derived_columns(pd.read_csv(ckpt_path))
                seed_to_results[seed][key] = df
                n_loaded += 1
            except Exception:
                pass  # corrupt — re-run

    if n_loaded:
        print(f"\n  Loaded {n_loaded} checkpoint(s) — skipping those combos.")

    # Build deploy configs
    deploy_configs = []
    for tr in train_results:
        if not tr["success"]:
            continue
        seed       = tr["seed"]
        reward     = tr["reward"]
        constraint = tr["constraint"]
        key        = _combo_key(reward, constraint)
        if key in seed_to_results.get(seed, {}):
            continue
        deploy_configs.append({
            "seed":             seed,
            "reward_function":  reward,
            "constraint_type":  constraint,
            "weights_path":     tr["weights_path"],
            "deploy_episodes":  args.deploy_episodes,
            "N_male":           args.N_male,
            "N_female":         args.N_female,
            "T":                args.T,
            "dt":               args.dt,
            "hidden_dim":       args.hidden_dim,
            "lr":               args.lr,
            "lambda_lr":        args.lambda_lr,
            "alpha_lr":         args.alpha_lr,
            "entropy_coef":     args.entropy_coef,
            "deploy_artifacts_dir": os.path.join(args.results_dir, "deploy_artifacts"),
            "theta":            test_theta,
            "male_X":           _male_X,
            "female_X":         _female_X,
            "gt_male":          _gt_male,
            "gt_female":        _gt_female,
            "run_id":           len(deploy_configs) + 1,
            "total_runs":       n_ok - n_loaded,
        })

    if not deploy_configs:
        print("\n  All combos already checkpointed — skipping to aggregation.")
    else:
        print(f"\n  Running {len(deploy_configs)} deploy workers ({n_loaded} already done)…")
        with mp.Pool(processes=args.workers) as pool:
            deploy_iter = tqdm(
                pool.imap_unordered(_deploy_worker, deploy_configs),
                total=len(deploy_configs),
                desc="Phase 2: deploying",
                unit="run",
            )
            for raw in deploy_iter:
                if raw is None:
                    continue
                seed, key, df = raw
                df = add_derived_columns(df)
                seed_to_results[seed][key] = df

                ckpt_path = os.path.join(checkpoint_dir, f"seed{seed}_{key}.csv")
                df.to_csv(ckpt_path, index=False)
                print(f"  Checkpointed: seed{seed}_{key}")

                if args.save_per_seed_csv:
                    df.to_csv(
                        os.path.join(args.results_dir, f"seed{seed}_{key}_{timestamp}.csv"),
                        index=False,
                    )

    # ------------------------------------------------------------------
    # Phase 3: Aggregate + Plot
    # ------------------------------------------------------------------
    expected_keys = {_combo_key(r, c) for r, c in combos}
    seed_results = [
        seed_to_results[s]
        for s in seeds
        if expected_keys.issubset(seed_to_results[s].keys())
    ]

    if not seed_results:
        print("No complete seed results to aggregate. Exiting.")
        return

    n_complete = len(seed_results)
    print(f"\n[3/3] Aggregating mean ± std across {n_complete} seeds…")
    aggregated = aggregate_across_seeds(seed_results)

    for key, (mdf, sdf) in aggregated.items():
        mdf.to_csv(
            os.path.join(args.results_dir, f"mean_{key}_{timestamp}.csv"), index=False
        )
        sdf.to_csv(
            os.path.join(args.results_dir, f"std_{key}_{timestamp}.csv"), index=False
        )
    print(f"  Aggregated CSVs saved to {args.results_dir}/")

    print_summary_table(aggregated, n_complete)

    if not args.no_plots:
        print("  Generating plots…")
        for ct in ["social", "dm", "two_sided"]:
            plot_comparison_agg(aggregated, args.results_dir, timestamp, n_complete, ct)
            plot_wealth_agg(aggregated, args.results_dir, timestamp, n_complete, ct)
            plot_social_welfare_agg(aggregated, args.results_dir, timestamp, n_complete, ct)
            plot_inequality_agg(aggregated, args.results_dir, timestamp, n_complete, ct)

    print("\nDone.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
