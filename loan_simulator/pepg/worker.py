import os

import numpy as np
import torch


def pepg_train_worker(config: dict) -> dict:
    """Train a single PePG V2 configuration with specified seed, reward function, and constraints."""
    try:
        # Set GPU for this worker
        if config["gpu_id"] >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu_id"])
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        import random

        # Set seeds
        np.random.seed(config["seed"])
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config["seed"])
            torch.cuda.manual_seed_all(config["seed"])
            torch.backends.cudnn.deterministic = True

        # Import here to avoid CUDA initialization conflicts in multiprocessing
        from loan_simulator.data_loader import AdultIncomeDataLoader
        from loan_simulator.environment import IncomeEnvironment
        from loan_simulator.io_utils import save_episode_metrics, save_lambda_history
        from loan_simulator.transition_learner import TransitionParameterLearner

        from .agent import PePGAgentV2

        # Create individual run directory
        run_name = (
            f"pepg_v2_{config['reward_function']}_{config['constraint_type']}_seed{config['seed']}"
        )
        run_dir = os.path.join(config["base_output_dir"], "runs", run_name)
        os.makedirs(run_dir, exist_ok=True)

        run_weights_dir = os.path.join(run_dir, "weights")
        run_lambda_dir = os.path.join(run_dir, "lambda_trajectories")
        run_metrics_dir = os.path.join(run_dir, "episode_metrics")
        os.makedirs(run_weights_dir, exist_ok=True)
        os.makedirs(run_lambda_dir, exist_ok=True)
        os.makedirs(run_metrics_dir, exist_ok=True)

        print(
            f"\n[TRAIN {config.get('run_id', '?')}/{config.get('total_runs', '?')}] "
            f"Seed={config['seed']}, Reward={config['reward_function']}, "
            f"Constraint={config['constraint_type']}, GPU={config['gpu_id']}"
        )

        # Load data
        loader = AdultIncomeDataLoader(
            filepath=config["data_filepath"], sample_size=20000
        )
        loader.load_data()
        loader.preprocess()

        # Learn theta parameters
        theta_learner = TransitionParameterLearner(
            default_rate_min=0.02, default_rate_max=0.15
        )
        theta_learner.fit(loader.data)

        # Create environment
        env = IncomeEnvironment(
            theta_params=theta_learner,
            initial_wealth_male=loader.male_data["X"].values,
            initial_wealth_female=loader.female_data["X"].values,
            N_male=3000,
            N_female=3000,
            T=100,
            dt=0.5,
            seed=config["seed"],
        )

        # Create PePG V2 agent
        agent = PePGAgentV2(
            env,
            reward_function=config["reward_function"],
            constraint_type=config["constraint_type"],
            lambda_wealth=config.get("lambda_wealth", 2.0),
            lambda_approval=config.get("lambda_approval", 2.0),
            lambda_lr=config.get("lambda_lr", 1e-2),
            buffer_capacity=config.get("buffer_capacity", 50),
            warmup_episodes=config.get("warmup_episodes", 0),
            alpha_R=env.alpha_R,
            alpha_B=env.alpha_B,
            beta_R=env.beta_R,
            beta_B=env.beta_B,
            hawkes_weight=config.get("hawkes_weight", 1.0),
            wealth_weight=config.get("wealth_weight", 1.0),
            transition_weight=config.get("transition_weight", 1.0),
            reward_weight=config.get("reward_weight", 1.0),
        )

        # Train
        agent.train(num_episodes=config["num_episodes"], use_performative=True)

        # Construct weights path
        weights_filename = (
            f"pepg_{config['reward_function']}_{config['constraint_type']}_seed{config['seed']}.pt"
        )
        weights_path = os.path.join(run_weights_dir, weights_filename)

        # Save model
        agent.save_model(weights_path)

        # Save lambda trajectories
        if config["reward_function"] != "utilitarian_profit":
            save_lambda_history(
                agent.lambda_history,
                f"pepg_v2_{config['reward_function']}",
                config["constraint_type"],
                config["seed"],
                save_dir=run_lambda_dir,
                format="both",
            )

        # Save episode metrics
        save_episode_metrics(
            agent.episode_metrics,
            f"pepg_v2_{config['reward_function']}",
            config["constraint_type"],
            config["seed"],
            save_dir=run_metrics_dir,
            format="both",
        )

        # Extract final metrics
        final_metrics = {
            "seed": config["seed"],
            "reward_function": config["reward_function"],
            "constraint_type": config["constraint_type"],
            "weights_path": weights_path,
            "final_episode_reward": (
                agent.episode_rewards[-1] if agent.episode_rewards else None
            ),
            "num_episodes": len(agent.episode_rewards),
            "final_wealth_gap": (
                env.history["wealth_gap"][-1] if env.history["wealth_gap"] else None
            ),
            "final_approval_disparity": (
                env.history["approval_disparity"][-1]
                if env.history["approval_disparity"]
                else None
            ),
            "total_profit": (
                sum(env.history["profit"]) if env.history["profit"] else 0
            ),
        }

        # Add episode metrics if available
        if agent.episode_metrics.get("episode"):
            final_metrics.update(
                {
                    "train_rho_mean": float(
                        np.mean(agent.episode_metrics["rho_episode"])
                    ),
                    "train_rho_std": float(
                        np.std(agent.episode_metrics["rho_episode"])
                    ),
                    "train_rho_final": float(
                        agent.episode_metrics["rho_episode"][-1]
                    ),
                    "train_mu_M_final": float(agent.episode_metrics["mu_M_end"][-1]),
                    "train_mu_F_final": float(agent.episode_metrics["mu_F_end"][-1]),
                }
            )

        print(
            f"[TRAIN DONE {config.get('run_id', '?')}/{config.get('total_runs', '?')}] "
            f"Seed={config['seed']}, Reward={config['reward_function']}"
        )

        return {"success": True, "config": config, "metrics": final_metrics}

    except Exception as e:
        print(
            f"[TRAIN FAILED] Seed={config['seed']}, Reward={config['reward_function']}: {e}"
        )
        import traceback

        traceback.print_exc()
        return {"success": False, "config": config, "error": str(e)}
