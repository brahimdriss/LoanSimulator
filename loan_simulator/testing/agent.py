import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from ..agent import LearnableLambdas


class PolicyNet(nn.Module):
    """Beta-distribution policy network (same architecture as training)."""

    def __init__(self, input_dim: int = 12, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.alpha_head = nn.Linear(hidden_dim, 1)
        self.beta_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        alpha = F.softplus(self.alpha_head(x)) + 1.0
        beta = F.softplus(self.beta_head(x)) + 1.0
        return alpha, beta


class TestingAgent:
    """Inference-only agent — no gradient updates."""

    def __init__(self, hidden_dim: int = 128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PolicyNet(12, hidden_dim).to(self.device)
        self.learnable_lambdas = None
        self.reward_func_name = None
        self.constraint_type = None

    def load_model(self, filepath: str):
        """Load trained model weights from a .pt checkpoint."""
        print(f"Loading model from {filepath}...")

        try:
            checkpoint = torch.load(
                filepath, map_location=self.device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(filepath, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.policy_net.eval()

        self.reward_func_name = checkpoint.get("reward_function", "unknown")
        self.constraint_type = checkpoint.get("constraint_type", "unknown")

        if "learnable_lambdas_state_dict" in checkpoint:
            self.learnable_lambdas = LearnableLambdas(
                constraint_type=self.constraint_type,
                init_lambda_wealth=checkpoint.get("final_lambda_wealth", 2.0),
                init_lambda_approval=checkpoint.get("final_lambda_approval", 2.0),
            ).to(self.device)
            self.learnable_lambdas.load_state_dict(
                checkpoint["learnable_lambdas_state_dict"]
            )

        print(f"  Model loaded successfully")
        print(f"    Reward function: {self.reward_func_name}")
        print(f"    Constraint type: {self.constraint_type}")
        return checkpoint

    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> float:
        """Get action from policy (no gradient computation).

        Args:
            obs: Observation array.
            deterministic: If True, use the mode of the Beta distribution
                           instead of sampling.
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            alpha, beta = self.policy_net(obs_tensor)

            if deterministic:
                action = (alpha - 1) / (alpha + beta - 2)
                action = torch.clamp(action, 0.0, 1.0)
            else:
                dist = Beta(alpha, beta)
                action = dist.sample()

        return action.cpu().item()
