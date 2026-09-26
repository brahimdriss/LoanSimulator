"""
Shared Beta-policy network for PolicyGradientAgent (PG) and PePGAgentV2.

Both agents previously carried their own byte-identical copy of this class
inside _build_network / _build_policy_network. One definition guarantees
the two agents are compared with the same policy class, and it carries the
two guards that the run10 post-mortem showed were missing:

1. Fixed input scaling. The 12-dim observation mixes features on very
   different scales: loan_amount is raw (~30 +/- 10), the two arrival
   rates are ~8-15 at N=12000, and mu/100 grows to ~16 under the Matthew
   dynamics during deploy, while the rest are in [0, 1]. Unscaled, the
   hidden activations are O(100), so an Adam step of lr=1e-3 on a head row
   moves the head pre-activation by O(0.1-1) and a 1000-episode random walk
   carries it to |x| ~ 30 -- measured directly on the run10 nets (stuck
   seeds: x in [-40, -21]; approve-all seeds: alpha pre-activation up to
   +1378). The divisors below are fixed constants (not learned, not
   data-dependent) chosen so every feature is O(1) over the ranges the
   environment actually produces.

2. Lower pre-activation clamp. alpha = softplus(x) + 1 has softplus'(x) =
   sigmoid(x) ~ 1e-9 at x = -20, so once a head reaches the floor every
   gradient into it vanishes and Beta(1, 1) -- a uniform coin flip that
   ignores the applicant -- becomes an absorbing state. That is exactly
   what 6 of 30 PG social-fairness seeds converged to, and where PePG sat
   under social fairness in every seed. Clamping x >= -6 keeps
   sigmoid(x) >= 0.0025 (alpha >= 1.0025), so the policy can always leave
   the floor. The clamp has zero gradient below -6, so it is a guard, not
   the mechanism that keeps the policy away from the corner -- input
   scaling and the head weight decay set in the agents' optimizers do that.

   There is deliberately NO upper clamp on the pre-activation. The first
   pilot (fixpilot) had one at +8, which meant alpha <= softplus(8)+1 = 9
   while beta could sit at its floor of ~1, capping the expressible Beta
   mean at 9/10 = 0.90 (and the mirror case at 0.10): 30 of 36 pilot
   policies parked at exactly 0.90/0.90 with zero gradient. softplus does
   not saturate at the high end (softplus' -> 1), so the gradient needs no
   protection there; only the concentration needs bounding, and CONC_MAX
   below does that (alpha, beta <= 1000 -> mean in [0.001, 0.999]).

The parameter names (fc1, fc2, alpha_head, beta_head) are unchanged from
the old inline classes, so existing state_dicts still load.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Divisors for the 12 observation columns, in the order produced by
# IncomeEnvironment._get_observation / build_group_states:
#   X/100, S, mu_R/100, mu_B/100, lambda_R, lambda_B, rho_R, rho_B,
#   theta_approval_prob, theta_S, theta_X, loan_amount
OBS_SCALE = (10.0, 1.0, 10.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 5.0, 5.0, 50.0)

PREACT_MIN = -6.0
CONC_MAX = 1000.0


class BetaPolicyNet(nn.Module):
    def __init__(self, input_dim: int = 12, hidden_dim: int = 128):
        super().__init__()
        if input_dim != len(OBS_SCALE):
            raise ValueError(f"BetaPolicyNet expects {len(OBS_SCALE)} inputs, got {input_dim}")
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.alpha_head = nn.Linear(hidden_dim, 1)
        self.beta_head = nn.Linear(hidden_dim, 1)
        self.register_buffer("obs_scale", torch.tensor(OBS_SCALE, dtype=torch.float32))

    def head_parameters(self):
        """The two output heads -- the parameters the agents apply weight
        decay to (a restoring force toward the uninformative prior that does
        not pass through the saturating softplus gradient)."""
        return list(self.alpha_head.parameters()) + list(self.beta_head.parameters())

    def body_parameters(self):
        return list(self.fc1.parameters()) + list(self.fc2.parameters())

    def preactivations(self, x):
        """(alpha_preact, beta_preact) after input scaling and the hidden
        layers, before clamp/softplus -- exposed for diagnostics."""
        x = x / self.obs_scale
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.alpha_head(x), self.beta_head(x)

    def forward(self, x):
        xa, xb = self.preactivations(x)
        xa = torch.clamp(xa, min=PREACT_MIN)   # lower guard only, see module docstring
        xb = torch.clamp(xb, min=PREACT_MIN)
        alpha = torch.clamp(F.softplus(xa) + 1.0, max=CONC_MAX)
        beta = torch.clamp(F.softplus(xb) + 1.0, max=CONC_MAX)
        return alpha, beta


HEAD_WEIGHT_DECAY = 0.1


def make_optimizer(net: BetaPolicyNet, lr: float,
                   head_weight_decay: float = HEAD_WEIGHT_DECAY) -> torch.optim.Optimizer:
    """AdamW with decoupled weight decay on the two output heads only
    (body: no decay). Decay is a restoring force toward the uninformative
    prior (alpha = beta = 1 + softplus(0)) that acts on the head weights
    directly, i.e. it does NOT pass through softplus'(x), so it can pull a
    head back from a saturated corner where the reward gradient has
    vanished. At lr=1e-3 the per-step shrink is lr*wd = 1e-4, ~30% over a
    4000-episode train+deploy -- a mild prior, not a constraint. Used by
    both agents so they share one optimizer definition."""
    return torch.optim.AdamW(
        [
            {"params": net.body_parameters(), "weight_decay": 0.0},
            {"params": net.head_parameters(), "weight_decay": head_weight_decay},
        ],
        lr=lr,
    )
