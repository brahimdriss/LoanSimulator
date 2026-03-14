from collections import deque
from typing import List


class PerformativeReplayBuffer:
    """
    FIFO buffer for storing recent episodes in performative settings.

    Stores complete episode data including decision-level information
    needed for computing explicit performative gradients.
    """

    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.total_episodes_added = 0

    def add_episode(self, episode_data: dict):
        """Add an episode to the buffer."""
        episode_data["buffer_id"] = self.total_episodes_added
        self.buffer.append(episode_data)
        self.total_episodes_added += 1

    def get_recent(self, n: int = None) -> List[dict]:
        """Get the n most recent episodes."""
        if n is None or n >= len(self.buffer):
            return list(self.buffer)
        return list(self.buffer)[-n:]

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

    def get_statistics(self) -> dict:
        if not self.buffer:
            return {"num_episodes": 0, "total_transitions": 0}

        total_transitions = sum(len(ep.get("decisions", [])) for ep in self.buffer)

        return {
            "num_episodes": len(self.buffer),
            "total_transitions": total_transitions,
            "capacity": self.capacity,
        }


class DecisionTracker:
    """
    Tracks individual lending decisions within an episode.

    For each decision, stores:
    - State observation
    - Policy output (approval probability)
    - Whether approved (sampled action)
    - Applicant info (group, wealth, loan amount, default prob)
    - Outcome (if approved: defaulted or not)
    - Timestamps for Hawkes computation
    """

    def __init__(self):
        self.decisions = []
        self.hawkes_events_R = []  # Timestamps of male approvals
        self.hawkes_events_B = []  # Timestamps of female approvals

    def add_decision(self, decision_data: dict):
        """
        Add a lending decision.

        Args:
            decision_data: {
                'time': float,
                'state': np.array,
                'approval_prob': float (policy output),
                'approved': bool,
                'group': str ('male' or 'female'),
                'applicant_wealth': float,
                'loan_amount': float,
                'default_prob': float,
                'defaulted': bool or None (if not approved),
                'wealth_gain': float (0 if defaulted or rejected),
                'log_prob': float (log π(a|s)),
            }
        """
        self.decisions.append(decision_data)

        # Track Hawkes events (approved loans)
        if decision_data["approved"]:
            if decision_data["group"] == "male":
                self.hawkes_events_R.append(decision_data["time"])
            else:
                self.hawkes_events_B.append(decision_data["time"])

    def get_decisions(self) -> List[dict]:
        return self.decisions

    def get_approved_decisions(self) -> List[dict]:
        return [d for d in self.decisions if d["approved"]]

    def get_group_decisions(self, group: str) -> List[dict]:
        return [d for d in self.decisions if d["group"] == group]

    def clear(self):
        self.decisions = []
        self.hawkes_events_R = []
        self.hawkes_events_B = []
