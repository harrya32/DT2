from obp.ope import OffPolicyEvaluation, InverseProbabilityWeighting
from obp.policy import BasePolicy
import torch
from stable_baselines3 import PPO
import numpy as np

class PPOEvaluationPolicy(BasePolicy):
    def __init__(self, model):
        self.model = model

    def predict_proba(self, X):
        # X: array of states
        # returns: array of shape (n_samples, n_actions)
        import numpy as np
        probs = []
        for state in X:
            action_dist = self.model.policy.get_distribution(torch.tensor(state, dtype=torch.float32))
            probs.append(action_dist.distribution.probs.detach().numpy())
        return np.array(probs)

# Load data
import pandas as pd
df = pd.read_pickle("offline_random_dataset.pkl")

# Format data for OBP
n_actions = df["action"].max() + 1
action = df["action"].to_numpy()
reward = df["reward"].to_numpy()
pscore = df["action_prob"].to_numpy()

# Get evaluation policy probabilities
ppo_policy = PPO.load("runs/ppo_sepsis/ppo_sepsis_final.zip")
eval_policy = PPOEvaluationPolicy(ppo_policy)
action_dist = eval_policy.predict_proba(np.stack(df["state"].to_numpy()))

# Run OPE
ope = OffPolicyEvaluation(
    bandit_feedback=dict(
        n_actions=n_actions,
        action=action,
        reward=reward,
        pscore=pscore,
        position=np.zeros_like(action),  # not used in simple bandits
        action_dist=action_dist
    ),
    ope_estimators=[InverseProbabilityWeighting()]
)
print(ope.summarize_off_policy_estimates())
