import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from tqdm import tqdm

from scope_rl.ope.ope import OffPolicyEvaluation
from scope_rl.ope.continuous.basic_estimators import (
    SelfNormalizedTIS,
    TrajectoryWiseImportanceSampling,
)

GAMMA = 0.99
N_TRAJECTORIES = 50
BANDWIDTH = 0.5
ROLLOUT_HORIZON = 50


def collect_continuous_data(policy, env, horizon, n_trajectories=N_TRAJECTORIES):
    """Roll out the behavior policy and capture per-dimension pscores."""
    trajectories = []
    action_low, action_high = env.action_space.low, env.action_space.high

    for _ in tqdm(range(n_trajectories), desc="Collecting trajectories"):
        obs, _ = env.reset()
        done = False
        episode = {
            "state": [],
            "action": [],
            "reward": [],
            "done": [],
            "pscore": [],
        }

        for _ in range(horizon):
            if done:
                obs, _ = env.reset()
                done = False

            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                dist = policy.policy.get_distribution(obs_tensor)
                action_tensor = dist.sample()
                log_prob_vec = dist.distribution.log_prob(action_tensor).cpu().numpy()[0]

            action = action_tensor.squeeze(0).cpu().numpy()
            clipped_action = np.clip(action, action_low, action_high)
            pscore = np.exp(log_prob_vec)

            next_obs, reward, terminated, truncated, _ = env.step(clipped_action)
            done_flag = bool(terminated or truncated)

            episode["state"].append(obs)
            episode["action"].append(clipped_action)
            episode["reward"].append(reward)
            episode["done"].append(done_flag)
            episode["pscore"].append(pscore)

            obs = next_obs
            done = done_flag

        trajectories.append(episode)

    return trajectories


def build_logged_dataset(trajectories):
    """Convert episodes to the LoggedDataset mapping expected by scope_rl."""
    n_trajectories = len(trajectories)
    step_per_trajectory = len(trajectories[0]["state"])

    states = np.concatenate([np.asarray(ep["state"], dtype=np.float32) for ep in trajectories])
    actions = np.concatenate([np.asarray(ep["action"], dtype=np.float32) for ep in trajectories])
    rewards = np.concatenate([np.asarray(ep["reward"], dtype=np.float32) for ep in trajectories])
    pscores = np.concatenate([np.asarray(ep["pscore"], dtype=np.float32) for ep in trajectories])
    dones = np.concatenate([np.asarray(ep["done"], dtype=np.int32) for ep in trajectories])

    obs_dim = states.shape[1]
    action_dim = actions.shape[1]

    logged_dataset = {
        "n_trajectories": n_trajectories,
        "action_type": "continuous",
        "n_actions": None,
        "action_dim": action_dim,
        "state_dim": obs_dim,
        "step_per_trajectory": step_per_trajectory,
        "state": states,
        "action": actions,
        "reward": rewards,
        "pscore": pscores,
        "done": dones,
        "terminal": dones.copy(),
    }

    return logged_dataset


def collect_evaluation_actions(policy, states):
    eval_actions = []
    for state in states:
        action, _ = policy.predict(state, deterministic=True)
        eval_actions.append(action)
    return np.asarray(eval_actions, dtype=np.float32)


def estimate_ground_truth_return(env_id, policy, horizon, n_sim=50):
    env = gym.make(env_id)
    returns = []
    for _ in range(n_sim):
        obs, _ = env.reset()
        discount = 1.0
        total_return = 0.0
        for _ in range(horizon):
            action, _ = policy.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_return += discount * reward
            discount *= GAMMA
            obs = next_obs
            if terminated or truncated:
                break
        returns.append(total_return)
    env.close()
    return float(np.mean(returns))


if __name__ == "__main__":
    env_id = "HalfCheetah-v5"

    reference_env = gym.make(env_id)
    max_steps = reference_env.spec.max_episode_steps
    horizon = min(max_steps, ROLLOUT_HORIZON)
    reference_env.close()

    behavior_env = gym.make(env_id)
    behavior_policy = PPO("MlpPolicy", behavior_env, verbose=1, device="cpu")
    behavior_policy.learn(total_timesteps=2000)

    evaluation_env = gym.make(env_id)
    evaluation_policy = PPO("MlpPolicy", evaluation_env, verbose=1, device="cpu")
    evaluation_policy.learn(total_timesteps=50000)

    data_env = gym.make(env_id)
    raw_dataset = collect_continuous_data(behavior_policy, data_env, horizon)
    data_env.close()

    logged_dataset = build_logged_dataset(raw_dataset)

    evaluation_actions = collect_evaluation_actions(
        evaluation_policy,
        logged_dataset["state"],
    )

    ope_input = {
        "evaluation_policy": {
            "evaluation_policy_action": evaluation_actions,
            "evaluation_policy_action_dist": None,
            "state_action_value_prediction": None,
            "initial_state_value_prediction": None,
            "state_action_marginal_importance_weight": None,
            "state_marginal_importance_weight": None,
            "on_policy_policy_value": None,
            "gamma": GAMMA,
            "behavior_policy": "behavior_policy",
            "evaluation_policy": "evaluation_policy",
            "dataset_id": 0,
        }
    }

    ope = OffPolicyEvaluation(
        logged_dataset=logged_dataset,
        ope_estimators=[
            TrajectoryWiseImportanceSampling(),
            SelfNormalizedTIS(),
        ],
        bandwidth=BANDWIDTH,
    )

    policy_value = ope.estimate_policy_value(input_dict=ope_input)
    estimates = policy_value["evaluation_policy"]

    print("\n--- OPE Results ---")
    for name, value in estimates.items():
        if name == "on_policy" or value is None:
            continue
        print(f"{name}: {value:.3f}")

    true_value = estimate_ground_truth_return(env_id, evaluation_policy, horizon)
    print("True (simulated) policy value:", true_value)

    behavior_env.close()
    evaluation_env.close()
