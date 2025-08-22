import gymnasium as gym
import icu_sepsis  # noqa
import numpy as np
import pandas as pd

def generate_random_dataset(env_id="Sepsis/ICU-Sepsis-v2", n_episodes=1000, seed=42):
    rng = np.random.default_rng(seed)
    env = gym.make(env_id)
    dataset = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=int(rng.integers(1e9)))
        done, truncated = False, False
        while not (done or truncated):
            action = env.action_space.sample()
            action_prob = 1.0 / env.action_space.n  # uniform prob for discrete actions
            next_obs, reward, done, truncated, info = env.step(action)

            dataset.append({
                "episode": ep,
                "state": obs,
                "action": action,
                "reward": reward,
                "next_state": next_obs,
                "done": done,
                "action_prob": action_prob
            })

            obs = next_obs

    env.close()
    df = pd.DataFrame(dataset)
    df.to_pickle("offline_random_dataset.pkl")
    print(f"Saved dataset with {len(df)} transitions.")
    return df

if __name__ == "__main__":
    generate_random_dataset()
