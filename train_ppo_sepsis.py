
import os
import argparse
import numpy as np
import gymnasium as gym
import icu_sepsis  # noqa: F401  # ensures env is registered
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.utils import set_random_seed

def linear_schedule(initial_value: float):
    """Linear learning rate schedule."""
    def func(progress_remaining: float):
        return progress_remaining * initial_value
    return func

def select_policy(env):
    obs_space = env.observation_space
    # Multi-Input (Dict) observations
    if isinstance(obs_space, gym.spaces.Dict):
        return "MultiInputPolicy"
    # Image-like observations
    if isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 3:
        return "CnnPolicy"
    # Default: flat / vector observations
    return "MlpPolicy"

def evaluate(model, env_id: str, n_episodes: int = 10, seed: int = 0):
    """Run evaluation episodes and return mean reward."""
    rng = np.random.RandomState(seed)
    episode_rewards = []
    for ep in range(n_episodes):
        env = gym.make(env_id)
        obs, info = env.reset(seed=int(rng.randint(0, 1e9)))
        done = False
        truncated = False
        ep_reward = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += float(reward)
        episode_rewards.append(ep_reward)
        env.close()
    return float(np.mean(episode_rewards)), float(np.std(episode_rewards))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="Sepsis/ICU-Sepsis-v2", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--total-timesteps", default=1_000_000, type=int)
    parser.add_argument("--n-envs", default=8, type=int)
    parser.add_argument("--batch-size", default=2048, type=int)
    parser.add_argument("--n-steps", default=4096, type=int, help="rollout length per env before update")
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--gae-lambda", default=0.95, type=float)
    parser.add_argument("--clip-range", default=0.2, type=float)
    parser.add_argument("--ent-coef", default=0.0, type=float)
    parser.add_argument("--vf-coef", default=0.5, type=float)
    parser.add_argument("--device", default="auto", type=str)
    parser.add_argument("--logdir", default="runs/ppo_sepsis", type=str)
    parser.add_argument("--eval-every", default=50_000, type=int, help="timesteps between evaluations")
    parser.add_argument("--eval-episodes", default=10, type=int)
    parser.add_argument("--save-every", default=100_000, type=int, help="timesteps between checkpoints")
    parser.add_argument("--lr-linear-decay", action="store_true", help="use linear lr schedule")
    args = parser.parse_args()

    set_random_seed(args.seed)
    os.makedirs(args.logdir, exist_ok=True)

    # Build vectorized training env
    vec_env = make_vec_env(
        args.env_id,
        n_envs=args.n_envs,
        seed=args.seed,
        env_kwargs=dict(render_mode=None),
    )
    vec_env = VecMonitor(vec_env)  # logs episode rewards/lengths

    # Probe a single env to pick policy
    probe_env = gym.make(args.env_id)
    policy = select_policy(probe_env)
    probe_env.close()

    # Callbacks: evaluation + checkpointing
    eval_env = gym.make(args.env_id, render_mode=None)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.logdir, "best"),
        log_path=os.path.join(args.logdir, "eval"),
        eval_freq=max(args.eval_every // args.n_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_every // args.n_envs, 1),
        save_path=os.path.join(args.logdir, "checkpoints"),
        name_prefix="ppo_sepsis",
    )

    lr = linear_schedule(args.lr) if args.lr_linear_decay else args.lr

    model = PPO(
        policy=policy,
        env=vec_env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        device=args.device,
        verbose=1,
        tensorboard_log=args.logdir,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(args.logdir, "ppo_sepsis_final")
    model.save(final_path)
    print(f"Saved final model to: {final_path}")

    # Evaluate final model
    mean_r, std_r = evaluate(model, args.env_id, n_episodes=args.eval_episodes, seed=args.seed)
    print(f"Evaluation over {args.eval_episodes} episodes -> mean reward: {mean_r:.3f} +/- {std_r:.3f}")

    vec_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
