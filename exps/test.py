"""
# -----------------------
# Environment & Reward
# -----------------------
"""
#     supervised MSE + Bellman TD residual where s' is sampled from the model
#     using target policy actions. Evaluate via model rollouts.
#
# Notes:
# - Reward function is known (Pendulum's true reward).
# - Separate V network per target policy, as requested.
# - Simple MLPs for both value and dynamics models (you can swap Dynamics MLP
#   with a Transformer if desired).
# - Target policies are Gaussian with different gains/variances.
# - This is a *minimal* research PoC, not production code.
#
# Requirements:
#   pip install gymnasium torch numpy
# Optional for rendering/debug:
#   pip install matplotlib
#
# Run:
#   python offline_ope_poc.py
# -----------------------------------------------------------

import argparse
import math
import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

METHOD_CHOICES = ("model", "value", "qvalue", "value-aware", "q-aware", "ground-truth")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline OPE proof-of-concept runner for LunarLanderContinuous-v3."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHOD_CHOICES,
        default=list(METHOD_CHOICES),
        help="Subset of evaluation methods to run. Defaults to all methods.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Base random seed for reproducibility.")
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=5,
        help="Number of consecutive seeds to run (starting from --seed).",
    )
    parser.add_argument("--gamma", type=float, default=0.97, help="Discount factor.")
    parser.add_argument(
        "--horizon",
        type=int,
        default=500,
        help="Rollout horizon for model rollouts and ground truth.",
    )
    parser.add_argument(
        "--dataset-episodes",
        type=int,
        default=100,
        help="Number of behavior-policy episodes collected for the offline dataset.",
    )
    parser.add_argument(
        "--dataset-max-steps",
        type=int,
        default=500,
        help="Maximum number of steps per behavior-policy episode.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes for model-based and ground-truth estimates.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Mini-batch size used across training routines.",
    )
    parser.add_argument(
        "--dyn-epochs",
        type=int,
        default=200,
        help="Training epochs for the supervised dynamics model.",
    )
    parser.add_argument(
        "--dyn-lr",
        type=float,
        default=1e-3,
        help="Learning rate for the supervised dynamics model.",
    )
    parser.add_argument(
        "--value-epochs",
        type=int,
        default=500,
        help="Training epochs for state-value FQE.",
    )
    parser.add_argument(
        "--value-lr",
        type=float,
        default=1e-3,
        help="Learning rate for state-value FQE.",
    )
    parser.add_argument(
        "--q-epochs",
        type=int,
        default=500,
        help="Training epochs for Q-function FQE.",
    )
    parser.add_argument(
        "--q-lr",
        type=float,
        default=1e-3,
        help="Learning rate for Q-function FQE.",
    )
    parser.add_argument(
        "--q-action-samples",
        type=int,
        default=10,
        help="Number of action samples per next-state when building Q targets.",
    )
    parser.add_argument(
        "--q-eval-samples",
        type=int,
        default=64,
        help="Number of action samples per initial state when estimating values from Q.",
    )
    parser.add_argument(
        "--value-aware-epochs",
        type=int,
        default=200,
        help="Training epochs for the value-aware dynamics model.",
    )
    parser.add_argument(
        "--value-aware-lambda",
        type=float,
        default=0.1,
        help="Weight on the TD consistency term for the value-aware dynamics model.",
    )
    parser.add_argument(
        "--value-aware-lr",
        type=float,
        default=1e-3,
        help="Learning rate for the value-aware dynamics model.",
    )
    parser.add_argument(
        "--q-aware-epochs",
        type=int,
        default=200,
        help="Training epochs for the Q-aware dynamics model.",
    )
    parser.add_argument(
        "--q-aware-lr",
        type=float,
        default=5e-4,
        help="Learning rate for the Q-aware dynamics model.",
    )
    parser.add_argument(
        "--q-aware-lambda",
        type=float,
        default=0.1,
        help="Weight on the TD consistency term for the Q-aware dynamics model.",
    )
    parser.add_argument(
        "--q-aware-action-samples",
        type=int,
        default=4,
        help="Number of target-policy action samples when computing Q-aware TD targets.",
    )
    return parser.parse_args()


# -----------------------
# Utilities & Seeding
# -----------------------
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------
# Environment & Reward (LunarLanderContinuous)
# -----------------------
def make_env():
    env = gym.make("LunarLanderContinuous-v3")
    return env

def lunarlander_reward_fn(state: np.ndarray, action: np.ndarray) -> float:
    """
    Approximation of LunarLanderContinuous-v2's true reward.
    Args:
        state: (8,) [x, y, xdot, ydot, theta, thetadot, leg1_contact, leg2_contact]
        action: (2,) [main engine, side engine]
    Returns:
        float reward
    """
    x, y, xdot, ydot, theta, thetadot, leg1, leg2 = state
    # Gym's internal reward shaping (see environment source)
    shaping = (
        -100 * np.sqrt(x**2 + y**2)
        - 100 * np.sqrt(xdot**2 + ydot**2)
        - 100 * abs(theta)
        - 10 * abs(thetadot)
        + (leg1 + leg2) * 10
    )
    # Action penalty
    reward = shaping - 0.3 * np.square(action).sum()
    return reward

def lunarlander_reward_torch(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    x, y, xdot, ydot, theta, thetadot, leg1, leg2 = state.unbind(-1)
    shaping = (
        -100 * torch.sqrt(x**2 + y**2)
        - 100 * torch.sqrt(xdot**2 + ydot**2)
        - 100 * torch.abs(theta)
        - 10 * torch.abs(thetadot)
        + (leg1 + leg2) * 10
    )
    reward = shaping - 0.3 * torch.sum(action**2, dim=-1)
    return reward

# -----------------------
# Policies (Gaussian linear policies)
# -----------------------
class GaussianLinearPolicy:
    def __init__(self, W: np.ndarray, std: float, name: str):
        """
        Linear-Gaussian policy: a ~ N(W @ s, std^2 I)
        W: (act_dim, state_dim)
        """
        self.W = W
        self.std = std
        self.name = name

    def mean_action(self, s: np.ndarray) -> np.ndarray:
        mu = self.W @ s
        return np.clip(mu, -1.0, 1.0)

    def sample(self, s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mu = self.mean_action(s)
        return np.clip(rng.normal(mu, self.std), -1.0, 1.0)

    def log_prob(self, a: np.ndarray, s: np.ndarray) -> np.ndarray:
        mu = np.einsum("ij,bj->bi", self.W, s)
        var = self.std ** 2
        return -0.5 * np.sum(np.log(2 * np.pi * var) + ((a - mu) ** 2) / var, axis=1)

    def __repr__(self):
        return f"GaussianLinearPolicy(name={self.name}, std={self.std})"

# -----------------------
# Dataset collection under known behavior policy β
# -----------------------
def collect_dataset(env, behavior_policy: GaussianLinearPolicy, n_episodes=100, max_steps=500, seed=0):
    rng = np.random.default_rng(seed)
    D = []  # (s, a, r, s_next, done)
    initial_states = []

    for ep in range(n_episodes):
        s, _ = env.reset(seed=seed + ep)
        initial_states.append(s.copy())
        for t in range(max_steps):
            a = behavior_policy.sample(s, rng)
            s_next, r, terminated, truncated, _info = env.step(a)
            done = terminated or truncated
            #D.append((s.copy(), a.copy(), r, s_next.copy(), done))
            r = lunarlander_reward_fn(s, a)
            D.append((s.copy(), a.copy(), r, s_next.copy(), done))
            s = s_next
            if done:
                break

    D = tuple(map(np.array, zip(*D)))
    return {
        "s": np.stack(D[0], axis=0),
        "a": np.stack(D[1], axis=0),
        "r": np.array(D[2], dtype=np.float32),
        "s_next": np.stack(D[3], axis=0),
        "done": np.array(D[4], dtype=np.bool_),
        "s0": np.stack(initial_states, axis=0),
    }

"""
# -----------------------
# Environment & Reward
# -----------------------
def make_env():
    env = gym.make("Pendulum-v1")
    return env

def pendulum_reward(state, action):
    # state: (cosθ, sinθ, θdot)
    # reward = -(θ^2 + 0.1*θdot^2 + 0.001*a^2)
    cos_th, sin_th, thdot = state
    theta = math.atan2(sin_th, cos_th)
    a = np.clip(action, -2.0, 2.0)
    cost = theta**2 + 0.1 * (thdot**2) + 0.001 * (a**2)
    return -cost

def pendulum_reward_torch(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    state = state.to(dtype=torch.float32)
    action = action.to(dtype=torch.float32)
    theta = torch.atan2(state[..., 1], state[..., 0])
    thdot = state[..., 2]
    a = action.clamp(-2.0, 2.0)
    cost = theta.pow(2) + 0.1 * thdot.pow(2) + 0.001 * a.pow(2)
    return -cost

# -----------------------
# Policies (Gaussian)
# -----------------------
class GaussianPolicy:
    def __init__(self, K: float, std: float, name: str):

        self.K = K
        self.std = std
        self.name = name

    def mean_action(self, s: np.ndarray) -> float:
        # s = [cos θ, sin θ, θdot]
        theta = math.atan2(s[1], s[0])
        thdot = s[2]
        mu = self.K * (-theta) - 0.1 * thdot
        return float(mu)

    def sample(self, s: np.ndarray, rng: np.random.Generator) -> float:
        mu = self.mean_action(s)
        return float(rng.normal(mu, self.std))

    def log_prob(self, a: np.ndarray, s: np.ndarray) -> np.ndarray:
        # a and s can be batched; std is scalar
        mu = np.array([self.mean_action(si) for si in s])
        var = self.std ** 2
        return -0.5 * (np.log(2 * np.pi * var) + ((a - mu) ** 2) / var)

    def __repr__(self):
        return f"GaussianPolicy(name={self.name}, K={self.K}, std={self.std})"


# -----------------------
# Dataset collection under known behavior policy β
# -----------------------
def collect_dataset(env, behavior_policy: GaussianPolicy, n_episodes=200, max_steps=200, seed=0):
    rng = np.random.default_rng(seed)
    D = []  # list of (s, a, r, s_next, done)
    initial_states = []

    for ep in range(n_episodes):
        s, _ = env.reset(seed=seed + ep)
        initial_states.append(s.copy())
        for t in range(max_steps):
            a = behavior_policy.sample(s, rng)
            s_next, r, terminated, truncated, _info = env.step(np.array([a], dtype=np.float32))
            done = terminated or truncated
            # overwrite reward using known oracle for consistency
            r = pendulum_reward(s, a)
            D.append((s.copy(), np.array([a], dtype=np.float32), r, s_next.copy(), done))
            s = s_next
            if done:
                break

    D = tuple(map(np.array, zip(*D)))  # tuple of arrays
    # Shapes: s: (N,3), a:(N,1), r:(N,), s_next:(N,3), done:(N,)
    return {
        "s": np.stack(D[0], axis=0),
        "a": np.stack(D[1], axis=0),
        "r": np.array(D[2], dtype=np.float32),
        "s_next": np.stack(D[3], axis=0),
        "done": np.array(D[4], dtype=np.bool_),
        "s0": np.stack(initial_states, axis=0),
    }
"""
# -----------------------
# Models
# -----------------------
class ValueNet(nn.Module):
    def __init__(self, state_dim=3, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, s):
        return self.net(s).squeeze(-1)  # (B,)

class DynamicsNet(nn.Module):
    def __init__(self, state_dim=3, act_dim=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # Predict Gaussian for s' given (s,a): mean and logvar per state dim
        self.mean_head = nn.Linear(hidden, state_dim)
        self.logvar_head = nn.Linear(hidden, state_dim)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        h = self.net(x)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h).clamp(-8, 3)  # stabilize
        return mean, logvar

    def nll(self, s, a, s_next):
        mean, logvar = self.forward(s, a)
        inv_var = torch.exp(-logvar)
        # Gaussian NLL per dimension
        nll = 0.5 * (logvar + (s_next - mean) ** 2 * inv_var + math.log(2 * math.pi))
        return nll.sum(dim=-1).mean()

    def mse(self, s, a, s_next):
        mean, _ = self.forward(s, a)
        return F.mse_loss(mean, s_next)

    @torch.no_grad()
    def sample_next(self, s, a):
        mean, logvar = self.forward(s, a)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps

class QNet(nn.Module):
    def __init__(self, state_dim=3, act_dim=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.net(x).squeeze(-1)

# -----------------------
# Fitted V and Q evaluation (state-only), with optional per-step IS
# -----------------------
def _log_prob_gaussian_torch(a, s, policy, device=DEVICE):
    """Torch version of GaussianLinearPolicy.log_prob for vectorized batches."""
    if not hasattr(policy, "_W_torch"):
        policy._W_torch = torch.tensor(policy.W, dtype=torch.float32, device=device)
        policy._std_torch = torch.tensor(policy.std, dtype=torch.float32, device=device)

    mu = s @ policy._W_torch.t()                     # (B, act_dim)
    var = policy._std_torch**2
    logp = -0.5 * ((a - mu) ** 2 / var + torch.log(2 * torch.pi * var))
    return logp.sum(dim=-1)                          # (B,)


def train_value_fqe_state(
    dataset,
    target_policy,
    behavior_policy,
    gamma=0.99,
    epochs=200,
    batch_size=1024,
    lr=1e-3,
    use_is=True,
    seed=0,
    use_amp=True,
):
    set_seed(seed)
    s = torch.tensor(dataset["s"], dtype=torch.float32, device=DEVICE)
    a = torch.tensor(dataset["a"], dtype=torch.float32, device=DEVICE)
    r = torch.tensor(dataset["r"], dtype=torch.float32, device=DEVICE)
    s_next = torch.tensor(dataset["s_next"], dtype=torch.float32, device=DEVICE)
    done = torch.tensor(dataset["done"], dtype=torch.float32, device=DEVICE)

    N, state_dim = s.shape
    r_mean, r_std = r.mean().item(), r.std().item()
    r_norm = (r - r_mean) / (r_std + 1e-6)

    # ---- Networks ----
    V = ValueNet(state_dim=state_dim, hidden=256).to(DEVICE)
    V_t = ValueNet(state_dim=state_dim, hidden=256).to(DEVICE)
    V_t.load_state_dict(V.state_dict())
    for p in V_t.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(V.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and DEVICE.type == "cuda"))
    idx = torch.arange(N, device=DEVICE)

    # ---- Importance weights ----
    if use_is:
        logp_pi = _log_prob_gaussian_torch(a, s, target_policy)
        logp_beta = _log_prob_gaussian_torch(a, s, behavior_policy)
        w = torch.exp(logp_pi - logp_beta).clamp(1e-4, 100.0)
    else:
        w = torch.ones(N, device=DEVICE)
    w = w / (w.mean() + 1e-8) 

    # ---- Training ----
    for ep in range(epochs):
        perm = idx[torch.randperm(N, device=DEVICE)]
        for start in range(0, N, batch_size):
            sel = perm[start:start+batch_size]
            sb, snb, rb, db, wb = s[sel], s_next[sel], r_norm[sel], done[sel], w[sel]

            with torch.no_grad():
                y = rb + gamma * (1.0 - db) * V_t(snb)

            if use_amp and DEVICE.type == "cuda":
                with torch.amp.autocast('cuda'):
                    v = V(sb)
                    loss = ((v - y) ** 2 * wb).mean()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(V.parameters(), 10.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            else:
                v = V(sb)
                loss = ((v - y) ** 2 * wb).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(V.parameters(), 10.0)
                opt.step()

        if (ep + 1) % 5 == 0:
            V_t.load_state_dict(V.state_dict())

    # ---- Rescaling wrapper ----
    class RescaledVWrapper(nn.Module):
        def __init__(self, V, r_mean, r_std, gamma):
            super().__init__()
            self.V = V
            self.r_mean = r_mean
            self.r_std = r_std
            self.gamma = gamma
        def forward(self, s):
            v_norm = self.V(s)
            return v_norm * self.r_std + self.r_mean / (1 - self.gamma)

    return RescaledVWrapper(V, r_mean, r_std, gamma)




def train_value_fqe_state_nstep(
    dataset,
    target_policy: GaussianLinearPolicy,
    behavior_policy: GaussianLinearPolicy,
    gamma=0.97,
    epochs=200,
    batch_size=1024,
    lr=1e-3,
    use_is=True,
    n_step=5,
    seed=0,
):
    """
    Fitted Value Evaluation (state-only) with reward normalization and n-step TD targets.
    Uses hard target updates (no soft averaging) for consistency with single-step FQE.
    """
    set_seed(seed)
    s = torch.tensor(dataset["s"], dtype=torch.float32, device=DEVICE)
    a = torch.tensor(dataset["a"], dtype=torch.float32, device=DEVICE)
    r = torch.tensor(dataset["r"], dtype=torch.float32, device=DEVICE)
    s_next = torch.tensor(dataset["s_next"], dtype=torch.float32, device=DEVICE)
    done = torch.tensor(dataset["done"], dtype=torch.float32, device=DEVICE)

    N = len(s)
    state_dim = s.shape[1]

    # --- Reward normalization ---
    r_mean = r.mean().item()
    r_std = r.std().item()
    print(f"[debug] Reward mean={r_mean:.2f}, std={r_std:.2f}")
    r_norm = (r - r_mean) / (r_std + 1e-6)

    # --- Build episode segmentation ---
    done_np = done.cpu().numpy()
    episode_ends = np.where(done_np)[0]
    episode_starts = np.concatenate(([0], episode_ends[:-1] + 1))
    episode_indices = [list(range(start, end + 1)) for start, end in zip(episode_starts, episode_ends)]

    # --- Networks ---
    V = ValueNet(state_dim=state_dim, hidden=256).to(DEVICE)
    V_target = ValueNet(state_dim=state_dim, hidden=256).to(DEVICE)
    V_target.load_state_dict(V.state_dict())
    opt = torch.optim.Adam(V.parameters(), lr=lr)

    # --- IS weights ---
    if use_is:
        a_np = a.cpu().numpy()
        s_np = s.cpu().numpy()
        logp_pi = target_policy.log_prob(a_np, s_np)
        logp_beta = behavior_policy.log_prob(a_np, s_np)
        w = np.exp(logp_pi - logp_beta)
        w = np.clip(w, 1e-3, 30.0)
        w = torch.tensor(w, dtype=torch.float32, device=DEVICE)
    else:
        w = torch.ones(N, dtype=torch.float32, device=DEVICE)

    # --- Helper: compute n-step targets ---
    def compute_nstep_targets():
        y_all = torch.zeros(N, dtype=torch.float32, device=DEVICE)
        for ep_indices in episode_indices:
            L = len(ep_indices)
            for t in range(L):
                idx = ep_indices[t]
                G, gpow = 0.0, 1.0
                for k in range(n_step):
                    if t + k >= L:
                        break
                    j = ep_indices[t + k]
                    G += gpow * r_norm[j].item()
                    gpow *= gamma
                    if done[j]:
                        break
                # bootstrap from s_{t+n} if not terminal
                if (t + n_step) < L and not done[ep_indices[t + n_step - 1]]:
                    with torch.no_grad():
                        G += gpow * V_target(s[ep_indices[t + n_step]]).item()
                y_all[idx] = G
        return y_all

    # --- Training loop ---
    for ep in range(epochs):
        y = compute_nstep_targets()

        ds = TensorDataset(s, r_norm, s_next, done, w, y)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

        for sb, rb, snb, db, wb, yb in loader:
            vb = V(sb)
            wb = wb / (wb.mean() + 1e-8)
            loss = ((vb - yb) ** 2) * wb
            loss = loss.mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

        # --- Hard target update every 5 epochs ---
        if (ep + 1) % 5 == 0:
            V_target.load_state_dict(V.state_dict())

        if (ep + 1) % 20 == 0:
            with torch.no_grad():
                s0 = torch.tensor(dataset["s0"], dtype=torch.float32, device=DEVICE)
                v_est = V(s0).mean().item()
            print(f"[Epoch {ep+1:03d}] V(s0) ≈ {v_est:.3f}")

    # --- Rescale predictions back to environment scale ---
    class RescaledVWrapper(nn.Module):
        def __init__(self, V, r_mean, r_std, gamma):
            super().__init__()
            self.V = V
            self.r_mean = r_mean
            self.r_std = r_std
            self.gamma = gamma

        def forward(self, s):
            v_norm = self.V(s)
            return v_norm * self.r_std + self.r_mean / (1 - self.gamma)

    return RescaledVWrapper(V, r_mean, r_std, gamma)



@torch.no_grad()
def expect_Q_under_pi(Q, s_next, policy, K=10):
    n = s_next.size(0)
    s_rep = s_next.unsqueeze(1).repeat(1, K, 1).reshape(-1, s_next.size(1))
    # vectorized policy sampling
    s_np = s_rep.cpu().numpy()
    mu = np.array([policy.mean_action(si) for si in s_np])
    std = policy.std
    a_np = np.clip(np.random.randn(len(mu)) * std + mu, -2.0, 2.0)
    a = torch.tensor(a_np[:, None], dtype=torch.float32, device=s_next.device)
    q = Q(s_rep, a).view(n, K)
    return q.mean(dim=1)

def _sample_actions_pi_torch(states, policy, K, act_low=-1.0, act_high=1.0, device=DEVICE):
    """
    Vectorized, on-GPU action sampling for GaussianLinearPolicy.
    states: (B, state_dim) torch.float32 on device
    Returns: (B*K, act_dim) torch.float32 on device
    """
    B, state_dim = states.shape
    act_dim = policy.W.shape[0]

    # W: (act_dim, state_dim) -> to torch on device once
    if not hasattr(policy, "_W_torch"):
        policy._W_torch = torch.tensor(policy.W, dtype=torch.float32, device=device)
        policy._std_torch = torch.tensor(policy.std, dtype=torch.float32, device=device)

    # mu = s @ W^T  -> (B, act_dim), then repeat for K samples
    mu = states @ policy._W_torch.t()                       # (B, act_dim)
    mu_rep = mu.unsqueeze(1).expand(B, K, act_dim).reshape(B*K, act_dim)

    noise = torch.randn(B*K, act_dim, device=device) * policy._std_torch
    a = mu_rep + noise
    if act_low is not None and act_high is not None:
        a = a.clamp(min=act_low, max=act_high)
    return a


def train_q_fqe(
    dataset, target_policy, gamma=0.99, epochs=200, batch_size=1024, lr=3e-4, seed=0, K=16,
    act_low=-1.0, act_high=1.0, use_amp=True
):
    set_seed(seed)
    s = torch.tensor(dataset["s"], dtype=torch.float32, device=DEVICE)
    a = torch.tensor(dataset["a"], dtype=torch.float32, device=DEVICE)
    r = torch.tensor(dataset["r"], dtype=torch.float32, device=DEVICE)
    s_next = torch.tensor(dataset["s_next"], dtype=torch.float32, device=DEVICE)
    done = torch.tensor(dataset["done"], dtype=torch.float32, device=DEVICE)

    # Reward normalization
    r_mean = r.mean().item()
    r_std = r.std().item()
    r_norm = (r - r_mean) / (r_std + 1e-6)

    state_dim = s.shape[1]
    act_dim = a.shape[1]

    Q = QNet(state_dim, act_dim).to(DEVICE)
    Q_t = QNet(state_dim, act_dim).to(DEVICE)
    Q_t.load_state_dict(Q.state_dict())
    for p in Q_t.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(Q.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and DEVICE.type == "cuda"))

    N = s.size(0)
    idx = torch.arange(N, device=DEVICE)

    for ep in range(epochs):
        perm = idx[torch.randperm(N, device=DEVICE)]
        for start in range(0, N, batch_size):
            sel = perm[start:start+batch_size]
            sb, ab, rb_norm, snb, db = s[sel], a[sel], r_norm[sel], s_next[sel], done[sel]

            with torch.no_grad():
                s_rep = snb.repeat_interleave(K, dim=0)  
                a_pi = _sample_actions_pi_torch(snb, target_policy, K, act_low, act_high, device=DEVICE)
                q_next = Q_t(s_rep, a_pi).view(snb.size(0), K).mean(dim=1)
                y = rb_norm + gamma * (1.0 - db) * q_next

            # Forward & loss (AMP)
            if use_amp and DEVICE.type == "cuda":
                with torch.amp.autocast('cuda'):
                    q = Q(sb, ab)
                    loss = F.mse_loss(q, y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(Q.parameters(), 10.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            else:
                q = Q(sb, ab)
                loss = F.mse_loss(q, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(Q.parameters(), 10.0)
                opt.step()

        if (ep + 1) % 5 == 0:
            Q_t.load_state_dict(Q.state_dict())

    class RescaledQWrapper(nn.Module):
        def __init__(self, Q_norm, r_mean, r_std, gamma):
            super().__init__()
            self.Q_norm = Q_norm
            self.r_mean = r_mean
            self.r_std = r_std
            self.gamma = gamma

        def forward(self, s, a):
            q_norm = self.Q_norm(s, a)
            return q_norm * self.r_std + self.r_mean / (1 - self.gamma)

    return RescaledQWrapper(Q, r_mean, r_std, gamma)



@torch.no_grad()
def estimate_V_from_Q_on_s0(Q, s0, policy, K=64):
    s0_t = torch.tensor(s0, dtype=torch.float32, device=DEVICE)
    n = s0_t.size(0)
    s_rep = s0_t.repeat_interleave(K, dim=0)
    s_np = s_rep.cpu().numpy()
    a_pi = np.array([policy.sample(si, np.random.default_rng()) for si in s_np],
                    dtype=np.float32)
    a_pi_t = torch.tensor(a_pi, dtype=torch.float32, device=DEVICE)
    q = Q(s_rep, a_pi_t).view(n, K)
    return q.mean(dim=1).mean().item()

# -----------------------
# Model-based MC evaluation
# -----------------------
def train_dynamics_supervised(dataset, epochs=20, batch_size=1024, lr=5e-4, seed=0):
    set_seed(seed)
    s = torch.tensor(dataset["s"], dtype=torch.float32, device=DEVICE)
    a = torch.tensor(dataset["a"], dtype=torch.float32, device=DEVICE)
    s_next = torch.tensor(dataset["s_next"], dtype=torch.float32, device=DEVICE)

    model = DynamicsNet(state_dim=s.shape[1], act_dim=a.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    ds = TensorDataset(s, a, s_next)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    for ep in range(epochs):
        for sb, ab, snb in loader:
            loss = model.nll(sb, ab, snb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()

    return model

@torch.no_grad()
def evaluate_with_model_rollouts(env, model: DynamicsNet, target_policy: GaussianLinearPolicy,
                                 n_episodes=100, H=200, gamma=0.97, seed=0):
    rng = np.random.default_rng(seed)
    # Draw initial states by resetting env but do not use env dynamics afterward
    s0_list = []
    for i in range(n_episodes):
        s0, _ = env.reset(seed=seed + i)
        s0_list.append(s0)
    s0 = np.stack(s0_list, axis=0)

    # Rollouts in model
    returns = []
    for i in range(n_episodes):
        s = torch.tensor(s0[i], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        G = 0.0
        gpow = 1.0
        for t in range(H):
            a = np.asarray(target_policy.sample(s.squeeze(0).cpu().numpy(), rng), dtype=np.float32)
            a_t = torch.tensor(a, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # shape (1, act_dim)
            s_next = model.sample_next(s, a_t)
            s_next_np = s_next.squeeze(0).cpu().numpy()
            r = lunarlander_reward_fn(s.squeeze(0).cpu().numpy(), a)
            G += gpow * r
            gpow *= gamma
            s = s_next
        returns.append(G)
    return float(np.mean(returns)), float(np.std(returns) / math.sqrt(n_episodes))

# -----------------------
# Value-aware model training (Method 3)
# -----------------------
def train_value_aware_model(
    dataset,
    target_policy: GaussianLinearPolicy,
    V_fixed: nn.Module,
    gamma=0.97,
    lambda_td=1.0,
    epochs=20,
    batch_size=1024,
    lr=5e-4,
    seed=0,
    use_amp=True,
    act_low=-1.0,
    act_high=1.0,
):
    """
    Value-aware dynamics training optimized for GPU:
      - torch-native, vectorized policy sampling
      - AMP mixed precision
      - in-GPU batching (no DataLoader)
    """
    set_seed(seed)
    device = DEVICE
    s = torch.tensor(dataset["s"], dtype=torch.float32, device=device)
    a = torch.tensor(dataset["a"], dtype=torch.float32, device=device)
    s_next = torch.tensor(dataset["s_next"], dtype=torch.float32, device=device)
    N, state_dim = s.shape
    act_dim = a.shape[1]

    # Freeze value network
    V_fixed = V_fixed.to(device)
    for p in V_fixed.parameters():
        p.requires_grad_(False)

    model = DynamicsNet(state_dim=state_dim, act_dim=act_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and device.type == "cuda"))
    idx = torch.arange(N, device=device)

    # Cache policy tensors on GPU
    if not hasattr(target_policy, "_W_torch"):
        target_policy._W_torch = torch.tensor(target_policy.W, dtype=torch.float32, device=device)
        target_policy._std_torch = torch.tensor(target_policy.std, dtype=torch.float32, device=device)

    def sample_actions_pi_torch(states, K=1):
        """Vectorized GaussianLinearPolicy sampling on GPU."""
        mu = states @ target_policy._W_torch.t()       # (B, act_dim)
        noise = torch.randn_like(mu) * target_policy._std_torch
        a_pi = mu + noise
        return a_pi.clamp(min=act_low, max=act_high)

    for ep in range(epochs):
        perm = idx[torch.randperm(N, device=device)]
        for start in range(0, N, batch_size):
            sel = perm[start:start+batch_size]
            sb, ab, snb = s[sel], a[sel], s_next[sel]

            with torch.amp.autocast('cuda', enabled=(use_amp and device.type == "cuda")):
                # 1. Supervised dynamics fit
                mean, logvar = model.forward(sb, ab)
                inv_var = torch.exp(-logvar)
                nll = 0.5 * (logvar + (snb - mean).pow(2) * inv_var + math.log(2 * math.pi))
                mse_loss = nll.sum(dim=-1).mean()

                # 2. TD consistency (sample a_pi and s'_model)
                a_pi_t = sample_actions_pi_torch(sb)
                s_next_model = model.sample_next(sb, a_pi_t)
                r_tensor = lunarlander_reward_torch(sb, a_pi_t.squeeze(-1))
                td = V_fixed(sb) - (r_tensor + gamma * V_fixed(s_next_model))
                td_loss = (td.pow(2)).mean()

                loss = (1 - lambda_td) * mse_loss + lambda_td * td_loss

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

    return model


@torch.no_grad()
def evaluate_in_real_env(env, target_policy: GaussianLinearPolicy,
                         n_episodes=100, H=200, gamma=0.97, seed=0):
    """
    True Monte Carlo evaluation using the real environment dynamics.
    """
    rng = np.random.default_rng(seed)
    returns = []

    for i in range(n_episodes):
        s, _ = env.reset(seed=seed + i)
        G = 0.0
        gpow = 1.0
        for t in range(H):
            a = target_policy.sample(s, rng)
            s_next, r, terminated, truncated, _ = env.step(np.array(a, dtype=np.float32))
            # use known analytic reward for consistency
            r = lunarlander_reward_fn(s, a)
            G += gpow * r
            gpow *= gamma
            s = s_next
            if terminated or truncated:
                break
        returns.append(G)

    mean = float(np.mean(returns))
    se = float(np.std(returns) / math.sqrt(n_episodes))
    return mean, se


def run_model_based_mc(env, dataset, targets, gamma, horizon, eval_episodes,
                       batch_size, epochs, lr, seed):
    """Train a supervised dynamics model and evaluate targets via rollouts."""
    print("\n=== Method 1: Model-based MC (supervised dynamics) ===")
    dyn_model = train_dynamics_supervised(dataset, epochs=epochs, batch_size=batch_size, lr=lr, seed=seed)
    estimates = {}
    for pi in targets:
        est_mean, est_se = evaluate_with_model_rollouts(
            env, dyn_model, pi, n_episodes=eval_episodes, H=horizon, gamma=gamma, seed=seed
        )
        print(f"[{pi.name}] V^pi (model rollouts, supervised dyn): {est_mean:.3f} ± {1.96*est_se:.3f}")
        estimates[pi.name] = est_mean
    return dyn_model, estimates


def run_value_fqe_block(dataset, targets, behavior_policy, gamma, epochs, batch_size,
                        lr, seed, report_estimates=True):
    """Train state-value FQE networks for each target policy."""
    if report_estimates:
        print("\n=== Method 2: Fitted V Evaluation (state-only) ===")
    V_nets = {}
    s0 = torch.tensor(dataset["s0"], dtype=torch.float32, device=DEVICE)
    estimates = {}
    for pi in targets:
        V_pi = train_value_fqe_state(
            dataset,
            target_policy=pi,
            behavior_policy=behavior_policy,
            gamma=gamma,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            use_is=True,
            seed=seed,
        )
        V_nets[pi.name] = V_pi
        with torch.no_grad():
            v_est = V_pi(s0).mean().item()
        estimates[pi.name] = v_est
        if report_estimates:
            print(f"[{pi.name}] V^pi (FQE on s0): {v_est:.3f}")
    return V_nets, estimates


def run_q_fqe_block(dataset, targets, gamma, epochs, batch_size, lr, seed, action_samples, eval_samples):
    """Train Q-function FQE networks and report their value estimates."""
    print("\n=== Method 2b: FQE with Q-network ===")
    Q_nets = {}
    estimates = {}
    for pi in targets:
        Q_pi = train_q_fqe(
            dataset,
            target_policy=pi,
            gamma=gamma,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            K=action_samples,
        )
        Q_nets[pi.name] = Q_pi
        v_est = estimate_V_from_Q_on_s0(Q_pi, dataset["s0"], pi, K=eval_samples)
        print(f"[{pi.name}] V^pi (Q-FQE on s0): {v_est:.3f}")
        estimates[pi.name] = v_est
    return Q_nets, estimates


def run_value_aware_block(dataset, targets, value_nets, gamma, lambda_td, epochs,
                          batch_size, lr, seed, env, horizon, eval_episodes):
    """Train value-aware dynamics models using frozen state-value networks."""
    if not value_nets:
        raise ValueError("Value-aware training requires pre-trained value networks.")
    print("\n=== Method 3: Value-aware model (freeze V) ===")
    estimates = {}
    for pi in targets:
        V_fixed = value_nets.get(pi.name)
        if V_fixed is None:
            raise KeyError(f"Missing value network for policy {pi.name}")
        dyn_va = train_value_aware_model(
            dataset,
            pi,
            V_fixed,
            gamma=gamma,
            lambda_td=lambda_td,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
        )
        est_mean, est_se = evaluate_with_model_rollouts(
            env, dyn_va, pi, n_episodes=eval_episodes, H=horizon, gamma=gamma, seed=seed
        )
        print(f"[{pi.name}] V^pi (model rollouts, value-aware dyn): {est_mean:.3f} ± {1.96*est_se:.3f}")
        estimates[pi.name] = est_mean
    return estimates


def run_ground_truth_block(env, targets, gamma, horizon, eval_episodes, seed):
    """Evaluate target policies in the true environment."""
    print("\n=== Ground-truth Monte Carlo ===")
    estimates = {}
    for pi in targets:
        true_mean, true_se = evaluate_in_real_env(
            env, pi, n_episodes=eval_episodes, H=horizon, gamma=gamma, seed=seed
        )
        print(f"[{pi.name}] True V^pi (real env): {true_mean:.3f} ± {1.96*true_se:.3f}")
        estimates[pi.name] = true_mean
    return estimates


def run_q_aware_block(
    dataset,
    targets,
    q_nets,
    gamma,
    lambda_td,
    epochs,
    batch_size,
    lr,
    seed,
    env,
    horizon,
    eval_episodes,
    action_samples,
):
    """Train Q-aware dynamics models leveraging frozen Q networks."""
    if not q_nets:
        raise ValueError("Q-aware training requires pre-trained Q networks.")

    print("\n=== Method 3b: Q-aware model (freeze Q) ===")
    estimates = {}
    for pi in targets:
        Q_fixed = q_nets.get(pi.name)
        if Q_fixed is None:
            raise KeyError(f"Missing Q network for policy {pi.name}")
        dyn_qaware = train_q_aware_model(
            dataset,
            target_policy=pi,
            Q_fixed=Q_fixed,
            gamma=gamma,
            lambda_td=lambda_td,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            K=action_samples,
        )
        est_mean, est_se = evaluate_with_model_rollouts(
            env, dyn_qaware, pi, n_episodes=eval_episodes, H=horizon, gamma=gamma, seed=seed
        )
        print(f"[{pi.name}] V^pi (model rollouts, Q-aware dyn): {est_mean:.3f} ± {1.96*est_se:.3f}")
        estimates[pi.name] = est_mean
    return estimates


def train_q_aware_model(
    dataset,
    target_policy: GaussianLinearPolicy,
    Q_fixed: nn.Module,                 # frozen FQE Q(s,a)
    gamma=0.97,
    lambda_td=1.0,                      # weight on Q-TD consistency term
    epochs=20,
    batch_size=1024,
    lr=5e-4,
    seed=0,
    use_amp=True,
    act_low=-1.0,
    act_high=1.0,
    K=4,                                # action samples for E_{a'~pi} Q(s',a')
):
    """
    Q-aware dynamics training:
      L = (1 - lambda_td) * NLL(s'|s,a) + lambda_td * E_{a_pi}[ ( Q(s,a_pi) - (r(s,a_pi) + gamma * E_{a'~pi} Q(s'_model,a')) )^2 ]
    Optimized with:
      - torch-native, vectorized policy sampling on GPU
      - AMP mixed precision
      - in-GPU batching (no DataLoader)
      - vectorized K-sample expectation for next-step Q
    """
    set_seed(seed)
    device = DEVICE

    # ---- Load dataset tensors to device ----
    s      = torch.tensor(dataset["s"],      dtype=torch.float32, device=device)
    a      = torch.tensor(dataset["a"],      dtype=torch.float32, device=device)
    s_next = torch.tensor(dataset["s_next"], dtype=torch.float32, device=device)

    N, state_dim = s.shape
    act_dim = a.shape[1]

    # ---- Freeze Q network ----
    Q_fixed = Q_fixed.to(device)
    for p in Q_fixed.parameters():
        p.requires_grad_(False)

    # ---- Dynamics model ----
    model = DynamicsNet(state_dim=state_dim, act_dim=act_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and device.type == "cuda"))
    idx = torch.arange(N, device=device)

    # ---- Cache policy tensors on GPU ----
    if not hasattr(target_policy, "_W_torch"):
        target_policy._W_torch  = torch.tensor(target_policy.W,  dtype=torch.float32, device=device)
        target_policy._std_torch = torch.tensor(target_policy.std, dtype=torch.float32, device=device)

    def sample_actions_pi_torch(states, Ksamples=1):
        """
        Vectorized GaussianLinearPolicy sampler on GPU.
        states: (B, state_dim)
        returns: (B, act_dim) if Ksamples=1 else (B*K, act_dim) aligned with repeat_interleave
        """
        B = states.shape[0]
        # mu = s @ W^T  -> (B, act_dim)
        mu = states @ target_policy._W_torch.t()
        if Ksamples == 1:
            a_pi = mu + torch.randn_like(mu) * target_policy._std_torch
            return a_pi.clamp(min=act_low, max=act_high)
        else:
            mu_rep = mu.unsqueeze(1).expand(B, Ksamples, act_dim).reshape(B * Ksamples, act_dim)
            noise  = torch.randn(B * Ksamples, act_dim, device=device) * target_policy._std_torch
            a_pi   = mu_rep + noise
            return a_pi.clamp(min=act_low, max=act_high)

    for ep in range(epochs):
        perm = idx[torch.randperm(N, device=device)]
        for start in range(0, N, batch_size):
            sel = perm[start:start + batch_size]
            sb, ab, snb = s[sel], a[sel], s_next[sel]

            with torch.amp.autocast('cuda', enabled=(use_amp and device.type == "cuda")):
                # ---- 1) Supervised dynamics NLL on observed (s,a)->s' ----
                mean, logvar = model.forward(sb, ab)
                inv_var = torch.exp(-logvar)
                # Gaussian NLL per dim
                nll = 0.5 * (logvar + (snb - mean).pow(2) * inv_var + math.log(2 * math.pi))
                nll_loss = nll.sum(dim=-1).mean()

                # ---- 2) Q-aware TD consistency term ----
                # Sample action from target policy at current state
                a_pi = sample_actions_pi_torch(sb, Ksamples=1)                # (B, act_dim)
                # Model one step: s'_model ~ p_theta(.|s, a_pi)
                s_next_model = model.sample_next(sb, a_pi)                     # (B, state_dim)

                # Immediate reward r(s, a_pi) on GPU (use your env's reward fn)
                # lunarlander_reward_torch expects (B, state_dim) and (B, act_dim)
                r_tensor = lunarlander_reward_torch(sb, a_pi)                  # (B,)

                # Q_next(s'_model) = E_{a'~pi}[ Q(s'_model, a') ] via K samples
                if K > 1:
                    s_rep  = s_next_model.repeat_interleave(K, dim=0)         # (B*K, state_dim)
                    a_next = sample_actions_pi_torch(s_next_model, Ksamples=K) # (B*K, act_dim)
                    q_next = Q_fixed(s_rep, a_next).view(s_next_model.size(0), K).mean(dim=1)  # (B,)
                else:
                    a_next = sample_actions_pi_torch(s_next_model, Ksamples=1) # (B, act_dim)
                    q_next = Q_fixed(s_next_model, a_next)                      # (B,)

                # Q-TD residual: Q(s, a_pi) - (r + gamma * E_{a'} Q(s', a'))
                q_curr = Q_fixed(sb, a_pi)                                      # (B,)
                td = q_curr - (r_tensor + gamma * q_next)
                td_loss = (td.pow(2)).mean()

                # ---- Total loss ----
                loss = (1.0 - lambda_td) * nll_loss + lambda_td * td_loss

            # ---- Optimize ----
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

    return model


def compare_estimates_to_ground_truth(ground_truth, method_estimates):
    """Compute and print average absolute deviation from ground truth for each method."""
    if not ground_truth:
        print("\n[Compare] Ground-truth estimates unavailable; skipping comparison.")
        return

    gt_keys = set(ground_truth.keys())
    print("\n=== Method Comparison vs. Ground Truth ===")
    for method, estimates in method_estimates.items():
        overlap = gt_keys.intersection(estimates.keys())
        if not overlap:
            print(f"[Compare] {method}: no overlapping policies with ground truth.")
            continue
        diffs = [abs(estimates[k] - ground_truth[k]) for k in overlap]
        avg_diff = float(np.mean(diffs)) if diffs else float('nan')
        print(f"[Compare] {method}: avg |estimate - ground_truth| = {avg_diff:.3f} over {len(overlap)} policies")


# -----------------------
# Main experiment
# -----------------------
"""def main():
    set_seed(1)
    env = make_env()
    gamma = 0.97
    H = 200

    # Define behavior and target policies
    beta = GaussianPolicy(K=2.0, std=0.8, name="behavior")
    targets = [
        GaussianPolicy(K=0.2, std=1.5, name="pi_very_noisy"),
        GaussianPolicy(K=1.2, std=0.8, name="pi_weak"),
        GaussianPolicy(K=3.0, std=0.4, name="pi_strong"),
        GaussianPolicy(K=6.0, std=0.2, name="pi_overcontrol"), 
    ]

    print("Behavior:", beta)
    print("Targets:", targets)

    # Collect offline dataset under behavior policy
    dataset = collect_dataset(env, beta, n_episodes=300, max_steps=200, seed=0)
    print("Dataset sizes:", {k: v.shape if hasattr(v, "shape") else len(v) for k, v in dataset.items()})

    # Method (1): Train dynamics supervised once; reuse for all target policies
    dyn_supervised = train_dynamics_supervised(dataset, epochs=5, batch_size=1024, lr=5e-4, seed=0)

    # Evaluate each target with model-based rollouts
    print("\n=== Method 1: Model-based MC (supervised dynamics) ===")
    for pi in targets:
        est_mean, est_se = evaluate_with_model_rollouts(env, dyn_supervised, pi, n_episodes=100, H=H, gamma=gamma, seed=0)
        print(f"[{pi.name}] V^pi (model rollouts, supervised dyn): {est_mean:.3f} ± {1.96*est_se:.3f} (95% CI)")

    # Method (2): Fitted state-value evaluation per policy (with IS)
    print("\n=== Method 2: Fitted V Evaluation (state-only, off-policy) ===")
    V_nets = {}
    for pi in targets:
        V_pi = train_value_fqe_state(dataset, target_policy=pi, behavior_policy=beta,
                                     gamma=gamma, epochs=200, batch_size=1024, lr=3e-4, use_is=True, seed=0)
        V_nets[pi.name] = V_pi
        # Estimate value via averaging V over initial-state distribution
        s0 = torch.tensor(dataset["s0"], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            v_est = V_pi(s0).mean().item()
        print(f"[{pi.name}] V^pi (Fitted-V on s0): {v_est:.3f}")

    print("\n=== Method 2b: FQE with Q-network (recommended) ===")
    Q_nets = {}
    for pi in targets:
        Q_pi = train_q_fqe(dataset, pi, gamma=gamma, epochs=200, batch_size=1024, lr=3e-4, seed=0, K=10)
        Q_nets[pi.name] = Q_pi
        v_est = estimate_V_from_Q_on_s0(Q_pi, dataset["s0"], pi, K=64)
        print(f"[{pi.name}] V^pi (Q-FQE on s0): {v_est:.3f}")

    # Method (3): Value-aware dynamics per policy, freeze V from (2), evaluate by rollouts
    print("\n=== Method 3: Value-aware model (freeze V; TD-consistency with model-sampled s') ===")
    for pi in targets:
        V_fixed = V_nets[pi.name]
        dyn_va = train_value_aware_model(dataset, pi, V_fixed, gamma=gamma, lambda_td=1.0,
                                         epochs=5, batch_size=1024, lr=5e-4, seed=0)
        est_mean, est_se = evaluate_with_model_rollouts(env, dyn_va, pi, n_episodes=100, H=H, gamma=gamma, seed=0)
        print(f"[{pi.name}] V^pi (model rollouts, value-aware dyn): {est_mean:.3f} ± {1.96*est_se:.3f} (95% CI)")

    # Ground-truth Monte Carlo evaluation in the real env
    print("\n=== Ground-truth Monte Carlo in real env ===")
    for pi in targets:
        true_mean, true_se = evaluate_in_real_env(env, pi, n_episodes=100, H=H, gamma=gamma, seed=0)
        print(f"[{pi.name}] True V^pi (real env): {true_mean:.3f} ± {1.96*true_se:.3f} (95% CI)")"""


def main():
    args = parse_args()
    methods = list(dict.fromkeys(args.methods))
    print(f"Selected methods: {', '.join(methods)}")

    seeds = [args.seed + i for i in range(args.num_seeds)]
    method_estimates_acc = defaultdict(lambda: defaultdict(list))
    ground_truth_acc = defaultdict(list)

    for run_seed in seeds:
        print(f"\n===== Seed {run_seed} =====")
        set_seed(run_seed)
        env = make_env()
        gamma = args.gamma
        horizon = args.horizon

        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        beta = GaussianLinearPolicy(W=0.3 * np.random.randn(act_dim, state_dim), std=0.3, name="behavior")
        targets = [
            GaussianLinearPolicy(W=0.1 * np.random.randn(act_dim, state_dim), std=0.6, name="pi_noisy"),
            GaussianLinearPolicy(W=0.5 * np.random.randn(act_dim, state_dim), std=0.3, name="pi_moderate"),
            GaussianLinearPolicy(W=1.0 * np.random.randn(act_dim, state_dim), std=0.2, name="pi_strong"),
        ]

        print("Behavior:", beta)
        print("Targets:", targets)

        dataset = collect_dataset(
            env,
            beta,
            n_episodes=args.dataset_episodes,
            max_steps=args.dataset_max_steps,
            seed=run_seed,
        )
        print("Dataset sizes:", {k: v.shape if hasattr(v, "shape") else len(v) for k, v in dataset.items()})

        run_method_estimates = {}
        value_nets = None
        q_nets = None

        if "model" in methods:
            _, model_estimates = run_model_based_mc(
                env,
                dataset,
                targets,
                gamma,
                horizon,
                args.eval_episodes,
                args.batch_size,
                args.dyn_epochs,
                args.dyn_lr,
                run_seed,
            )
            run_method_estimates["model"] = model_estimates

        if "value" in methods:
            value_nets, value_estimates = run_value_fqe_block(
                dataset,
                targets,
                beta,
                gamma,
                args.value_epochs,
                args.batch_size,
                args.value_lr,
                run_seed,
                report_estimates=True,
            )
            run_method_estimates["value"] = value_estimates

        if "qvalue" in methods:
            q_nets, qvalue_estimates = run_q_fqe_block(
                dataset,
                targets,
                gamma,
                args.q_epochs,
                args.batch_size,
                args.q_lr,
                run_seed,
                args.q_action_samples,
                args.q_eval_samples,
            )
            run_method_estimates["qvalue"] = qvalue_estimates

        if "value-aware" in methods:
            if value_nets is None:
                value_nets, _ = run_value_fqe_block(
                    dataset,
                    targets,
                    beta,
                    gamma,
                    args.value_epochs,
                    args.batch_size,
                    args.value_lr,
                    run_seed,
                    report_estimates=False,
                )
                print("\nPrepared state-value networks for value-aware modeling.")
            value_aware_estimates = run_value_aware_block(
                dataset,
                targets,
                value_nets,
                gamma,
                args.value_aware_lambda,
                args.value_aware_epochs,
                args.batch_size,
                args.value_aware_lr,
                run_seed,
                env,
                horizon,
                args.eval_episodes,
            )
            run_method_estimates["value-aware"] = value_aware_estimates

        if "q-aware" in methods:
            if q_nets is None:
                q_nets, _ = run_q_fqe_block(
                    dataset,
                    targets,
                    gamma,
                    args.q_epochs,
                    args.batch_size,
                    args.q_lr,
                    run_seed,
                    args.q_action_samples,
                    args.q_eval_samples,
                )
                print("\nPrepared Q networks for Q-aware modeling.")
            q_aware_estimates = run_q_aware_block(
                dataset,
                targets,
                q_nets,
                gamma,
                args.q_aware_lambda,
                args.q_aware_epochs,
                args.batch_size,
                args.q_aware_lr,
                run_seed,
                env,
                horizon,
                args.eval_episodes,
                args.q_aware_action_samples,
            )
            run_method_estimates["q-aware"] = q_aware_estimates

        ground_truth_estimates = {}
        if "ground-truth" in methods:
            ground_truth_estimates = run_ground_truth_block(
                env,
                targets,
                gamma,
                horizon,
                args.eval_episodes,
                run_seed,
            )

        for method_name, estimates in run_method_estimates.items():
            for policy_name, value in estimates.items():
                method_estimates_acc[method_name][policy_name].append(value)

        for policy_name, value in ground_truth_estimates.items():
            ground_truth_acc[policy_name].append(value)

        env.close()

    method_estimates_avg = {
        method: {policy: float(np.mean(vals)) for policy, vals in policy_dict.items()}
        for method, policy_dict in method_estimates_acc.items()
    }
    ground_truth_avg = {policy: float(np.mean(vals)) for policy, vals in ground_truth_acc.items()}

    if method_estimates_avg:
        print("\n=== Average Estimates Across Seeds ===")
        for method, pol_dict in method_estimates_avg.items():
            print(f"{method}:")
            for policy, val in pol_dict.items():
                print(f"  {policy}: {val:.3f}")

    if ground_truth_avg:
        print("\n=== Ground Truth (Average Across Seeds) ===")
        for policy, val in ground_truth_avg.items():
            print(f"{policy}: {val:.3f}")

    compare_estimates_to_ground_truth(ground_truth_avg, method_estimates_avg)

if __name__ == "__main__":
    main()
