
# offline_ope_poc.py
# -----------------------------------------------------------
# Offline OPE PoC on a continuous MDP (Gymnasium: Pendulum-v1)
# Compares three methods:
# (1) Model-based MC: Learn p(s'|s,a), then roll out under target π.
# (2) Fitted V evaluation (state-value): Learn V^\pi off-policy (state-only),
#     optionally with per-step importance sampling (IS) ratios.
# (3) Value-aware model: Freeze V^\pi from (2), then train p(s'|s,a) with
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

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
):
    """
    Fitted Value Evaluation (state-only) with reward normalization.
    Works across continuous-control tasks (e.g. LunarLanderContinuous-v3, Pendulum-v1).
    """
    set_seed(seed)
    s = torch.tensor(dataset["s"], dtype=torch.float32, device=DEVICE)
    a = torch.tensor(dataset["a"], dtype=torch.float32, device=DEVICE)
    r = torch.tensor(dataset["r"], dtype=torch.float32, device=DEVICE)
    s_next = torch.tensor(dataset["s_next"], dtype=torch.float32, device=DEVICE)
    done = torch.tensor(dataset["done"], dtype=torch.float32, device=DEVICE)

    # --- Reward normalization (preserve long-horizon scale) ---
    r_mean = r.mean().item()
    r_std = r.std().item()
    r_norm = (r - r_mean) / (r_std + 1e-6)

    # --- Networks ---
    V = ValueNet(state_dim=s.shape[1], hidden=256).to(DEVICE)
    V_target = ValueNet(state_dim=s.shape[1], hidden=256).to(DEVICE)
    V_target.load_state_dict(V.state_dict())
    opt = torch.optim.Adam(V.parameters(), lr=lr)

    # --- Optional importance weights ---
    if use_is:
        a_np = a.cpu().numpy()
        s_np = s.cpu().numpy()
        logp_pi = target_policy.log_prob(a_np, s_np)
        logp_beta = behavior_policy.log_prob(a_np, s_np)
        w = np.exp(logp_pi - logp_beta)
        w = np.clip(w, 1e-4, 100.0)
        w = torch.tensor(w, dtype=torch.float32, device=DEVICE)
    else:
        w = torch.ones(len(s), dtype=torch.float32, device=DEVICE)

    ds = TensorDataset(s, a, r_norm, s_next, done, w)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    for ep in range(epochs):
        for sb, ab, rb, snb, db, wb in loader:
            with torch.no_grad():
                # Compute target using normalized reward but unscaled bootstrap
                y = rb + gamma * (1.0 - db) * V_target(snb)
            v = V(sb)
            loss = ((v - y) ** 2) * wb
            loss = loss.mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

        # --- Hard target update ---
        if (ep + 1) % 5 == 0:
            V_target.load_state_dict(V.state_dict())

    class RescaledVWrapper(nn.Module):
        def __init__(self, V, r_mean, r_std, gamma):
            super().__init__()
            self.V = V
            self.r_mean = r_mean
            self.r_std = r_std
            self.gamma = gamma  

        def forward(self, s):
            v_norm = self.V(s)
            # rescale: undo reward normalization and restore discount scaling
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

def train_q_fqe(dataset, target_policy, gamma=0.99,
                epochs=200, batch_size=1024, lr=3e-4, seed=0, K=10):
    set_seed(seed)
    s = torch.tensor(dataset["s"], dtype=torch.float32, device=DEVICE)
    a = torch.tensor(dataset["a"], dtype=torch.float32, device=DEVICE)
    r = torch.tensor(dataset["r"], dtype=torch.float32, device=DEVICE)
    s_next = torch.tensor(dataset["s_next"], dtype=torch.float32, device=DEVICE)
    done = torch.tensor(dataset["done"], dtype=torch.float32, device=DEVICE)

    state_dim = s.shape[1]
    act_dim = a.shape[1]
    Q = QNet(state_dim, act_dim).to(DEVICE)
    Q_t = QNet(state_dim, act_dim).to(DEVICE)
    Q_t.load_state_dict(Q.state_dict())

    opt = torch.optim.Adam(Q.parameters(), lr=lr)
    ds = TensorDataset(s, a, r, s_next, done)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    for ep in range(epochs):
        for sb, ab, rb, snb, db in loader:
            with torch.no_grad():
                # Sample K actions from target policy for each next state
                s_rep = snb.repeat_interleave(K, dim=0)
                s_np = s_rep.cpu().numpy()
                a_pi = np.array([target_policy.sample(si, np.random.default_rng()) for si in s_np],
                                dtype=np.float32)
                a_pi_t = torch.tensor(a_pi, dtype=torch.float32, device=DEVICE)
                q_next = Q_t(s_rep, a_pi_t).view(len(snb), K)
                v_next = q_next.mean(dim=1)
                y = rb + gamma * (1.0 - db) * v_next

            q = Q(sb, ab)
            loss = F.mse_loss(q, y)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(Q.parameters(), 10.0)
            opt.step()

        # --- Hard target update ---
        if (ep + 1) % 5 == 0:   # update every few epochs
            Q_t.load_state_dict(Q.state_dict())

    return Q


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
def train_value_aware_model(dataset, target_policy: GaussianLinearPolicy, V_fixed: ValueNet,
                            gamma=0.97, lambda_td=1.0, epochs=20, batch_size=1024, lr=5e-4, seed=0):
    set_seed(seed)
    s = torch.tensor(dataset["s"], dtype=torch.float32, device=DEVICE)
    a = torch.tensor(dataset["a"], dtype=torch.float32, device=DEVICE)
    s_next = torch.tensor(dataset["s_next"], dtype=torch.float32, device=DEVICE)

    V_fixed = V_fixed.to(DEVICE)
    for p in V_fixed.parameters():
        p.requires_grad_(False)

    model = DynamicsNet(state_dim=s.shape[1], act_dim=a.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    ds = TensorDataset(s, a, s_next)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    rng = np.random.default_rng(seed + 123)

    for ep in range(epochs):
        for sb, ab, snb in loader:
            # Supervised fit on observed transitions (behavior actions)
            mse_loss = model.nll(sb, ab, snb)

            # TD consistency under target policy: sample a_pi, then s'_model ~ p_theta(.|s, a_pi)
            with torch.no_grad():
                s_np = sb.cpu().numpy()
                a_pi = np.array([target_policy.sample(si, rng) for si in s_np], dtype=np.float32)
                if a_pi.ndim == 1:  # handle scalar-action envs like Pendulum
                    a_pi = a_pi[:, None]
            a_pi_t = torch.tensor(a_pi, dtype=torch.float32, device=DEVICE)  # (B, act_dim)
            sp_model = model.sample_next(sb, a_pi_t)

            # Bellman residual: (V(s) - [r(s,a_pi,sp_model) + gamma V(sp_model)])^2
            V_s = V_fixed(sb)
            r_tensor = lunarlander_reward_torch(sb, a_pi_t.squeeze(-1))

            V_sp = V_fixed(sp_model)
            td = V_s - (r_tensor + gamma * V_sp)
            td_loss = (td ** 2).mean()

            loss = (1 - lambda_td) * mse_loss + lambda_td * td_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()

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
    set_seed(1)
    env = make_env()
    gamma = 0.97
    H = 500

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Define simple linear Gaussian policies
    W_base = np.zeros((act_dim, state_dim))
    beta = GaussianLinearPolicy(W=0.3 * np.random.randn(act_dim, state_dim), std=0.3, name="behavior")

    targets = [
        GaussianLinearPolicy(W=0.1 * np.random.randn(act_dim, state_dim), std=0.6, name="pi_noisy"),
        GaussianLinearPolicy(W=0.5 * np.random.randn(act_dim, state_dim), std=0.3, name="pi_moderate"),
        GaussianLinearPolicy(W=1.0 * np.random.randn(act_dim, state_dim), std=0.2, name="pi_strong"),
    ]

    print("Behavior:", beta)
    print("Targets:", targets)

    # Smaller dataset (harder for model)
    dataset = collect_dataset(env, beta, n_episodes=100, max_steps=500, seed=0)
    print("Dataset sizes:", {k: v.shape if hasattr(v, "shape") else len(v) for k, v in dataset.items()})

    # 1. Train supervised dynamics
    dyn_supervised = train_dynamics_supervised(dataset, epochs=200, batch_size=1024, lr=1e-3, seed=0)

    print("\n=== Method 1: Model-based MC (supervised dynamics) ===")
    for pi in targets:
        est_mean, est_se = evaluate_with_model_rollouts(env, dyn_supervised, pi, n_episodes=50, H=H, gamma=gamma, seed=0)
        print(f"[{pi.name}] V^pi (model rollouts, supervised dyn): {est_mean:.3f} ± {1.96*est_se:.3f}")

    # 2. Fitted V evaluation (with hard target + optional n-step)
    print("\n=== Method 2: Fitted V Evaluation (state-only) ===")
    V_nets = {}
    for pi in targets:
        V_pi = train_value_fqe_state(
            dataset, target_policy=pi, behavior_policy=beta,
            gamma=gamma, epochs=500, batch_size=1024, lr=1e-3, use_is=True, seed=0)
        V_nets[pi.name] = V_pi
        s0 = torch.tensor(dataset["s0"], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            v_est = V_pi(s0).mean().item()
        print(f"[{pi.name}] V^pi (FQE on s0): {v_est:.3f}")

    """print("\n=== Method 2b: FQE with Q-network (hard updates) ===")
    Q_nets = {}
    for pi in targets:
        Q_pi = train_q_fqe(dataset, pi, gamma=gamma, epochs=200,
                        batch_size=1024, lr=3e-4, seed=0, K=10)
        Q_nets[pi.name] = Q_pi
        v_est = estimate_V_from_Q_on_s0(Q_pi, dataset["s0"], pi, K=64)
        print(f"[{pi.name}] V^pi (Q-FQE on s0): {v_est:.3f}")"""

    # 3. Value-aware model
    print("\n=== Method 3: Value-aware model (freeze V) ===")
    for pi in targets:
        V_fixed = V_nets[pi.name]
        dyn_va = train_value_aware_model(dataset, pi, V_fixed, gamma=gamma, lambda_td=0.1,
                                         epochs=200, batch_size=1024, lr=1e-3, seed=0)
        est_mean, est_se = evaluate_with_model_rollouts(env, dyn_va, pi, n_episodes=50, H=H, gamma=gamma, seed=0)
        print(f"[{pi.name}] V^pi (model rollouts, value-aware dyn): {est_mean:.3f} ± {1.96*est_se:.3f}")

    # 4. Ground-truth
    print("\n=== Ground-truth Monte Carlo ===")
    for pi in targets:
        true_mean, true_se = evaluate_in_real_env(env, pi, n_episodes=50, H=H, gamma=gamma, seed=0)
        print(f"[{pi.name}] True V^pi (real env): {true_mean:.3f} ± {1.96*true_se:.3f}")

if __name__ == "__main__":
    main()
