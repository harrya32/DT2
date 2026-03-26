from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .datasets import OfflineDataset
from .policies import GaussianLinearPolicy, TorchPolicy
from .utils import DEVICE, set_seed


# -----------------------------------------------------------------------------
# Paper-faithful defaults (MOReL Appendix C.2/C.5, arXiv:2005.05951)
# -----------------------------------------------------------------------------
# - Dynamics network: 2-layer ReLU MLP
# - Ensemble size: 4
# - Optimizer: Adam, stepsize 5e-4
# - Batch size: 256
# - Epochs: 300 (Ant/Hopper/Walker), 3000 (HalfCheetah)
# - Negative reward offset (r_min(D) - offset):
#       Walker=30, Hopper=50, Ant=100, HalfCheetah=200


@dataclass(frozen=True)
class MorelEnvDefaults:
    hidden_dim: int
    epochs: int
    horizon: int
    reward_offset: float


def get_morel_env_defaults(env_name: Optional[str] = None) -> MorelEnvDefaults:
    """Return environment-specific defaults reported in the MOReL appendix.

    Args:
        env_name: Gym-style environment id/name.

    Returns:
        MorelEnvDefaults with paper-aligned values when recognized.
    """
    if not env_name:
        return MorelEnvDefaults(hidden_dim=64, epochs=300, horizon=500, reward_offset=50.0)

    name = env_name.lower()
    if "halfcheetah" in name:
        return MorelEnvDefaults(hidden_dim=64, epochs=300, horizon=500, reward_offset=200.0)
    if "ant" in name:
        return MorelEnvDefaults(hidden_dim=64, epochs=300, horizon=500, reward_offset=100.0)
    if "hopper" in name:
        return MorelEnvDefaults(hidden_dim=64, epochs=300, horizon=400, reward_offset=50.0)
    if "walker" in name:
        return MorelEnvDefaults(hidden_dim=64, epochs=300, horizon=400, reward_offset=30.0)
    return MorelEnvDefaults(hidden_dim=64, epochs=300, horizon=500, reward_offset=50.0)


class MorelDynamicsModel(nn.Module):
    """MOReL-style Gaussian dynamics mean model.

    Parameterization follows the practical section:
        f(s,a) = s + sigma_delta * MLP((s-mu_s)/sigma_s, (a-mu_a)/sigma_a)

    We optimize one-step prediction error on normalized deltas, equivalent to
    Gaussian MLE with fixed diagonal covariance.
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_dim: int = 512,
        state_low: Optional[torch.Tensor] = None,
        state_high: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.act_dim = int(act_dim)

        self.net = nn.Sequential(
            nn.Linear(state_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.register_buffer("action_mean", torch.zeros(act_dim))
        self.register_buffer("action_std", torch.ones(act_dim))
        self.register_buffer("delta_std", torch.ones(state_dim))
        self._normalizer_fitted = False

        if state_low is not None:
            self.register_buffer("state_low", state_low.detach().to(torch.float32))
        else:
            self.state_low = None

        if state_high is not None:
            self.register_buffer("state_high", state_high.detach().to(torch.float32))
        else:
            self.state_high = None

    def fit_normalizer(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            self.state_mean = states.mean(dim=0)
            self.state_std = states.std(dim=0).clamp(min=1e-6)
            self.action_mean = actions.mean(dim=0)
            self.action_std = actions.std(dim=0).clamp(min=1e-6)
            deltas = next_states - states
            self.delta_std = deltas.std(dim=0).clamp(min=1e-6)
        self._normalizer_fitted = True

    def _normalized_delta_pred(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        s_norm = (states - self.state_mean) / (self.state_std + 1e-8)
        a_norm = (actions - self.action_mean) / (self.action_std + 1e-8)
        x = torch.cat([s_norm, a_norm], dim=-1)
        return self.net(x)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        delta_norm = self._normalized_delta_pred(states, actions)
        deltas = delta_norm * (self.delta_std + 1e-8)
        next_states = states + deltas
        if self.state_low is not None and self.state_high is not None:
            next_states = torch.max(torch.min(next_states, self.state_high), self.state_low)
        return next_states

    def loss(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        pred_delta_norm = self._normalized_delta_pred(states, actions)
        target_delta_norm = (next_states - states) / (self.delta_std + 1e-8)
        return F.mse_loss(pred_delta_norm, target_delta_norm)


class MorelDynamicsEnsemble(nn.Module):
    """Ensemble wrapper with USAD and pessimistic-MDP transition logic."""

    def __init__(
        self,
        models: Sequence[MorelDynamicsModel],
        threshold: float,
        halt_reward: float,
    ) -> None:
        super().__init__()
        if not models:
            raise ValueError("At least one dynamics model is required.")
        self.models = nn.ModuleList(models)
        self.threshold = float(threshold)
        self.halt_reward = float(halt_reward)

    @property
    def state_dim(self) -> int:
        return self.models[0].state_dim

    @property
    def act_dim(self) -> int:
        return self.models[0].act_dim

    def member_next_states(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        preds = [m(states, actions) for m in self.models]
        return torch.stack(preds, dim=0)  # [E, B, D]

    def predict_next(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        preds = self.member_next_states(states, actions)
        if deterministic or preds.size(0) == 1:
            return preds.mean(dim=0)
        member_idx = torch.randint(0, preds.size(0), (states.size(0),), device=states.device)
        batch_idx = torch.arange(states.size(0), device=states.device)
        return preds[member_idx, batch_idx, :]

    def disagreement(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """MOReL practical discrepancy: max pairwise L2 distance between ensemble means."""
        preds = self.member_next_states(states, actions)  # [E, B, D]
        # [E, E, B, D] pairwise differences across ensemble members.
        pairwise = preds[:, None, :, :] - preds[None, :, :, :]
        pairwise_l2 = torch.linalg.vector_norm(pairwise, ord=2, dim=-1)  # [E, E, B]
        return pairwise_l2.flatten(0, 1).max(dim=0).values  # [B]

    def unknown_mask(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.disagreement(states, actions) > self.threshold


@dataclass
class MorelTrainingInfo:
    model_losses: List[List[float]]
    member_best_val_loss: List[Optional[float]]
    disagreement_mean: float
    disagreement_std: float
    disagreement_max: float
    threshold: float
    beta_effective: Optional[float]
    halt_reward: float
    reward_min: float
    reward_offset: float
    val_fraction: float
    early_stop_patience: int
    min_epochs: int
    min_delta: float


def _as_dict(dataset: OfflineDataset | Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    return dataset.as_dict() if isinstance(dataset, OfflineDataset) else dataset


def _flatten_with_optional_mask(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_states: np.ndarray,
    dones: np.ndarray,
    mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if states.ndim != 3:
        return states, actions, rewards, next_states, dones

    bsz, tsz = states.shape[:2]
    flat_s = states.reshape(bsz * tsz, -1)
    flat_a = actions.reshape(bsz * tsz, -1)
    flat_r = rewards.reshape(bsz * tsz)
    flat_sn = next_states.reshape(bsz * tsz, -1)
    flat_d = dones.reshape(bsz * tsz)

    if mask is None:
        return flat_s, flat_a, flat_r, flat_sn, flat_d

    keep = mask.reshape(bsz * tsz) > 0.5
    return flat_s[keep], flat_a[keep], flat_r[keep], flat_sn[keep], flat_d[keep]


def _dataset_tensors(
    dataset: OfflineDataset | Mapping[str, np.ndarray],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = _as_dict(dataset)
    states = np.asarray(data["s"], dtype=np.float32)
    actions = np.asarray(data["a"], dtype=np.float32)
    rewards = np.asarray(data["r"], dtype=np.float32)
    next_states = np.asarray(data["s_next"], dtype=np.float32)
    dones = np.asarray(data["done"], dtype=np.float32)
    mask = np.asarray(data["mask"], dtype=np.float32) if "mask" in data else None

    states, actions, rewards, next_states, dones = _flatten_with_optional_mask(
        states, actions, rewards, next_states, dones, mask
    )

    return (
        torch.tensor(states, dtype=torch.float32, device=device),
        torch.tensor(actions, dtype=torch.float32, device=device),
        torch.tensor(rewards, dtype=torch.float32, device=device),
        torch.tensor(next_states, dtype=torch.float32, device=device),
        torch.tensor(dones, dtype=torch.float32, device=device),
    )


def _split_train_val_indices(
    num_samples: int,
    val_fraction: float,
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if num_samples <= 1 or val_fraction <= 0.0:
        return torch.arange(num_samples, device=device), None

    val_fraction = float(max(0.0, min(val_fraction, 0.9)))
    val_count = max(1, int(num_samples * val_fraction))
    val_count = min(num_samples - 1, val_count)
    if val_count <= 0:
        return torch.arange(num_samples, device=device), None

    perm = torch.randperm(num_samples, device=device)
    return perm[val_count:], perm[:val_count]


def _resolve_threshold(
    disagreements: torch.Tensor,
    threshold: Optional[float],
    threshold_beta: float,
    threshold_mode: str,
    threshold_frac_of_max: float,
) -> Tuple[float, Optional[float], float, float, float]:
    d_mean = float(disagreements.mean().item())
    d_std = float(disagreements.std(unbiased=False).item())
    d_max = float(disagreements.max().item())

    if threshold is not None:
        return float(threshold), None, d_mean, d_std, d_max

    mode = threshold_mode.lower()
    if mode not in {"mean_std", "fraction_max"}:
        raise ValueError("threshold_mode must be one of {'mean_std', 'fraction_max'}.")

    if mode == "fraction_max":
        th = float(max(0.0, threshold_frac_of_max)) * d_max
        return th, None, d_mean, d_std, d_max

    # mean + beta * std with beta clipped to dataset-supported range.
    beta_max = 0.0 if d_std <= 1e-12 else max(0.0, (d_max - d_mean) / d_std)
    beta_eff = float(np.clip(threshold_beta, 0.0, beta_max))
    th = d_mean + beta_eff * d_std
    return float(th), beta_eff, d_mean, d_std, d_max


def _resolve_reward_offset(env_name: Optional[str], reward_offset: Optional[float]) -> float:
    if reward_offset is not None:
        return float(reward_offset)
    return get_morel_env_defaults(env_name).reward_offset


@torch.no_grad()
def _batched_disagreements(
    ensemble: MorelDynamicsEnsemble,
    states: torch.Tensor,
    actions: torch.Tensor,
    batch_size: int = 8192,
) -> torch.Tensor:
    values: List[torch.Tensor] = []
    for start in range(0, states.size(0), batch_size):
        sb = states[start : start + batch_size]
        ab = actions[start : start + batch_size]
        disc = ensemble.disagreement(sb, ab)
        values.append(disc.detach().to("cpu"))
    return torch.cat(values, dim=0)


@torch.no_grad()
def _mean_validation_loss(
    model: MorelDynamicsModel,
    states: torch.Tensor,
    actions: torch.Tensor,
    next_states: torch.Tensor,
    val_idx: torch.Tensor,
    batch_size: int,
) -> float:
    total = 0.0
    count = 0
    for start in range(0, val_idx.numel(), batch_size):
        idx = val_idx[start : start + batch_size]
        loss = model.loss(states[idx], actions[idx], next_states[idx])
        bsz = int(idx.numel())
        total += float(loss.item()) * bsz
        count += bsz
    return float(total / max(1, count))


def train_dynamics_ensemble(
    dataset: OfflineDataset | Mapping[str, np.ndarray],
    ensemble_size: int = 4,
    hidden_dim: Optional[int] = None,
    epochs: Optional[int] = None,
    batch_size: int = 256,
    lr: float = 5e-4,
    seed: int = 0,
    env_name: Optional[str] = None,
    reward_offset: Optional[float] = None,
    halt_reward: Optional[float] = None,
    threshold: Optional[float] = None,
    threshold_beta: float = 5.0,
    threshold_mode: str = "mean_std",
    threshold_frac_of_max: float = 1.0,
    state_low: Optional[torch.Tensor] = None,
    state_high: Optional[torch.Tensor] = None,
    bootstrap: bool = True,
    val_fraction: float = 0.1,
    early_stop_patience: int = 50,
    min_epochs: int = 50,
    min_delta: float = 0.0,
    device: Optional[torch.device] = None,
    log_hook: Optional[Callable[[int, int, float, Optional[float]], None]] = None,
) -> Tuple[MorelDynamicsEnsemble, MorelTrainingInfo]:
    """Train a MOReL-style ensemble and calibrate USAD threshold.

    Args:
        dataset: Offline transitions.
        ensemble_size: Number of dynamics models (paper default: 4).
        hidden_dim: MLP width. If None, uses env-aligned paper defaults.
        epochs: Training epochs. If None, uses env-aligned paper defaults.
        batch_size: Mini-batch size (paper default: 256).
        lr: Adam learning rate (paper default: 5e-4).
        reward_offset: Uses halt_reward = min_reward - reward_offset.
        halt_reward: If provided, overrides reward_offset logic.
        threshold: If provided, directly sets USAD threshold.
        threshold_beta: Used when threshold_mode='mean_std':
            threshold = mean(disagreement) + beta * std(disagreement).
        threshold_mode: 'mean_std' (paper-style) or 'fraction_max' (ablation-style).
        threshold_frac_of_max: Used when threshold_mode='fraction_max'.
        bootstrap: Whether to bootstrap sample each ensemble member's training set.
        val_fraction: Fraction of data reserved for validation/early stopping.
        early_stop_patience: Epochs to wait for validation improvement.
        min_epochs: Minimum epochs before applying early stopping.
        min_delta: Minimum validation improvement to reset patience.
        log_hook: Optional callback(model_idx, epoch, avg_loss, val_loss).
    """
    set_seed(seed)
    device = device or DEVICE
    defaults = get_morel_env_defaults(env_name)
    hidden_dim = int(defaults.hidden_dim if hidden_dim is None else hidden_dim)
    epochs = int(defaults.epochs if epochs is None else epochs)
    ensemble_size = int(ensemble_size)
    if ensemble_size < 1:
        raise ValueError("ensemble_size must be >= 1.")

    states, actions, rewards, next_states, _ = _dataset_tensors(dataset, device)
    n_samples = states.size(0)
    if n_samples < 2:
        raise ValueError("Dataset is too small for MOReL dynamics training.")
    train_idx, val_idx = _split_train_val_indices(n_samples, val_fraction, device)
    state_dim = states.size(-1)
    act_dim = actions.size(-1)

    model_losses: List[List[float]] = []
    member_best_val_losses: List[Optional[float]] = []
    models: List[MorelDynamicsModel] = []

    for model_idx in range(ensemble_size):
        model = MorelDynamicsModel(
            state_dim=state_dim,
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            state_low=state_low,
            state_high=state_high,
        ).to(device)
        if bootstrap:
            train_indices = train_idx[torch.randint(0, train_idx.numel(), (train_idx.numel(),), device=device)]
        else:
            train_indices = train_idx
        model.fit_normalizer(states[train_indices], actions[train_indices], next_states[train_indices])

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_ds = TensorDataset(states[train_indices], actions[train_indices], next_states[train_indices])
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

        losses: List[float] = []
        best_state = copy.deepcopy(model.state_dict()) if val_idx is not None else None
        best_val = float("inf")
        best_val_loss: Optional[float] = None
        epochs_without_improve = 0
        for epoch in range(epochs):
            total = 0.0
            batches = 0
            for sb, ab, snb in loader:
                loss = model.loss(sb, ab, snb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.item())
                batches += 1

            avg_loss = total / max(1, batches)
            losses.append(avg_loss)
            val_loss: Optional[float] = None
            should_stop = False

            if val_idx is not None:
                val_loss = _mean_validation_loss(
                    model=model,
                    states=states,
                    actions=actions,
                    next_states=next_states,
                    val_idx=val_idx,
                    batch_size=max(1, batch_size),
                )

                if val_loss + min_delta < best_val:
                    best_val = val_loss
                    best_val_loss = float(val_loss)
                    epochs_without_improve = 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    epochs_without_improve += 1

                if (
                    early_stop_patience > 0
                    and epoch + 1 >= min_epochs
                    and epochs_without_improve >= early_stop_patience
                ):
                    should_stop = True

            if log_hook is not None:
                log_hook(model_idx, epoch, avg_loss, val_loss)

            if should_stop:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        models.append(model)
        model_losses.append(losses)
        member_best_val_losses.append(best_val_loss)

    ensemble = MorelDynamicsEnsemble(models=models, threshold=0.0, halt_reward=0.0).to(device)
    disagreements = _batched_disagreements(ensemble, states, actions, batch_size=8192)
    threshold_value, beta_eff, d_mean, d_std, d_max = _resolve_threshold(
        disagreements=disagreements,
        threshold=threshold,
        threshold_beta=threshold_beta,
        threshold_mode=threshold_mode,
        threshold_frac_of_max=threshold_frac_of_max,
    )
    ensemble.threshold = float(threshold_value)

    reward_min = float(rewards.min().item())
    resolved_offset = _resolve_reward_offset(env_name=env_name, reward_offset=reward_offset)
    resolved_halt_reward = float(reward_min - resolved_offset) if halt_reward is None else float(halt_reward)
    ensemble.halt_reward = resolved_halt_reward

    info = MorelTrainingInfo(
        model_losses=model_losses,
        member_best_val_loss=member_best_val_losses,
        disagreement_mean=d_mean,
        disagreement_std=d_std,
        disagreement_max=d_max,
        threshold=threshold_value,
        beta_effective=beta_eff,
        halt_reward=resolved_halt_reward,
        reward_min=reward_min,
        reward_offset=resolved_offset if halt_reward is None else float("nan"),
        val_fraction=float(max(0.0, min(val_fraction, 0.9))),
        early_stop_patience=int(early_stop_patience),
        min_epochs=int(min_epochs),
        min_delta=float(min_delta),
    )
    return ensemble, info


def ensemble_disagreement(
    ensemble: MorelDynamicsEnsemble,
    states: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    return ensemble.disagreement(states, actions)


def usad_mask(
    ensemble: MorelDynamicsEnsemble,
    states: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    return ensemble.unknown_mask(states, actions)


@torch.no_grad()
def pessimistic_step(
    ensemble: MorelDynamicsEnsemble,
    states: torch.Tensor,
    actions: torch.Tensor,
    reward_fn_torch: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    halted: Optional[torch.Tensor] = None,
    deterministic_dynamics: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One P-MDP transition for a batch.

    Returns:
        next_states, rewards, next_halted, unknown_now
    """
    if halted is None:
        halted = torch.zeros(states.size(0), dtype=torch.bool, device=states.device)
    else:
        halted = halted.to(dtype=torch.bool, device=states.device)

    active = ~halted
    unknown_now = torch.zeros_like(halted)
    if active.any():
        unknown_now[active] = ensemble.unknown_mask(states[active], actions[active])

    next_halted = halted | unknown_now

    rewards = torch.empty(states.size(0), dtype=torch.float32, device=states.device)
    rewards[next_halted] = float(ensemble.halt_reward)
    if (~next_halted).any():
        rewards[~next_halted] = reward_fn_torch(states[~next_halted], actions[~next_halted]).to(torch.float32)

    next_states = states.clone()
    if (~next_halted).any():
        next_states[~next_halted] = ensemble.predict_next(
            states[~next_halted],
            actions[~next_halted],
            deterministic=deterministic_dynamics,
        )
    if next_halted.any():
        # HALT is represented implicitly by the boolean mask; use zeros as placeholder state.
        next_states[next_halted] = 0.0

    return next_states, rewards, next_halted, unknown_now


@torch.no_grad()
def rollout_in_pessimistic_mdp(
    ensemble: MorelDynamicsEnsemble,
    policy: TorchPolicy | GaussianLinearPolicy,
    initial_states: np.ndarray | torch.Tensor,
    reward_fn_torch: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    horizon: int,
    gamma: float = 0.99,
    rollouts: int = 500,
    seed: int = 0,
    act_low: float = -1.0,
    act_high: float = 1.0,
    deterministic_policy: bool = True,
    deterministic_dynamics: bool = True,
) -> Tuple[float, float, Dict[str, float]]:
    """Evaluate a fixed policy in the MOReL pessimistic MDP."""
    if isinstance(initial_states, np.ndarray):
        s0 = torch.tensor(initial_states, dtype=torch.float32, device=next(ensemble.parameters()).device)
    else:
        s0 = initial_states.to(dtype=torch.float32, device=next(ensemble.parameters()).device)

    if s0.size(0) == 0:
        raise ValueError("No initial states were provided for rollout.")

    rng = np.random.default_rng(seed)
    idx = rng.choice(s0.size(0), size=rollouts, replace=s0.size(0) < rollouts)
    states = s0[idx].clone()

    halted = torch.zeros(rollouts, dtype=torch.bool, device=states.device)
    returns = torch.zeros(rollouts, dtype=torch.float32, device=states.device)
    discount = 1.0

    unknown_count = 0.0
    active_steps = 0.0

    for _ in range(horizon):
        actions = policy.sample_torch_actions(
            states,
            repeats=1,
            deterministic=deterministic_policy,
            act_low=act_low,
            act_high=act_high,
        )

        active_before = (~halted).sum().item()
        next_states, rewards, halted, unknown_now = pessimistic_step(
            ensemble=ensemble,
            states=states,
            actions=actions,
            reward_fn_torch=reward_fn_torch,
            halted=halted,
            deterministic_dynamics=deterministic_dynamics,
        )

        returns = returns + discount * rewards
        discount *= gamma
        states = next_states

        unknown_count += float(unknown_now.sum().item())
        active_steps += float(active_before)

    mean = float(returns.mean().item())
    stderr = float(returns.std(unbiased=False).item() / np.sqrt(max(1, rollouts)))
    metrics = {
        "unknown_rate": float(unknown_count / max(1.0, active_steps)),
        "halted_fraction": float(halted.float().mean().item()),
    }
    return mean, stderr, metrics


@torch.no_grad()
def evaluate_policies_in_pessimistic_mdp(
    ensemble: MorelDynamicsEnsemble,
    policies: Sequence[TorchPolicy | GaussianLinearPolicy],
    initial_states: np.ndarray | torch.Tensor,
    reward_fn_torch: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    horizon: int,
    gamma: float = 0.99,
    rollouts: int = 500,
    seed: int = 0,
    act_low: float = -1.0,
    act_high: float = 1.0,
    deterministic_policy: bool = True,
    deterministic_dynamics: bool = True,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for i, policy in enumerate(policies):
        mean, stderr, metrics = rollout_in_pessimistic_mdp(
            ensemble=ensemble,
            policy=policy,
            initial_states=initial_states,
            reward_fn_torch=reward_fn_torch,
            horizon=horizon,
            gamma=gamma,
            rollouts=rollouts,
            seed=seed + i,
            act_low=act_low,
            act_high=act_high,
            deterministic_policy=deterministic_policy,
            deterministic_dynamics=deterministic_dynamics,
        )
        name = getattr(policy, "name", f"policy_{i}")
        results[name] = {"mean": mean, "stderr": stderr, **metrics}
    return results
