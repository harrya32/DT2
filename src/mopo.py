from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, TensorDataset

from .datasets import OfflineDataset
from .policies import GaussianLinearPolicy, TorchPolicy
from .utils import DEVICE, set_seed


# -----------------------------------------------------------------------------
# MOPO paper-aligned defaults (Appendix G.2, arXiv:2005.13239)
# -----------------------------------------------------------------------------
# - Ensemble size: 7, keep best 5 on holdout.
# - Dynamics architecture: 4-layer feedforward, hidden size 200.
# - Two-head output: mean + variance.
# - Spectral normalization on all layers except variance head.
# - Batch size 256 for SAC updates; we mirror this for dynamics minibatches.


@dataclass(frozen=True)
class MopoDefaults:
    hidden_dim: int = 200
    hidden_layers: int = 4
    ensemble_size: int = 7
    elite_size: int = 5
    holdout_size: int = 1000
    batch_size: int = 256
    epochs: int = 2000
    lr: float = 3e-4


def get_mopo_defaults(env_name: Optional[str] = None) -> MopoDefaults:
    # The paper reports one global architecture/ensemble recipe across domains.
    _ = env_name
    return MopoDefaults()


class MopoDynamicsModel(nn.Module):
    """MOPO-style probabilistic dynamics model for state deltas only.

    Outputs Gaussian statistics for delta_state in normalized space, then
    converts to next_state statistics in original units for rollout/sampling.
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
        state_low: Optional[torch.Tensor] = None,
        state_high: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1.")

        self.state_dim = int(state_dim)
        self.act_dim = int(act_dim)
        self.target_dim = self.state_dim  # delta_state only

        in_dim = self.state_dim + self.act_dim
        trunk_layers: List[nn.Module] = []
        for _ in range(hidden_layers):
            trunk_layers.append(spectral_norm(nn.Linear(in_dim, hidden_dim)))
            trunk_layers.append(nn.SiLU())  # swish
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*trunk_layers)
        self.mean_head = spectral_norm(nn.Linear(hidden_dim, self.target_dim))
        self.logvar_head = nn.Linear(hidden_dim, self.target_dim)

        # MBPO-style bounded log-variance parameters.
        self.max_logvar = nn.Parameter(torch.full((1, self.target_dim), 0.5))
        self.min_logvar = nn.Parameter(torch.full((1, self.target_dim), -10.0))

        self.register_buffer("state_mean", torch.zeros(self.state_dim))
        self.register_buffer("state_std", torch.ones(self.state_dim))
        self.register_buffer("action_mean", torch.zeros(self.act_dim))
        self.register_buffer("action_std", torch.ones(self.act_dim))
        self.register_buffer("delta_mean", torch.zeros(self.state_dim))
        self.register_buffer("delta_std", torch.ones(self.state_dim))

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
            self.delta_mean = deltas.mean(dim=0)
            self.delta_std = deltas.std(dim=0).clamp(min=1e-6)

    def _clamp_logvar(self, raw_logvar: torch.Tensor) -> torch.Tensor:
        # Soft bounds keep variance numerically stable during training.
        logvar = self.max_logvar - F.softplus(self.max_logvar - raw_logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return logvar

    def _normalized_outputs(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s_norm = (states - self.state_mean) / (self.state_std + 1e-8)
        a_norm = (actions - self.action_mean) / (self.action_std + 1e-8)
        x = torch.cat([s_norm, a_norm], dim=-1)
        h = self.trunk(x)
        mean_norm = self.mean_head(h)
        raw_logvar = self.logvar_head(h)
        logvar_norm = self._clamp_logvar(raw_logvar)
        return mean_norm, logvar_norm

    def nll_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        mean_norm, logvar_norm = self._normalized_outputs(states, actions)
        mean_norm = mean_norm[..., : self.state_dim]
        logvar_norm = logvar_norm[..., : self.state_dim]

        delta_target_norm = (next_states - states - self.delta_mean) / (self.delta_std + 1e-8)
        target = delta_target_norm

        inv_var = torch.exp(-logvar_norm)
        nll = 0.5 * (logvar_norm + (target - mean_norm) ** 2 * inv_var + math.log(2.0 * math.pi))
        return nll.sum(dim=-1).mean()

    def logvar_regularizer(self, coeff: float = 1e-2) -> torch.Tensor:
        return coeff * (self.max_logvar.sum() - self.min_logvar.sum())

    @torch.no_grad()
    def predict_distribution(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return mean/logvar in original units for next_state."""
        mean_norm, logvar_norm = self._normalized_outputs(states, actions)

        # Slice defensively so old checkpoints with extra heads still work.
        delta_mean_norm = mean_norm[..., : self.state_dim]
        delta_logvar_norm = logvar_norm[..., : self.state_dim]
        delta_mean = delta_mean_norm * (self.delta_std + 1e-8) + self.delta_mean

        next_state_mean = states + delta_mean
        if self.state_low is not None and self.state_high is not None:
            next_state_mean = torch.max(torch.min(next_state_mean, self.state_high), self.state_low)

        logvar_state = delta_logvar_norm + 2.0 * torch.log(self.delta_std + 1e-8)

        return next_state_mean, logvar_state

    @torch.no_grad()
    def uncertainty(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        _, logvar = self.predict_distribution(states, actions)
        std = torch.exp(0.5 * logvar)
        return torch.linalg.vector_norm(std, ord=2, dim=-1)


class MopoDynamicsEnsemble(nn.Module):
    """Ensemble wrapper for MOPO-style uncertainty penalized rollouts."""

    def __init__(
        self,
        models: Sequence[MopoDynamicsModel],
        penalty_coef: float = 1.0,
    ) -> None:
        super().__init__()
        if not models:
            raise ValueError("At least one MOPO dynamics model is required.")
        self.models = nn.ModuleList(models)
        self.penalty_coef = float(penalty_coef)

    @property
    def state_dim(self) -> int:
        return self.models[0].state_dim

    @property
    def act_dim(self) -> int:
        return self.models[0].act_dim

    def member_distributions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        means: List[torch.Tensor] = []
        logvars: List[torch.Tensor] = []
        for model in self.models:
            mean, logvar = model.predict_distribution(states, actions)
            means.append(mean)
            logvars.append(logvar)
        return torch.stack(means, dim=0), torch.stack(logvars, dim=0)  # [E, B, D], [E, B, D]

    @torch.no_grad()
    def uncertainty(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        per_member = [model.uncertainty(states, actions) for model in self.models]
        return torch.stack(per_member, dim=0).max(dim=0).values

    @torch.no_grad()
    def sample_next(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        means, logvars = self.member_distributions(states, actions)
        ensemble_size = means.size(0)

        if deterministic or ensemble_size == 1:
            mean = means.mean(dim=0)
            next_states = mean[:, : self.state_dim]
            return next_states

        batch_size = states.size(0)
        member_idx = torch.randint(0, ensemble_size, (batch_size,), device=states.device)
        batch_idx = torch.arange(batch_size, device=states.device)

        chosen_mean = means[member_idx, batch_idx, :]
        chosen_logvar = logvars[member_idx, batch_idx, :]
        noise = torch.randn_like(chosen_mean)
        sample = chosen_mean + torch.exp(0.5 * chosen_logvar) * noise

        next_states = sample[:, : self.state_dim]

        # Keep sampled states in valid range when bounds are provided.
        state_low = self.models[0].state_low
        state_high = self.models[0].state_high
        if state_low is not None and state_high is not None:
            next_states = torch.max(torch.min(next_states, state_high), state_low)

        return next_states


@dataclass
class MopoTrainingInfo:
    model_losses: List[List[float]]
    member_holdout_nll: List[float]
    member_best_holdout_nll: List[Optional[float]]
    elite_indices: List[int]
    uncertainty_mean: float
    uncertainty_std: float
    uncertainty_max: float
    penalty_coef: float
    holdout_size: int
    holdout_fraction: float
    ensemble_size: int
    elite_size: int
    hidden_dim: int
    hidden_layers: int
    epochs: int
    batch_size: int
    lr: float
    bootstrap: bool
    early_stop_patience: int
    min_epochs: int
    min_delta: float


def _as_dict(dataset: OfflineDataset | Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    return dataset.as_dict() if isinstance(dataset, OfflineDataset) else dataset


def _flatten_with_optional_mask(
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if states.ndim != 3:
        return states, actions, next_states

    bsz, tsz = states.shape[:2]
    flat_s = states.reshape(bsz * tsz, -1)
    flat_a = actions.reshape(bsz * tsz, -1)
    flat_sn = next_states.reshape(bsz * tsz, -1)

    if mask is None:
        return flat_s, flat_a, flat_sn

    keep = mask.reshape(bsz * tsz) > 0.5
    return flat_s[keep], flat_a[keep], flat_sn[keep]


def _dataset_tensors(
    dataset: OfflineDataset | Mapping[str, np.ndarray],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = _as_dict(dataset)
    states = np.asarray(data["s"], dtype=np.float32)
    actions = np.asarray(data["a"], dtype=np.float32)
    next_states = np.asarray(data["s_next"], dtype=np.float32)
    mask = np.asarray(data["mask"], dtype=np.float32) if "mask" in data else None

    states, actions, next_states = _flatten_with_optional_mask(
        states, actions, next_states, mask
    )
    return (
        torch.tensor(states, dtype=torch.float32, device=device),
        torch.tensor(actions, dtype=torch.float32, device=device),
        torch.tensor(next_states, dtype=torch.float32, device=device),
    )


@torch.no_grad()
def _mean_holdout_nll(
    model: MopoDynamicsModel,
    states: torch.Tensor,
    actions: torch.Tensor,
    next_states: torch.Tensor,
    holdout_idx: torch.Tensor,
    batch_size: int = 4096,
) -> float:
    total = 0.0
    count = 0
    for start in range(0, holdout_idx.numel(), batch_size):
        idx = holdout_idx[start : start + batch_size]
        loss = model.nll_loss(states[idx], actions[idx], next_states[idx])
        bsz = int(idx.numel())
        total += float(loss.item()) * bsz
        count += bsz
    return float(total / max(1, count))


@torch.no_grad()
def _batched_uncertainty(
    ensemble: MopoDynamicsEnsemble,
    states: torch.Tensor,
    actions: torch.Tensor,
    batch_size: int = 8192,
) -> torch.Tensor:
    values: List[torch.Tensor] = []
    for start in range(0, states.size(0), batch_size):
        sb = states[start : start + batch_size]
        ab = actions[start : start + batch_size]
        values.append(ensemble.uncertainty(sb, ab).detach().to("cpu"))
    return torch.cat(values, dim=0)


def train_mopo_dynamics_ensemble(
    dataset: OfflineDataset | Mapping[str, np.ndarray],
    ensemble_size: Optional[int] = None,
    elite_size: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    hidden_layers: Optional[int] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    holdout_size: Optional[int] = None,
    val_fraction: Optional[float] = None,
    penalty_coef: float = 1.0,
    seed: int = 0,
    env_name: Optional[str] = None,
    state_low: Optional[torch.Tensor] = None,
    state_high: Optional[torch.Tensor] = None,
    bootstrap: bool = True,
    early_stop_patience: int = 50,
    min_epochs: int = 50,
    min_delta: float = 0.0,
    device: Optional[torch.device] = None,
    log_hook: Optional[Callable[[int, int, float, Optional[float]], None]] = None,
) -> Tuple[MopoDynamicsEnsemble, MopoTrainingInfo]:
    """Train MOPO probabilistic dynamics ensemble and keep top holdout models."""
    set_seed(seed)
    device = device or DEVICE

    defaults = get_mopo_defaults(env_name)
    ensemble_size = int(defaults.ensemble_size if ensemble_size is None else ensemble_size)
    elite_size = int(defaults.elite_size if elite_size is None else elite_size)
    hidden_dim = int(defaults.hidden_dim if hidden_dim is None else hidden_dim)
    hidden_layers = int(defaults.hidden_layers if hidden_layers is None else hidden_layers)
    epochs = int(defaults.epochs if epochs is None else epochs)
    batch_size = int(defaults.batch_size if batch_size is None else batch_size)
    lr = float(defaults.lr if lr is None else lr)
    holdout_size = int(defaults.holdout_size if holdout_size is None else holdout_size)

    if ensemble_size < 1:
        raise ValueError("ensemble_size must be >= 1.")
    if elite_size < 1:
        raise ValueError("elite_size must be >= 1.")
    if elite_size > ensemble_size:
        raise ValueError("elite_size must be <= ensemble_size.")

    states, actions, next_states = _dataset_tensors(dataset, device)
    n_samples = int(states.size(0))
    if n_samples < 2:
        raise ValueError("Dataset is too small for MOPO dynamics training.")

    if val_fraction is not None and float(val_fraction) > 0.0:
        val_fraction = float(max(0.0, min(float(val_fraction), 0.9)))
        holdout_count = max(1, int(n_samples * val_fraction))
        holdout_count = min(n_samples - 1, holdout_count)
    else:
        holdout_count = int(np.clip(holdout_size, 1, n_samples - 1))
    holdout_fraction = float(holdout_count / n_samples)
    perm = torch.randperm(n_samples, device=device)
    holdout_idx = perm[:holdout_count]
    train_idx = perm[holdout_count:]

    state_dim = int(states.size(-1))
    act_dim = int(actions.size(-1))

    models: List[MopoDynamicsModel] = []
    member_losses: List[List[float]] = []
    member_holdout: List[float] = []
    member_best_holdout: List[Optional[float]] = []

    for member_idx in range(ensemble_size):
        model = MopoDynamicsModel(
            state_dim=state_dim,
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            state_low=state_low,
            state_high=state_high,
        ).to(device)
        model.fit_normalizer(
            states=states[train_idx],
            actions=actions[train_idx],
            next_states=next_states[train_idx],
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if bootstrap:
            bootstrap_idx = train_idx[torch.randint(0, train_idx.numel(), (train_idx.numel(),), device=device)]
        else:
            bootstrap_idx = train_idx

        train_ds = TensorDataset(
            states[bootstrap_idx],
            actions[bootstrap_idx],
            next_states[bootstrap_idx],
        )
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

        losses: List[float] = []
        best_state = copy.deepcopy(model.state_dict())
        best_holdout = float("inf")
        best_holdout_nll: Optional[float] = None
        epochs_without_improve = 0
        for epoch in range(epochs):
            total = 0.0
            batches = 0
            for sb, ab, snb in loader:
                loss = model.nll_loss(sb, ab, snb) + model.logvar_regularizer()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.item())
                batches += 1

            avg_loss = total / max(1, batches)
            losses.append(avg_loss)
            should_stop = False

            holdout_nll_epoch = _mean_holdout_nll(
                model=model,
                states=states,
                actions=actions,
                next_states=next_states,
                holdout_idx=holdout_idx,
                batch_size=4096,
            )
            if holdout_nll_epoch + min_delta < best_holdout:
                best_holdout = holdout_nll_epoch
                best_holdout_nll = float(holdout_nll_epoch)
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
                log_hook(member_idx, epoch, avg_loss, float(holdout_nll_epoch))

            if should_stop:
                break

        model.load_state_dict(best_state)

        holdout_nll = _mean_holdout_nll(
            model=model,
            states=states,
            actions=actions,
            next_states=next_states,
            holdout_idx=holdout_idx,
            batch_size=4096,
        )

        models.append(model)
        member_losses.append(losses)
        member_holdout.append(holdout_nll)
        member_best_holdout.append(best_holdout_nll)

    elite_size_eff = min(elite_size, len(models))
    elite_indices = np.argsort(np.asarray(member_holdout, dtype=np.float64))[:elite_size_eff].tolist()
    elite_models = [models[i] for i in elite_indices]

    ensemble = MopoDynamicsEnsemble(models=elite_models, penalty_coef=penalty_coef).to(device)
    uncertainties = _batched_uncertainty(ensemble, states, actions, batch_size=8192)
    unc_mean = float(uncertainties.mean().item())
    unc_std = float(uncertainties.std(unbiased=False).item())
    unc_max = float(uncertainties.max().item())

    info = MopoTrainingInfo(
        model_losses=member_losses,
        member_holdout_nll=[float(v) for v in member_holdout],
        member_best_holdout_nll=member_best_holdout,
        elite_indices=[int(i) for i in elite_indices],
        uncertainty_mean=unc_mean,
        uncertainty_std=unc_std,
        uncertainty_max=unc_max,
        penalty_coef=float(penalty_coef),
        holdout_size=int(holdout_count),
        holdout_fraction=float(holdout_fraction),
        ensemble_size=int(ensemble_size),
        elite_size=int(elite_size_eff),
        hidden_dim=int(hidden_dim),
        hidden_layers=int(hidden_layers),
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        bootstrap=bool(bootstrap),
        early_stop_patience=int(early_stop_patience),
        min_epochs=int(min_epochs),
        min_delta=float(min_delta),
    )
    return ensemble, info


@torch.no_grad()
def rollout_in_penalized_mdp(
    ensemble: MopoDynamicsEnsemble,
    policy: TorchPolicy | GaussianLinearPolicy,
    reward_fn_torch: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    initial_states: np.ndarray | torch.Tensor,
    horizon: int,
    gamma: float = 0.99,
    rollouts: int = 500,
    seed: int = 0,
    act_low: float = -1.0,
    act_high: float = 1.0,
    deterministic_policy: bool = True,
    deterministic_dynamics: bool = False,
) -> Tuple[float, float, Dict[str, float]]:
    """Evaluate a fixed policy with MOPO penalized reward in model rollouts."""
    if isinstance(initial_states, np.ndarray):
        s0 = torch.tensor(initial_states, dtype=torch.float32, device=next(ensemble.parameters()).device)
    else:
        s0 = initial_states.to(dtype=torch.float32, device=next(ensemble.parameters()).device)

    if s0.size(0) == 0:
        raise ValueError("No initial states were provided for rollout.")

    rng = np.random.default_rng(seed)
    idx = rng.choice(s0.size(0), size=rollouts, replace=s0.size(0) < rollouts)
    states = s0[idx].clone()

    returns = torch.zeros(rollouts, dtype=torch.float32, device=states.device)
    discount = 1.0

    uncertainty_sum = 0.0
    penalty_sum = 0.0
    reward_sum = 0.0
    step_count = 0.0

    for _ in range(horizon):
        actions = policy.sample_torch_actions(
            states,
            repeats=1,
            deterministic=deterministic_policy,
            act_low=act_low,
            act_high=act_high,
        )

        model_rewards = reward_fn_torch(states, actions).reshape(-1).to(torch.float32)
        next_states = ensemble.sample_next(
            states=states,
            actions=actions,
            deterministic=deterministic_dynamics,
        )
        uncertainty = ensemble.uncertainty(states, actions)
        penalties = ensemble.penalty_coef * uncertainty
        penalized_rewards = model_rewards - penalties

        returns = returns + discount * penalized_rewards
        discount *= gamma
        states = next_states

        uncertainty_sum += float(uncertainty.sum().item())
        penalty_sum += float(penalties.sum().item())
        reward_sum += float(model_rewards.sum().item())
        step_count += float(rollouts)

    mean = float(returns.mean().item())
    stderr = float(returns.std(unbiased=False).item() / np.sqrt(max(1, rollouts)))
    metrics = {
        "uncertainty_mean": float(uncertainty_sum / max(1.0, step_count)),
        "penalty_mean": float(penalty_sum / max(1.0, step_count)),
        "reward_mean": float(reward_sum / max(1.0, step_count)),
        # Backward-compatible alias retained for existing dashboards.
        "model_reward_mean": float(reward_sum / max(1.0, step_count)),
    }
    return mean, stderr, metrics


@torch.no_grad()
def evaluate_policies_in_penalized_mdp(
    ensemble: MopoDynamicsEnsemble,
    policies: Sequence[TorchPolicy | GaussianLinearPolicy],
    reward_fn_torch: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    initial_states: np.ndarray | torch.Tensor,
    horizon: int,
    gamma: float = 0.99,
    rollouts: int = 500,
    seed: int = 0,
    act_low: float = -1.0,
    act_high: float = 1.0,
    deterministic_policy: bool = True,
    deterministic_dynamics: bool = False,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for i, policy in enumerate(policies):
        mean, stderr, metrics = rollout_in_penalized_mdp(
            ensemble=ensemble,
            policy=policy,
            reward_fn_torch=reward_fn_torch,
            initial_states=initial_states,
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
