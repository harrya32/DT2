from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .datasets import OfflineDataset
from .policies import GaussianLinearPolicy, TorchPolicy
from .utils import DEVICE, set_seed


# -----------------------------------------------------------------------------
# ROMI defaults (aligned with paper Appendix D.2 / Table 3 where practical)
# -----------------------------------------------------------------------------
# - Dynamics ensemble size: 4
# - Dynamics architecture: 2 hidden layers, width 64
# - Weighting network: MLP width 256
# - Weight range: [0.5, 2.0]
# - Dynamics pretrain epochs: 50
# - Policy optimizer: SAC-style updates on mixed real/model data


@dataclass(frozen=True)
class RomiDefaults:
    dynamics_hidden_dim: int = 64
    dynamics_hidden_layers: int = 2
    weight_hidden_dim: int = 256
    weight_hidden_layers: int = 3
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256
    ensemble_size: int = 4
    batch_size: int = 256
    model_pretrain_epochs: int = 50
    epochs: int = 300
    dynamics_lr: float = 3e-4
    weight_lr: float = 1e-4
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 5e-3
    real_data_ratio: float = 0.5
    model_rollout_horizon: int = 5
    model_rollouts_per_epoch: int = 512
    policy_updates_per_epoch: int = 200
    dynamics_updates_per_epoch: int = 1
    weight_updates_per_epoch: int = 1
    uncertainty_scale: float = 0.1
    uncertainty_samples: int = 20
    weight_min: float = 0.5
    weight_max: float = 2.0
    model_buffer_capacity: int = 200_000
    bilevel_inner_lr: float = 3e-4
    bootstrap: bool = True
    automatic_entropy_tuning: bool = True
    policy_pretrain_steps: int = 0


def get_romi_defaults(env_name: Optional[str] = None) -> RomiDefaults:
    _ = env_name
    return RomiDefaults()


class RomiDynamicsModel(nn.Module):
    """Probabilistic next-state dynamics model used by ROMI.

    Model predicts Gaussian statistics for normalized state delta:
        p(s'|s,a) = N(mean_delta(s,a), diag(var_delta(s,a)))
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

        in_dim = self.state_dim + self.act_dim
        self.trunk = nn.ModuleList()
        for _ in range(hidden_layers):
            self.trunk.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim

        self.mean_head = nn.Linear(hidden_dim, self.state_dim)
        self.logvar_head = nn.Linear(hidden_dim, self.state_dim)
        self.max_logvar = nn.Parameter(torch.full((1, self.state_dim), 0.5))
        self.min_logvar = nn.Parameter(torch.full((1, self.state_dim), -10.0))

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

    def _clamp_logvar(self, raw_logvar: torch.Tensor, max_logvar: torch.Tensor, min_logvar: torch.Tensor) -> torch.Tensor:
        logvar = max_logvar - F.softplus(max_logvar - raw_logvar)
        logvar = min_logvar + F.softplus(logvar - min_logvar)
        return logvar

    def _normalized_outputs(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s_norm = (states - self.state_mean) / (self.state_std + 1e-8)
        a_norm = (actions - self.action_mean) / (self.action_std + 1e-8)
        x = torch.cat([s_norm, a_norm], dim=-1)
        for layer in self.trunk:
            x = F.silu(layer(x))
        mean_norm = self.mean_head(x)
        raw_logvar = self.logvar_head(x)
        logvar_norm = self._clamp_logvar(raw_logvar, self.max_logvar, self.min_logvar)
        return mean_norm, logvar_norm

    def _normalized_outputs_with_params(
        self,
        params: Mapping[str, torch.Tensor],
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        s_norm = (states - self.state_mean) / (self.state_std + 1e-8)
        a_norm = (actions - self.action_mean) / (self.action_std + 1e-8)
        x = torch.cat([s_norm, a_norm], dim=-1)

        for i in range(len(self.trunk)):
            w = params[f"trunk.{i}.weight"]
            b = params[f"trunk.{i}.bias"]
            x = F.silu(F.linear(x, w, b))

        mean_norm = F.linear(x, params["mean_head.weight"], params["mean_head.bias"])
        raw_logvar = F.linear(x, params["logvar_head.weight"], params["logvar_head.bias"])
        logvar_norm = self._clamp_logvar(raw_logvar, params["max_logvar"], params["min_logvar"])
        return mean_norm, logvar_norm

    def nll_loss_per_sample(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        mean_norm, logvar_norm = self._normalized_outputs(states, actions)
        delta_target_norm = (next_states - states - self.delta_mean) / (self.delta_std + 1e-8)
        inv_var = torch.exp(-logvar_norm)
        nll = 0.5 * (logvar_norm + (delta_target_norm - mean_norm) ** 2 * inv_var + math.log(2.0 * math.pi))
        return nll.sum(dim=-1)

    def nll_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.nll_loss_per_sample(states, actions, next_states).mean()

    def logvar_regularizer(self, coeff: float = 1e-2) -> torch.Tensor:
        return coeff * (self.max_logvar.sum() - self.min_logvar.sum())

    @torch.no_grad()
    def predict_distribution(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_norm, logvar_norm = self._normalized_outputs(states, actions)
        delta_mean = mean_norm * (self.delta_std + 1e-8) + self.delta_mean
        next_state_mean = states + delta_mean
        if self.state_low is not None and self.state_high is not None:
            next_state_mean = torch.max(torch.min(next_state_mean, self.state_high), self.state_low)

        logvar = logvar_norm + 2.0 * torch.log(self.delta_std + 1e-8)
        return next_state_mean, logvar

    def predict_distribution_with_params(
        self,
        params: Mapping[str, torch.Tensor],
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_norm, logvar_norm = self._normalized_outputs_with_params(params, states, actions)
        delta_mean = mean_norm * (self.delta_std + 1e-8) + self.delta_mean
        next_state_mean = states + delta_mean
        if self.state_low is not None and self.state_high is not None:
            next_state_mean = torch.max(torch.min(next_state_mean, self.state_high), self.state_low)

        logvar = logvar_norm + 2.0 * torch.log(self.delta_std + 1e-8)
        return next_state_mean, logvar

    @torch.no_grad()
    def sample_next(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        mean, logvar = self.predict_distribution(states, actions)
        if deterministic:
            return mean
        noise = torch.randn_like(mean)
        sample = mean + torch.exp(0.5 * logvar) * noise
        if self.state_low is not None and self.state_high is not None:
            sample = torch.max(torch.min(sample, self.state_high), self.state_low)
        return sample


class RomiDynamicsEnsemble(nn.Module):
    """Ensemble wrapper for ROMI dynamics models."""

    def __init__(self, models: Sequence[RomiDynamicsModel]) -> None:
        super().__init__()
        if not models:
            raise ValueError("At least one ROMI dynamics model is required.")
        self.models = nn.ModuleList(models)

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
        return torch.stack(means, dim=0), torch.stack(logvars, dim=0)

    @torch.no_grad()
    def uncertainty(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        per_member = []
        for model in self.models:
            _, logvar = model.predict_distribution(states, actions)
            std = torch.exp(0.5 * logvar)
            per_member.append(torch.linalg.vector_norm(std, ord=2, dim=-1))
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
            return means.mean(dim=0)

        batch_size = states.size(0)
        member_idx = torch.randint(0, ensemble_size, (batch_size,), device=states.device)
        batch_idx = torch.arange(batch_size, device=states.device)
        chosen_mean = means[member_idx, batch_idx, :]
        chosen_logvar = logvars[member_idx, batch_idx, :]
        noise = torch.randn_like(chosen_mean)
        next_states = chosen_mean + torch.exp(0.5 * chosen_logvar) * noise

        state_low = self.models[0].state_low
        state_high = self.models[0].state_high
        if state_low is not None and state_high is not None:
            next_states = torch.max(torch.min(next_states, state_high), state_low)
        return next_states


class AdaptiveWeightNet(nn.Module):
    """Adaptive sample reweighting network w_nu(s, a, s')."""

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 3,
        weight_min: float = 0.5,
        weight_max: float = 2.0,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1.")
        if weight_max <= weight_min:
            raise ValueError("weight_max must be > weight_min.")

        in_dim = state_dim + act_dim + state_dim
        layers: List[nn.Module] = []
        cur = in_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(cur, hidden_dim))
            layers.append(nn.SiLU())
            cur = hidden_dim
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(cur, 1)
        self.weight_min = float(weight_min)
        self.weight_max = float(weight_max)

    def forward(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, actions, next_states], dim=-1)
        h = self.body(x)
        raw = self.out(h).squeeze(-1)
        scale = 0.5 * (self.weight_max - self.weight_min)
        shift = 0.5 * (self.weight_max + self.weight_min)
        return torch.tanh(raw) * scale + shift


class TanhGaussianActor(nn.Module):
    """SAC-style tanh Gaussian actor that satisfies TorchPolicy protocol."""

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        name: str = "romi_policy",
    ) -> None:
        super().__init__()
        self.name = name
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)

    def _mean_log_std(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(states)
        mean = self.mean_head(h)
        raw_log_std = self.log_std_head(h)
        log_std = torch.tanh(raw_log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1.0)
        return mean, log_std

    def sample_with_logprob(
        self,
        states: torch.Tensor,
        deterministic: bool = False,
        act_low: float = -1.0,
        act_high: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self._mean_log_std(states)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)

        if deterministic:
            pre_tanh = mean
        else:
            pre_tanh = dist.rsample()
        tanh_action = torch.tanh(pre_tanh)

        action_scale = 0.5 * (act_high - act_low)
        action_bias = 0.5 * (act_high + act_low)
        actions = tanh_action * action_scale + action_bias

        if deterministic:
            log_prob = torch.zeros(states.size(0), dtype=states.dtype, device=states.device)
        else:
            log_prob = dist.log_prob(pre_tanh)
            # Change-of-variables correction for tanh and action scaling.
            correction = torch.log(action_scale * (1.0 - tanh_action.pow(2)) + 1e-6)
            log_prob = (log_prob - correction).sum(dim=-1)
        return actions, log_prob

    def sample_torch_actions(
        self,
        states: torch.Tensor,
        repeats: int = 1,
        deterministic: bool = False,
        act_low: float = -1.0,
        act_high: float = 1.0,
    ) -> torch.Tensor:
        base = states
        if repeats > 1:
            base = states.repeat_interleave(repeats, dim=0)
        actions, _ = self.sample_with_logprob(base, deterministic=deterministic, act_low=act_low, act_high=act_high)
        return actions


class QCritic(nn.Module):
    def __init__(self, state_dim: int, act_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + act_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, actions], dim=-1)
        return self.net(x)


class ModelReplayBuffer:
    """Simple fixed-size tensor replay buffer for model rollouts."""

    def __init__(self, capacity: int, state_dim: int, act_dim: int, device: torch.device) -> None:
        self.capacity = int(max(1, capacity))
        self.device = device
        self._ptr = 0
        self._size = 0
        self.states = torch.zeros((self.capacity, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((self.capacity, act_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((self.capacity,), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((self.capacity, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((self.capacity,), dtype=torch.float32, device=device)

    @property
    def size(self) -> int:
        return self._size

    @torch.no_grad()
    def add_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        n = int(states.size(0))
        if n <= 0:
            return

        if n >= self.capacity:
            states = states[-self.capacity :]
            actions = actions[-self.capacity :]
            rewards = rewards[-self.capacity :]
            next_states = next_states[-self.capacity :]
            dones = dones[-self.capacity :]
            n = self.capacity

        end = self._ptr + n
        if end <= self.capacity:
            sl = slice(self._ptr, end)
            self.states[sl] = states
            self.actions[sl] = actions
            self.rewards[sl] = rewards
            self.next_states[sl] = next_states
            self.dones[sl] = dones
        else:
            first = self.capacity - self._ptr
            second = n - first
            self.states[self._ptr :] = states[:first]
            self.actions[self._ptr :] = actions[:first]
            self.rewards[self._ptr :] = rewards[:first]
            self.next_states[self._ptr :] = next_states[:first]
            self.dones[self._ptr :] = dones[:first]

            self.states[:second] = states[first:]
            self.actions[:second] = actions[first:]
            self.rewards[:second] = rewards[first:]
            self.next_states[:second] = next_states[first:]
            self.dones[:second] = dones[first:]

        self._ptr = (self._ptr + n) % self.capacity
        self._size = min(self.capacity, self._size + n)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._size <= 0:
            raise ValueError("ModelReplayBuffer is empty.")
        idx = torch.randint(0, self._size, (int(batch_size),), device=self.device)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )


@dataclass
class RomiTrainingInfo:
    pretrain_losses: List[List[float]]
    epoch_dynamics_loss: List[float]
    epoch_weight_loss: List[float]
    epoch_actor_loss: List[float]
    epoch_critic1_loss: List[float]
    epoch_critic2_loss: List[float]
    epoch_alpha_loss: List[float]
    epoch_alpha: List[float]
    model_buffer_size: int
    uncertainty_mean: float
    uncertainty_std: float
    uncertainty_max: float
    ensemble_size: int
    uncertainty_scale: float
    uncertainty_samples: int
    real_data_ratio: float
    rollout_horizon: int
    rollouts_per_epoch: int
    policy_updates_per_epoch: int
    dynamics_updates_per_epoch: int
    weight_updates_per_epoch: int
    batch_size: int
    automatic_entropy_tuning: bool


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


def _sample_real_batch(
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, states.size(0), (int(batch_size),), device=states.device)
    return states[idx], actions[idx], rewards[idx], next_states[idx], dones[idx]


def _soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for src, tgt in zip(source.parameters(), target.parameters()):
            tgt.data.mul_(1.0 - tau).add_(tau * src.data)


def _target_state_values(
    states: torch.Tensor,
    actor: TanhGaussianActor,
    target_q1: QCritic,
    target_q2: QCritic,
    act_low: float,
    act_high: float,
) -> torch.Tensor:
    # Keep the policy fixed for RVL targets; gradient still flows through states via Q.
    with torch.no_grad():
        actions = actor.sample_torch_actions(states, deterministic=True, act_low=act_low, act_high=act_high)
    q1 = target_q1(states, actions).squeeze(-1)
    q2 = target_q2(states, actions).squeeze(-1)
    return torch.minimum(q1, q2)


@torch.no_grad()
def _min_uncertainty_target_values(
    next_states: torch.Tensor,
    actor: TanhGaussianActor,
    target_q1: QCritic,
    target_q2: QCritic,
    uncertainty_scale: float,
    uncertainty_samples: int,
    act_low: float,
    act_high: float,
    state_low: Optional[torch.Tensor] = None,
    state_high: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    bsz, sdim = next_states.shape
    k = int(max(1, uncertainty_samples))
    noise = torch.randn((bsz, k, sdim), dtype=next_states.dtype, device=next_states.device) * float(uncertainty_scale)
    candidates = next_states.unsqueeze(1) + noise
    if state_low is not None and state_high is not None:
        candidates = torch.max(torch.min(candidates, state_high.view(1, 1, -1)), state_low.view(1, 1, -1))

    flat = candidates.reshape(bsz * k, sdim)
    vals = _target_state_values(
        flat, actor=actor, target_q1=target_q1, target_q2=target_q2, act_low=act_low, act_high=act_high
    ).reshape(bsz, k)
    return vals.min(dim=1).values


def robust_value_aware_loss(
    model: RomiDynamicsModel,
    params: Mapping[str, torch.Tensor],
    states: torch.Tensor,
    actions: torch.Tensor,
    next_states: torch.Tensor,
    actor: TanhGaussianActor,
    target_q1: QCritic,
    target_q2: QCritic,
    uncertainty_scale: float,
    uncertainty_samples: int,
    act_low: float,
    act_high: float,
) -> torch.Tensor:
    pred_next_mean, _ = model.predict_distribution_with_params(params, states, actions)
    pred_values = _target_state_values(
        pred_next_mean, actor=actor, target_q1=target_q1, target_q2=target_q2, act_low=act_low, act_high=act_high
    )
    with torch.no_grad():
        min_values = _min_uncertainty_target_values(
            next_states=next_states,
            actor=actor,
            target_q1=target_q1,
            target_q2=target_q2,
            uncertainty_scale=uncertainty_scale,
            uncertainty_samples=uncertainty_samples,
            act_low=act_low,
            act_high=act_high,
            state_low=model.state_low,
            state_high=model.state_high,
        )
    return F.mse_loss(pred_values, min_values)


def _sac_update_step(
    actor: TanhGaussianActor,
    critic1: QCritic,
    critic2: QCritic,
    target_critic1: QCritic,
    target_critic2: QCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic1_optimizer: torch.optim.Optimizer,
    critic2_optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    alpha: float,
    tau: float,
    act_low: float,
    act_high: float,
    log_alpha: Optional[torch.Tensor],
    alpha_optimizer: Optional[torch.optim.Optimizer],
    target_entropy: Optional[float],
) -> Tuple[float, float, float, float]:
    with torch.no_grad():
        next_actions, next_logp = actor.sample_with_logprob(
            next_states, deterministic=False, act_low=act_low, act_high=act_high
        )
        target_q1 = target_critic1(next_states, next_actions).squeeze(-1)
        target_q2 = target_critic2(next_states, next_actions).squeeze(-1)
        target_q = torch.minimum(target_q1, target_q2) - alpha * next_logp
        target = rewards + gamma * (1.0 - dones) * target_q

    q1 = critic1(states, actions).squeeze(-1)
    q2 = critic2(states, actions).squeeze(-1)
    critic1_loss = F.mse_loss(q1, target)
    critic2_loss = F.mse_loss(q2, target)

    critic1_optimizer.zero_grad(set_to_none=True)
    critic1_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic1.parameters(), 10.0)
    critic1_optimizer.step()

    critic2_optimizer.zero_grad(set_to_none=True)
    critic2_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic2.parameters(), 10.0)
    critic2_optimizer.step()

    sampled_actions, logp = actor.sample_with_logprob(states, deterministic=False, act_low=act_low, act_high=act_high)
    q_pi = torch.minimum(
        critic1(states, sampled_actions).squeeze(-1),
        critic2(states, sampled_actions).squeeze(-1),
    )
    actor_loss = (alpha * logp - q_pi).mean()

    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
    actor_optimizer.step()

    alpha_loss_val = 0.0
    if log_alpha is not None and alpha_optimizer is not None and target_entropy is not None:
        alpha_loss = -(log_alpha * (logp + float(target_entropy)).detach()).mean()
        alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        alpha_optimizer.step()
        alpha_loss_val = float(alpha_loss.item())

    _soft_update(critic1, target_critic1, tau=tau)
    _soft_update(critic2, target_critic2, tau=tau)

    return (
        float(actor_loss.item()),
        float(critic1_loss.item()),
        float(critic2_loss.item()),
        float(alpha_loss_val),
    )


@torch.no_grad()
def _append_model_rollouts(
    ensemble: RomiDynamicsEnsemble,
    actor: TanhGaussianActor,
    reward_fn_torch: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    start_states: torch.Tensor,
    buffer: ModelReplayBuffer,
    horizon: int,
    act_low: float,
    act_high: float,
) -> None:
    states = start_states
    rollout_h = int(max(1, horizon))
    for _ in range(rollout_h):
        actions = actor.sample_torch_actions(states, deterministic=False, act_low=act_low, act_high=act_high)
        next_states = ensemble.sample_next(states, actions, deterministic=False)
        rewards = reward_fn_torch(states, actions).reshape(-1).to(torch.float32)
        dones = torch.zeros(states.size(0), dtype=torch.float32, device=states.device)
        buffer.add_batch(states, actions, rewards, next_states, dones)
        states = next_states


def _sample_mixed_batch(
    real_states: torch.Tensor,
    real_actions: torch.Tensor,
    real_rewards: torch.Tensor,
    real_next_states: torch.Tensor,
    real_dones: torch.Tensor,
    model_buffer: ModelReplayBuffer,
    batch_size: int,
    real_data_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if model_buffer.size <= 0:
        return _sample_real_batch(
            real_states, real_actions, real_rewards, real_next_states, real_dones, batch_size=batch_size
        )

    ratio = float(np.clip(real_data_ratio, 0.0, 1.0))
    real_bs = int(round(batch_size * ratio))
    real_bs = max(1, min(batch_size - 1, real_bs))
    model_bs = batch_size - real_bs

    rs, ra, rr, rsn, rd = _sample_real_batch(
        real_states, real_actions, real_rewards, real_next_states, real_dones, batch_size=real_bs
    )
    ms, ma, mr, msn, md = model_buffer.sample(model_bs)
    return (
        torch.cat([rs, ms], dim=0),
        torch.cat([ra, ma], dim=0),
        torch.cat([rr, mr], dim=0),
        torch.cat([rsn, msn], dim=0),
        torch.cat([rd, md], dim=0),
    )


@torch.no_grad()
def _batched_uncertainty(
    ensemble: RomiDynamicsEnsemble,
    states: torch.Tensor,
    actions: torch.Tensor,
    batch_size: int = 8192,
) -> torch.Tensor:
    vals: List[torch.Tensor] = []
    for start in range(0, states.size(0), batch_size):
        sb = states[start : start + batch_size]
        ab = actions[start : start + batch_size]
        vals.append(ensemble.uncertainty(sb, ab).detach().to("cpu"))
    return torch.cat(vals, dim=0)


def train_romi_full(
    dataset: OfflineDataset | Mapping[str, np.ndarray],
    reward_fn_torch: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ensemble_size: Optional[int] = None,
    dynamics_hidden_dim: Optional[int] = None,
    dynamics_hidden_layers: Optional[int] = None,
    weight_hidden_dim: Optional[int] = None,
    weight_hidden_layers: Optional[int] = None,
    actor_hidden_dim: Optional[int] = None,
    critic_hidden_dim: Optional[int] = None,
    model_pretrain_epochs: Optional[int] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    dynamics_lr: Optional[float] = None,
    weight_lr: Optional[float] = None,
    actor_lr: Optional[float] = None,
    critic_lr: Optional[float] = None,
    gamma: Optional[float] = None,
    tau: Optional[float] = None,
    real_data_ratio: Optional[float] = None,
    model_rollout_horizon: Optional[int] = None,
    model_rollouts_per_epoch: Optional[int] = None,
    policy_updates_per_epoch: Optional[int] = None,
    dynamics_updates_per_epoch: Optional[int] = None,
    weight_updates_per_epoch: Optional[int] = None,
    uncertainty_scale: Optional[float] = None,
    uncertainty_samples: Optional[int] = None,
    weight_min: Optional[float] = None,
    weight_max: Optional[float] = None,
    model_buffer_capacity: Optional[int] = None,
    bilevel_inner_lr: Optional[float] = None,
    bootstrap: Optional[bool] = None,
    automatic_entropy_tuning: Optional[bool] = None,
    policy_pretrain_steps: Optional[int] = None,
    act_low: float = -1.0,
    act_high: float = 1.0,
    state_low: Optional[torch.Tensor] = None,
    state_high: Optional[torch.Tensor] = None,
    seed: int = 0,
    env_name: Optional[str] = None,
    device: Optional[torch.device] = None,
    log_hook: Optional[Callable[[str, int, float], None]] = None,
) -> Tuple[RomiDynamicsEnsemble, TanhGaussianActor, RomiTrainingInfo]:
    """Train ROMI with joint dynamics + policy learning.

    Returns:
        (trained_dynamics_ensemble, trained_policy, training_info)
    """
    set_seed(seed)
    defaults = get_romi_defaults(env_name)
    device = device or DEVICE

    ensemble_size = int(defaults.ensemble_size if ensemble_size is None else ensemble_size)
    dynamics_hidden_dim = int(defaults.dynamics_hidden_dim if dynamics_hidden_dim is None else dynamics_hidden_dim)
    dynamics_hidden_layers = int(
        defaults.dynamics_hidden_layers if dynamics_hidden_layers is None else dynamics_hidden_layers
    )
    weight_hidden_dim = int(defaults.weight_hidden_dim if weight_hidden_dim is None else weight_hidden_dim)
    weight_hidden_layers = int(defaults.weight_hidden_layers if weight_hidden_layers is None else weight_hidden_layers)
    actor_hidden_dim = int(defaults.actor_hidden_dim if actor_hidden_dim is None else actor_hidden_dim)
    critic_hidden_dim = int(defaults.critic_hidden_dim if critic_hidden_dim is None else critic_hidden_dim)
    model_pretrain_epochs = int(defaults.model_pretrain_epochs if model_pretrain_epochs is None else model_pretrain_epochs)
    epochs = int(defaults.epochs if epochs is None else epochs)
    batch_size = int(defaults.batch_size if batch_size is None else batch_size)
    dynamics_lr = float(defaults.dynamics_lr if dynamics_lr is None else dynamics_lr)
    weight_lr = float(defaults.weight_lr if weight_lr is None else weight_lr)
    actor_lr = float(defaults.actor_lr if actor_lr is None else actor_lr)
    critic_lr = float(defaults.critic_lr if critic_lr is None else critic_lr)
    gamma = float(defaults.gamma if gamma is None else gamma)
    tau = float(defaults.tau if tau is None else tau)
    real_data_ratio = float(defaults.real_data_ratio if real_data_ratio is None else real_data_ratio)
    model_rollout_horizon = int(
        defaults.model_rollout_horizon if model_rollout_horizon is None else model_rollout_horizon
    )
    model_rollouts_per_epoch = int(
        defaults.model_rollouts_per_epoch if model_rollouts_per_epoch is None else model_rollouts_per_epoch
    )
    policy_updates_per_epoch = int(
        defaults.policy_updates_per_epoch if policy_updates_per_epoch is None else policy_updates_per_epoch
    )
    dynamics_updates_per_epoch = int(
        defaults.dynamics_updates_per_epoch if dynamics_updates_per_epoch is None else dynamics_updates_per_epoch
    )
    weight_updates_per_epoch = int(
        defaults.weight_updates_per_epoch if weight_updates_per_epoch is None else weight_updates_per_epoch
    )
    uncertainty_scale = float(defaults.uncertainty_scale if uncertainty_scale is None else uncertainty_scale)
    uncertainty_samples = int(defaults.uncertainty_samples if uncertainty_samples is None else uncertainty_samples)
    weight_min = float(defaults.weight_min if weight_min is None else weight_min)
    weight_max = float(defaults.weight_max if weight_max is None else weight_max)
    model_buffer_capacity = int(defaults.model_buffer_capacity if model_buffer_capacity is None else model_buffer_capacity)
    bilevel_inner_lr = float(defaults.bilevel_inner_lr if bilevel_inner_lr is None else bilevel_inner_lr)
    bootstrap = bool(defaults.bootstrap if bootstrap is None else bootstrap)
    automatic_entropy_tuning = bool(
        defaults.automatic_entropy_tuning if automatic_entropy_tuning is None else automatic_entropy_tuning
    )
    policy_pretrain_steps = int(defaults.policy_pretrain_steps if policy_pretrain_steps is None else policy_pretrain_steps)

    states, actions, rewards, next_states, dones = _dataset_tensors(dataset, device)
    n_samples = int(states.size(0))
    if n_samples < 2:
        raise ValueError("Dataset is too small for ROMI training.")

    state_dim = int(states.size(-1))
    act_dim = int(actions.size(-1))

    # Build dynamics ensemble + optimizers.
    models: List[RomiDynamicsModel] = []
    dyn_opts: List[torch.optim.Optimizer] = []
    pretrain_losses: List[List[float]] = []

    for member_idx in range(ensemble_size):
        model = RomiDynamicsModel(
            state_dim=state_dim,
            act_dim=act_dim,
            hidden_dim=dynamics_hidden_dim,
            hidden_layers=dynamics_hidden_layers,
            state_low=state_low,
            state_high=state_high,
        ).to(device)

        if bootstrap:
            boot_idx = torch.randint(0, n_samples, (n_samples,), device=device)
        else:
            boot_idx = torch.arange(n_samples, device=device)

        model.fit_normalizer(states[boot_idx], actions[boot_idx], next_states[boot_idx])
        optimizer = torch.optim.Adam(model.parameters(), lr=dynamics_lr)

        losses_member: List[float] = []
        for epoch_idx in range(model_pretrain_epochs):
            idx = torch.randint(0, boot_idx.size(0), (batch_size,), device=device)
            b = boot_idx[idx]
            loss = model.nll_loss(states[b], actions[b], next_states[b]) + model.logvar_regularizer()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            losses_member.append(float(loss.item()))
            if log_hook is not None:
                log_hook(f"romi/pretrain_member_{member_idx}", epoch_idx, float(loss.item()))

        models.append(model)
        dyn_opts.append(optimizer)
        pretrain_losses.append(losses_member)

    ensemble = RomiDynamicsEnsemble(models=models).to(device)

    # Build ROMI weighting network.
    weight_net = AdaptiveWeightNet(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_dim=weight_hidden_dim,
        hidden_layers=weight_hidden_layers,
        weight_min=weight_min,
        weight_max=weight_max,
    ).to(device)
    weight_optimizer = torch.optim.Adam(weight_net.parameters(), lr=weight_lr)

    # Build SAC policy/critics.
    actor = TanhGaussianActor(state_dim=state_dim, act_dim=act_dim, hidden_dim=actor_hidden_dim).to(device)
    critic1 = QCritic(state_dim=state_dim, act_dim=act_dim, hidden_dim=critic_hidden_dim).to(device)
    critic2 = QCritic(state_dim=state_dim, act_dim=act_dim, hidden_dim=critic_hidden_dim).to(device)
    target_critic1 = copy.deepcopy(critic1).to(device)
    target_critic2 = copy.deepcopy(critic2).to(device)
    for p in target_critic1.parameters():
        p.requires_grad_(False)
    for p in target_critic2.parameters():
        p.requires_grad_(False)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic1_optimizer = torch.optim.Adam(critic1.parameters(), lr=critic_lr)
    critic2_optimizer = torch.optim.Adam(critic2.parameters(), lr=critic_lr)

    log_alpha: Optional[torch.Tensor] = None
    alpha_optimizer: Optional[torch.optim.Optimizer] = None
    target_entropy: Optional[float] = None
    if automatic_entropy_tuning:
        log_alpha = torch.tensor(math.log(0.2), dtype=torch.float32, device=device, requires_grad=True)
        alpha_optimizer = torch.optim.Adam([log_alpha], lr=actor_lr)
        target_entropy = -float(act_dim)

    # Optional BC warm start for policy.
    if policy_pretrain_steps > 0:
        for _ in range(policy_pretrain_steps):
            idx = torch.randint(0, n_samples, (batch_size,), device=device)
            pred_actions = actor.sample_torch_actions(
                states[idx], deterministic=True, act_low=act_low, act_high=act_high
            )
            bc_loss = F.mse_loss(pred_actions, actions[idx])
            actor_optimizer.zero_grad(set_to_none=True)
            bc_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
            actor_optimizer.step()

    model_buffer = ModelReplayBuffer(
        capacity=model_buffer_capacity,
        state_dim=state_dim,
        act_dim=act_dim,
        device=device,
    )

    epoch_dynamics_loss: List[float] = []
    epoch_weight_loss: List[float] = []
    epoch_actor_loss: List[float] = []
    epoch_critic1_loss: List[float] = []
    epoch_critic2_loss: List[float] = []
    epoch_alpha_loss: List[float] = []
    epoch_alpha: List[float] = []

    for epoch_idx in range(epochs):
        # ------------------------------------------------------------------
        # 1) Build model buffer with current policy + current model ensemble.
        # ------------------------------------------------------------------
        if model_rollouts_per_epoch > 0:
            start_idx = torch.randint(0, n_samples, (model_rollouts_per_epoch,), device=device)
            _append_model_rollouts(
                ensemble=ensemble,
                actor=actor,
                reward_fn_torch=reward_fn_torch,
                start_states=states[start_idx],
                buffer=model_buffer,
                horizon=model_rollout_horizon,
                act_low=act_low,
                act_high=act_high,
            )

        # ------------------------------------------------------------------
        # 2) Policy optimization (SAC) on mixed real/model batches.
        # ------------------------------------------------------------------
        actor_losses_step: List[float] = []
        critic1_losses_step: List[float] = []
        critic2_losses_step: List[float] = []
        alpha_losses_step: List[float] = []

        for _ in range(policy_updates_per_epoch):
            sb, ab, rb, snb, db = _sample_mixed_batch(
                real_states=states,
                real_actions=actions,
                real_rewards=rewards,
                real_next_states=next_states,
                real_dones=dones,
                model_buffer=model_buffer,
                batch_size=batch_size,
                real_data_ratio=real_data_ratio,
            )

            alpha_value = float(log_alpha.exp().item()) if log_alpha is not None else 0.2
            a_loss, c1_loss, c2_loss, al_loss = _sac_update_step(
                actor=actor,
                critic1=critic1,
                critic2=critic2,
                target_critic1=target_critic1,
                target_critic2=target_critic2,
                actor_optimizer=actor_optimizer,
                critic1_optimizer=critic1_optimizer,
                critic2_optimizer=critic2_optimizer,
                states=sb,
                actions=ab,
                rewards=rb,
                next_states=snb,
                dones=db,
                gamma=gamma,
                alpha=alpha_value,
                tau=tau,
                act_low=act_low,
                act_high=act_high,
                log_alpha=log_alpha,
                alpha_optimizer=alpha_optimizer,
                target_entropy=target_entropy,
            )
            actor_losses_step.append(a_loss)
            critic1_losses_step.append(c1_loss)
            critic2_losses_step.append(c2_loss)
            alpha_losses_step.append(al_loss)

        # ------------------------------------------------------------------
        # 3) Inner dynamics updates: weighted supervised likelihood.
        # ------------------------------------------------------------------
        dyn_losses_step: List[float] = []
        for _ in range(dynamics_updates_per_epoch):
            for member_idx, (model, opt) in enumerate(zip(ensemble.models, dyn_opts)):
                idx = torch.randint(0, n_samples, (batch_size,), device=device)
                sb = states[idx]
                ab = actions[idx]
                snb = next_states[idx]
                with torch.no_grad():
                    weights = weight_net(sb, ab, snb)
                per_sample_nll = model.nll_loss_per_sample(sb, ab, snb)
                dyn_loss = (weights * per_sample_nll).mean() + model.logvar_regularizer()
                opt.zero_grad(set_to_none=True)
                dyn_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                opt.step()
                dyn_losses_step.append(float(dyn_loss.item()))
                if log_hook is not None:
                    log_hook(f"romi/inner_member_{member_idx}", epoch_idx, float(dyn_loss.item()))

        # ------------------------------------------------------------------
        # 4) Outer updates: one-step differentiable bilevel weighting update.
        # ------------------------------------------------------------------
        weight_losses_step: List[float] = []
        for _ in range(weight_updates_per_epoch):
            member_idx = int(torch.randint(0, ensemble_size, (1,), device=device).item())
            model = ensemble.models[member_idx]

            idx = torch.randint(0, n_samples, (batch_size,), device=device)
            sb = states[idx]
            ab = actions[idx]
            snb = next_states[idx]

            weights = weight_net(sb, ab, snb)
            per_sample_nll = model.nll_loss_per_sample(sb, ab, snb)
            inner_loss = (weights * per_sample_nll).mean() + model.logvar_regularizer()

            named_params = list(model.named_parameters())
            grad_list = torch.autograd.grad(
                inner_loss,
                [p for _, p in named_params],
                create_graph=True,
            )
            updated_params: Dict[str, torch.Tensor] = {
                name: p - float(bilevel_inner_lr) * g
                for (name, p), g in zip(named_params, grad_list)
            }

            outer_loss = robust_value_aware_loss(
                model=model,
                params=updated_params,
                states=sb,
                actions=ab,
                next_states=snb,
                actor=actor,
                target_q1=target_critic1,
                target_q2=target_critic2,
                uncertainty_scale=uncertainty_scale,
                uncertainty_samples=uncertainty_samples,
                act_low=act_low,
                act_high=act_high,
            )
            weight_optimizer.zero_grad(set_to_none=True)
            outer_loss.backward()
            torch.nn.utils.clip_grad_norm_(weight_net.parameters(), 10.0)
            weight_optimizer.step()
            weight_losses_step.append(float(outer_loss.item()))

        # Epoch metrics.
        mean_dyn = float(np.mean(dyn_losses_step)) if dyn_losses_step else 0.0
        mean_w = float(np.mean(weight_losses_step)) if weight_losses_step else 0.0
        mean_actor = float(np.mean(actor_losses_step)) if actor_losses_step else 0.0
        mean_c1 = float(np.mean(critic1_losses_step)) if critic1_losses_step else 0.0
        mean_c2 = float(np.mean(critic2_losses_step)) if critic2_losses_step else 0.0
        mean_alpha_loss = float(np.mean(alpha_losses_step)) if alpha_losses_step else 0.0
        alpha_value_epoch = float(log_alpha.exp().item()) if log_alpha is not None else 0.2

        epoch_dynamics_loss.append(mean_dyn)
        epoch_weight_loss.append(mean_w)
        epoch_actor_loss.append(mean_actor)
        epoch_critic1_loss.append(mean_c1)
        epoch_critic2_loss.append(mean_c2)
        epoch_alpha_loss.append(mean_alpha_loss)
        epoch_alpha.append(alpha_value_epoch)

        if log_hook is not None:
            log_hook("romi/epoch_dynamics_loss", epoch_idx, mean_dyn)
            log_hook("romi/epoch_weight_loss", epoch_idx, mean_w)
            log_hook("romi/epoch_actor_loss", epoch_idx, mean_actor)
            log_hook("romi/epoch_critic1_loss", epoch_idx, mean_c1)
            log_hook("romi/epoch_critic2_loss", epoch_idx, mean_c2)
            log_hook("romi/epoch_alpha", epoch_idx, alpha_value_epoch)

    uncertainties = _batched_uncertainty(ensemble, states, actions, batch_size=8192)
    info = RomiTrainingInfo(
        pretrain_losses=pretrain_losses,
        epoch_dynamics_loss=epoch_dynamics_loss,
        epoch_weight_loss=epoch_weight_loss,
        epoch_actor_loss=epoch_actor_loss,
        epoch_critic1_loss=epoch_critic1_loss,
        epoch_critic2_loss=epoch_critic2_loss,
        epoch_alpha_loss=epoch_alpha_loss,
        epoch_alpha=epoch_alpha,
        model_buffer_size=int(model_buffer.size),
        uncertainty_mean=float(uncertainties.mean().item()),
        uncertainty_std=float(uncertainties.std(unbiased=False).item()),
        uncertainty_max=float(uncertainties.max().item()),
        ensemble_size=int(ensemble_size),
        uncertainty_scale=float(uncertainty_scale),
        uncertainty_samples=int(max(1, uncertainty_samples)),
        real_data_ratio=float(np.clip(real_data_ratio, 0.0, 1.0)),
        rollout_horizon=int(max(1, model_rollout_horizon)),
        rollouts_per_epoch=int(max(0, model_rollouts_per_epoch)),
        policy_updates_per_epoch=int(max(0, policy_updates_per_epoch)),
        dynamics_updates_per_epoch=int(max(0, dynamics_updates_per_epoch)),
        weight_updates_per_epoch=int(max(0, weight_updates_per_epoch)),
        batch_size=int(max(1, batch_size)),
        automatic_entropy_tuning=bool(automatic_entropy_tuning),
    )
    return ensemble, actor, info


@torch.no_grad()
def rollout_in_romi_model(
    ensemble: RomiDynamicsEnsemble,
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
    deterministic_dynamics: bool = False,
) -> Tuple[float, float, Dict[str, float]]:
    """Evaluate a fixed policy in ROMI learned dynamics."""
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
    step_count = 0.0

    for _ in range(horizon):
        actions = policy.sample_torch_actions(
            states,
            repeats=1,
            deterministic=deterministic_policy,
            act_low=act_low,
            act_high=act_high,
        )
        rewards = reward_fn_torch(states, actions).reshape(-1).to(torch.float32)
        uncertainty = ensemble.uncertainty(states, actions)
        next_states = ensemble.sample_next(states, actions, deterministic=deterministic_dynamics)

        returns = returns + discount * rewards
        discount *= gamma
        states = next_states

        uncertainty_sum += float(uncertainty.sum().item())
        step_count += float(rollouts)

    mean = float(returns.mean().item())
    stderr = float(returns.std(unbiased=False).item() / np.sqrt(max(1, rollouts)))
    metrics = {
        "uncertainty_mean": float(uncertainty_sum / max(1.0, step_count)),
    }
    return mean, stderr, metrics


@torch.no_grad()
def evaluate_policies_in_romi_model(
    ensemble: RomiDynamicsEnsemble,
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
    deterministic_dynamics: bool = False,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for i, policy in enumerate(policies):
        mean, stderr, metrics = rollout_in_romi_model(
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
