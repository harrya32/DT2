from __future__ import annotations

import math
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .datasets import OfflineDataset
from .policies import GaussianLinearPolicy, TorchPolicy
from .utils import DEVICE


class DynamicsNet(nn.Module):
    def __init__(
        self,
        state_dim: int = 8,
        act_dim: int = 1,
        hidden: int = 128,
        state_low: Optional[torch.Tensor] = None,
        state_upper: Optional[torch.Tensor] = None,
        wrapped_dims: Optional[Sequence[int]] = None,
    ) -> None:

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, state_dim)
        self.logvar_head = nn.Linear(hidden, state_dim)

        if state_low is None:
            state_low = torch.tensor(
                [
                    -2.5,
                    -2.5,
                    -10.0,
                    -10.0,
                    -math.pi,
                    -10.0,
                    0.0,
                    0.0,
                ],
                dtype=torch.float32,
            )
        else:
            state_low = state_low.detach().to(dtype=torch.float32)

        if state_upper is None:
            state_upper = torch.tensor(
                [
                    2.5,
                    2.5,
                    10.0,
                    10.0,
                    math.pi,
                    10.0,
                    1.0,
                    1.0,
                ],
                dtype=torch.float32,
            )
        else:
            state_upper = state_upper.detach().to(dtype=torch.float32)

        if state_low.shape[0] != state_dim or state_upper.shape[0] != state_dim:
            raise ValueError("State bounds must match state_dim length.")
        if torch.any(state_upper <= state_low):
            raise ValueError("Each state_upper entry must exceed state_low.")

        self.register_buffer("state_low", state_low)
        self.register_buffer("state_high", state_upper)
        default_wrapped = [4] if state_dim >= 5 else []
        dims = wrapped_dims if wrapped_dims is not None else default_wrapped
        invalid = [d for d in dims if d < 0 or d >= state_dim]
        if invalid:
            raise ValueError(f"wrapped_dims indices out of range: {invalid}")
        self.wrapped_dims = tuple(int(d) for d in sorted(set(dims)))

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        x = torch.cat([s, a], dim=-1)
        h = self.net(x)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        return mean, logvar

    def nll(self, s: torch.Tensor, a: torch.Tensor, s_next: torch.Tensor):
        mean, logvar = self.forward(s, a)
        inv_var = torch.exp(-logvar)
        nll = 0.5 * (logvar + (s_next - mean) ** 2 * inv_var + math.log(2 * math.pi))
        return nll.sum(dim=-1).mean()

    def mse(self, s: torch.Tensor, a: torch.Tensor, s_next: torch.Tensor):
        mean, _ = self.forward(s, a)
        return F.mse_loss(mean, s_next)

    def balanced_loss(self, s: torch.Tensor, a: torch.Tensor, s_next: torch.Tensor):
        mean, logvar = self.forward(s, a)
        diff = self._circular_diff(s_next, mean, s_next - mean)

        # Compute variance-normalized MSE (acts like heteroskedastic NLL)
        inv_var = torch.exp(-logvar)
        nll = 0.5 * (logvar + diff**2 * inv_var + math.log(2 * math.pi))

        # Weight by inverse range^2
        weights = 1.0 / (self.state_high - self.state_low).pow(2)
        nll = (weights * nll).sum(dim=-1)
        return nll.mean()
    
    def _dataset_tensors(
        self, dataset: OfflineDataset | Mapping[str, np.ndarray], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(dataset, OfflineDataset):
            data = dataset.as_dict()
        else:
            data = dataset
        states = torch.tensor(data["s"], dtype=torch.float32, device=device)
        actions = torch.tensor(data["a"], dtype=torch.float32, device=device)
        next_states = torch.tensor(data["s_next"], dtype=torch.float32, device=device)
        return states, actions, next_states
    
    def _as_dict(self, dataset: OfflineDataset | Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        return dataset.as_dict() if isinstance(dataset, OfflineDataset) else dataset
    
    def _sample_policy_actions(
        self,
        policy: TorchPolicy | GaussianLinearPolicy,
        states: torch.Tensor,
        repeats: int,
        act_low: float,
        act_high: float,
        deterministic: bool = False,
    ) -> torch.Tensor:
        if isinstance(policy, GaussianLinearPolicy):
            return policy.sample_torch_actions(
                states,
                repeats=repeats,
                deterministic=deterministic,
                act_low=act_low,
                act_high=act_high,
            )
        if hasattr(policy, "sample_torch_actions"):
            return policy.sample_torch_actions(
                states,
                repeats=repeats,
                deterministic=deterministic,
                act_low=act_low,
                act_high=act_high,
            )
        raise TypeError("Policy must implement sample_torch_actions for dynamics training.")

    def _wrap_state_inplace(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.wrapped_dims:
            return tensor
        for dim in self.wrapped_dims:
            tensor[..., dim] = (tensor[..., dim] + math.pi) % (2 * math.pi) - math.pi
        return tensor

    def _circular_diff(
        self, target: torch.Tensor, pred: torch.Tensor, base_diff: torch.Tensor
    ) -> torch.Tensor:
        if not self.wrapped_dims:
            return base_diff
        diff = base_diff.clone()
        for dim in self.wrapped_dims:
            diff[..., dim] = torch.atan2(
                torch.sin(target[..., dim] - pred[..., dim]),
                torch.cos(target[..., dim] - pred[..., dim]),
            )
        return diff

    def _select_dynamics_loss(self, name: str) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        name = name.lower()
        losses = {
            "nll": self.nll,
            "mse": self.mse,
            "balanced": self.balanced_loss,
        }
        if name not in losses:
            raise ValueError("Unknown dynamics_loss '%s'." % name)
        return losses[name]

    @staticmethod
    def _num_batches(num_samples: int, batch_size: int) -> int:
        return max(1, (num_samples + batch_size - 1) // batch_size)
                    
    def train(
        self,
        dataset: OfflineDataset | Mapping[str, np.ndarray],
        epochs: int = 200,
        batch_size: int = 1024,
        lr: float = 1e-3,
        device: Optional[torch.device] = None,
        log_hook: Optional[Callable[[int, float], None]] = None,
    ) -> DynamicsNet:
        
        device = device or DEVICE
        self.to(device)
        states, actions, next_states = self._dataset_tensors(dataset, device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        loader = DataLoader(
            TensorDataset(states, actions, next_states),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for sb, ab, snb in loader:
                loss = self.balanced_loss(sb, ab, snb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(1, len(loader))
            losses.append(avg_loss)
            if log_hook is not None:
                log_hook(epoch, avg_loss)
        
        return losses
    
    def train_q_aware_model(
        self, 
        dataset: OfflineDataset | Mapping[str, np.ndarray],
        target_policy: TorchPolicy | GaussianLinearPolicy,
        q_fn: nn.Module,
        gamma: float = 0.97,
        lambda_td: float = 1.0,
        epochs: int = 20,
        batch_size: int = 1024,
        lr: float = 5e-4,
        use_amp: bool = True,
        act_low: float = -1.0,
        act_high: float = 1.0,
        samples: int = 4,
        hidden: int = 128,
        dynamics_loss: str = "balanced",
        reward_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        device: Optional[torch.device] = None,
        log_hook: Optional[Callable[[int, float], None]] = None,
    ) -> DynamicsNet:
        
        if reward_fn is None:
            raise ValueError("reward_fn must be provided for q-aware training.")

        _ = hidden  # maintained for API compatibility

        device = device or DEVICE
        self.to(device)
        data = self._as_dict(dataset)
        states = torch.tensor(data["s"], dtype=torch.float32, device=device)
        actions = torch.tensor(data["a"], dtype=torch.float32, device=device)
        next_states = torch.tensor(data["s_next"], dtype=torch.float32, device=device)
        N = states.shape[0]

        q_fn = q_fn.to(device)
        for param in q_fn.parameters():
            param.requires_grad_(False)

        dyn_loss_fn = self._select_dynamics_loss(dynamics_loss)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        use_amp = use_amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
        indices = torch.arange(N, device=device)

        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            perm = indices[torch.randperm(N, device=device)]
            for start in range(0, N, batch_size):
                idx = perm[start : start + batch_size]
                sb, ab, snb = states[idx], actions[idx], next_states[idx]

                with torch.amp.autocast("cuda", enabled=use_amp):
                    dyn_loss = dyn_loss_fn(sb, ab, snb)

                    a_pi = self._sample_policy_actions(target_policy, sb, 1, act_low, act_high)
                    s_next_model = self.sample_next(sb, a_pi)
                    reward = reward_fn(sb, a_pi)

                    if samples > 1:
                        s_rep = s_next_model.repeat_interleave(samples, dim=0)
                        a_rep = self._sample_policy_actions(target_policy, s_next_model, samples, act_low, act_high)
                        q_next = q_fn(s_rep, a_rep).view(s_next_model.size(0), samples).mean(dim=1)
                    else:
                        a_rep = self._sample_policy_actions(target_policy, s_next_model, 1, act_low, act_high)
                        q_next = q_fn(s_next_model, a_rep)

                    q_curr = q_fn(sb, a_pi)
                    td_loss = (q_curr - (reward + gamma * q_next)).pow(2).mean()
                    loss = (1.0 - lambda_td) * dyn_loss + lambda_td * td_loss

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                epoch_loss += loss.item()

            avg_loss = epoch_loss / self._num_batches(N, batch_size)
            losses.append(avg_loss)
            if log_hook is not None:
                log_hook(epoch, avg_loss)

        return losses

    def train_ranking_aware_model(
        self,
        dataset: OfflineDataset | Mapping[str, np.ndarray],
        policy_q_pairs: Sequence[Tuple[TorchPolicy | GaussianLinearPolicy, nn.Module]],
        gamma: float = 0.97,
        lambda_rank: float = 0.1,
        rollout_horizon: int = 50,
        rollout_episodes: int = 32,
        epochs: int = 20,
        batch_size: int = 1024,
        lr: float = 5e-4,
        use_amp: bool = True,
        act_low: float = -1.0,
        act_high: float = 1.0,
        hidden: int = 128,
        dynamics_loss: str = "balanced",
        reward_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        device: Optional[torch.device] = None,
        log_hook: Optional[Callable[[int, float], None]] = None,
    ) -> DynamicsNet:
        
        if reward_fn is None:
            raise ValueError("reward_fn must be provided for ranking-aware training.")

        from .fqe import estimate_V_from_Q_on_s0

        _ = hidden  # maintained for API compatibility

        device = device or DEVICE
        self.to(device)
        data = self._as_dict(dataset)
        states = torch.tensor(data["s"], dtype=torch.float32, device=device)
        actions = torch.tensor(data["a"], dtype=torch.float32, device=device)
        next_states = torch.tensor(data["s_next"], dtype=torch.float32, device=device)
        N = states.shape[0]

        dyn_loss_fn = self._select_dynamics_loss(dynamics_loss)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        use_amp = use_amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
        indices = torch.arange(N, device=device)

        for policy, q in policy_q_pairs:
            q = q.to(device)
            for param in q.parameters():
                param.requires_grad_(False)

        s0 = torch.tensor(data["s0"], dtype=torch.float32, device=device)

        def rollout_return(pi: TorchPolicy | GaussianLinearPolicy) -> torch.Tensor:
            idx = torch.randint(0, s0.size(0), (rollout_episodes,), device=device)
            s = s0[idx]
            total = torch.zeros(rollout_episodes, device=device)
            discount = torch.ones(rollout_episodes, device=device)
            for _ in range(rollout_horizon):
                a = self._sample_policy_actions(pi, s, 1, act_low, act_high)
                s_next = self.sample_next(s, a)
                r = reward_fn(s, a)
                total = total + discount * r
                discount = discount * gamma
                s = s_next
            return total.mean()

        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            perm = indices[torch.randperm(N, device=device)]
            for start in range(0, N, batch_size):
                idx = perm[start : start + batch_size]
                sb, ab, snb = states[idx], actions[idx], next_states[idx]

                with torch.amp.autocast("cuda", enabled=use_amp):
                    dyn_loss = dyn_loss_fn(sb, ab, snb)
                    loss = (1.0 - lambda_rank) * dyn_loss

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                epoch_loss += loss.item()

            if lambda_rank > 0.0:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    target_vals = []
                    model_vals = []
                    for pi, q in policy_q_pairs:
                        target_val = estimate_V_from_Q_on_s0(q, data["s0"], pi, K=32)
                        target_vals.append(target_val)
                        model_vals.append(rollout_return(pi))

                    model_tensor = torch.stack(model_vals)
                    target_tensor = torch.tensor(target_vals, dtype=model_tensor.dtype, device=device)

                    terms = []
                    for i in range(len(policy_q_pairs)):
                        for j in range(i + 1, len(policy_q_pairs)):
                            sign = torch.sign(target_tensor[i] - target_tensor[j])
                            diff = (model_tensor[i] - model_tensor[j]) * sign
                            terms.append(F.relu(-diff))
                    rank_loss = torch.stack(terms).mean() if terms else torch.zeros((), device=device)

                rank_term = lambda_rank * rank_loss
                if use_amp:
                    scaler.scale(rank_term).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    rank_term.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                epoch_loss += rank_term.item()

            avg_loss = epoch_loss / self._num_batches(N, batch_size)
            losses.append(avg_loss)
            if log_hook is not None:
                log_hook(epoch, avg_loss)
        
        return losses


    def sample_next(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        mean, logvar = self.forward(s, a)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        s_next = mean + std * eps

        # --- Handle angle wrapping for configured dims ---
        s_next = self._wrap_state_inplace(s_next)

        # --- Clamp remaining dims per state bounds ---
        s_next = torch.max(torch.min(s_next, self.state_high), self.state_low)
        return s_next

class RescaledQ(nn.Module):
    """Wrap a normalized Q network and rescale predictions."""

    def __init__(self, base: nn.Module, reward_mean: float, reward_std: float, gamma: float):
        super().__init__()
        self.base = base
        self.reward_mean = reward_mean
        self.reward_std = reward_std
        self.gamma = gamma

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        scale = self.reward_std + 1e-8
        offset = self.reward_mean / (1.0 - self.gamma)
        return self.base(states, actions) * scale + offset

class QNet(nn.Module):
    def __init__(self, state_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([states, actions], dim=-1)).squeeze(-1)
    
    def _dataset_to_tensors(self, dataset: OfflineDataset | Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
        if isinstance(dataset, OfflineDataset):
            arrays = dataset.as_dict()
        else:
            arrays = dataset
        return {
            "s": torch.tensor(arrays["s"], dtype=torch.float32, device=device),
            "a": torch.tensor(arrays["a"], dtype=torch.float32, device=device),
            "r": torch.tensor(arrays["r"], dtype=torch.float32, device=device),
            "s_next": torch.tensor(arrays["s_next"], dtype=torch.float32, device=device),
            "done": torch.tensor(arrays["done"], dtype=torch.float32, device=device),
            "s0": torch.tensor(arrays["s0"], dtype=torch.float32, device=device),
        }
    
    def train(
        self,
        dataset: OfflineDataset | Dict[str, np.ndarray],
        target_policy: TorchPolicy | GaussianLinearPolicy = None,
        gamma: float = 0.97,
        epochs: int = 200,
        batch_size: int = 1024,
        lr: float = 3e-4,
        samples: int = 16,
        act_low: float = -1.0,
        act_high: float = 1.0,
        hidden: int = 128,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
        log_hook: Optional[Callable[[int, float], None]] = None,
    ) -> RescaledQ:
        
        device = device or DEVICE
        self.to(device)
        tensors = self._dataset_to_tensors(dataset, device)
        states, actions, rewards = tensors["s"], tensors["a"], tensors["r"]
        next_states, dones = tensors["s_next"], tensors["done"]

        reward_mean = rewards.mean().item()
        reward_std = rewards.std().item()
        rewards_norm = (rewards - reward_mean) / (reward_std + 1e-6)

        target_q = QNet(state_dim=states.shape[1], act_dim=actions.shape[1], hidden=hidden).to(device)
        target_q.load_state_dict(self.state_dict())
        for param in target_q.parameters():
            param.requires_grad_(False)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        use_amp = use_amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
        indices = torch.arange(states.shape[0], device=device)

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            perm = indices[torch.randperm(states.shape[0], device=device)]
            for start in range(0, states.shape[0], batch_size):
                batch_idx = perm[start : start + batch_size]
                sb = states[batch_idx]
                ab = actions[batch_idx]
                rb = rewards_norm[batch_idx]
                snb = next_states[batch_idx]
                db = dones[batch_idx]

                with torch.no_grad():
                    actions_pi = target_policy.sample_torch_actions(
                        snb,
                        repeats=samples,
                        deterministic=False,
                        act_low=act_low,
                        act_high=act_high,
                    )
                    snb_rep = snb.repeat_interleave(samples, dim=0)
                    q_next = target_q(snb_rep, actions_pi).view(snb.size(0), samples).mean(dim=1)
                    target_values = rb + gamma * (1.0 - db) * q_next

                optimizer.zero_grad(set_to_none=True)

                if use_amp:
                    with torch.amp.autocast("cuda", enabled=True):
                        preds = self.forward(sb, ab)
                        loss = F.mse_loss(preds, target_values)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    preds = self.forward(sb, ab)
                    loss = F.mse_loss(preds, target_values)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
                    optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            if (epoch + 1) % 5 == 0:
                target_q.load_state_dict(self.state_dict())

            if log_hook is not None:
                avg_loss = epoch_loss / max(1, num_batches)
                log_hook(epoch, avg_loss)

        return RescaledQ(self, reward_mean, reward_std, gamma)

