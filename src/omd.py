from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # torch>=2.0
    from torch.func import functional_call
except Exception:  # pragma: no cover
    from torch.nn.utils.stateless import functional_call

from .datasets import OfflineDataset
from .networks import DynamicsNet
from .policies import GaussianLinearPolicy, TorchPolicy
from .utils import DEVICE, set_seed


class OmdCritic(nn.Module):
    """Q-network used by OMD inner-loop updates."""

    def __init__(self, state_dim: int, act_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + act_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, actions], dim=-1)
        return self.net(x).squeeze(-1)


@dataclass
class _CriticBundle:
    q1: OmdCritic
    q2: OmdCritic
    target_q1: OmdCritic
    target_q2: OmdCritic


@dataclass
class OmdTrainingInfo:
    inner_losses: List[float]
    outer_losses: List[float]
    val_outer_losses: List[Optional[float]]
    outer_steps: int
    inner_updates: int
    action_samples: int
    inner_lr: float
    model_lr: float
    target_tau: float
    num_policies: int


def _as_dict(dataset: OfflineDataset | Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    return dataset.as_dict() if isinstance(dataset, OfflineDataset) else dataset


def _flatten_with_optional_mask(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if tensor.dim() == 3:
        flat = tensor.reshape(-1, tensor.size(-1))
        if mask is None:
            return flat
        keep = mask.reshape(-1) > 0.5
        return flat[keep]

    if tensor.dim() == 2:
        if mask is None:
            return tensor
        if mask.dim() == 2:
            keep = mask.reshape(-1) > 0.5
        else:
            keep = mask > 0.5
        if keep.numel() == tensor.size(0):
            return tensor[keep]
    return tensor


def _flatten_scalar_with_optional_mask(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if tensor.dim() == 2:
        flat = tensor.reshape(-1)
        if mask is None:
            return flat
        keep = mask.reshape(-1) > 0.5
        return flat[keep]

    if tensor.dim() == 1:
        if mask is None:
            return tensor
        if mask.dim() == 2:
            keep = mask.reshape(-1) > 0.5
        else:
            keep = mask > 0.5
        if keep.numel() == tensor.size(0):
            return tensor[keep]
    return tensor


def _dataset_tensors(
    dataset: OfflineDataset | Mapping[str, np.ndarray],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = _as_dict(dataset)
    mask_np = data.get("mask")
    mask = None
    if mask_np is not None:
        mask = torch.tensor(mask_np, dtype=torch.float32, device=device)

    states = torch.tensor(data["s"], dtype=torch.float32, device=device)
    actions = torch.tensor(data["a"], dtype=torch.float32, device=device)
    rewards = torch.tensor(data["r"], dtype=torch.float32, device=device)
    next_states = torch.tensor(data["s_next"], dtype=torch.float32, device=device)
    dones = torch.tensor(data["done"], dtype=torch.float32, device=device)

    states = _flatten_with_optional_mask(states, mask)
    actions = _flatten_with_optional_mask(actions, mask)
    next_states = _flatten_with_optional_mask(next_states, mask)
    rewards = _flatten_scalar_with_optional_mask(rewards, mask)
    dones = _flatten_scalar_with_optional_mask(dones, mask)

    if states.size(0) == 0:
        raise ValueError("Dataset has zero valid transitions after applying mask.")

    return states, actions, rewards, next_states, dones


def _sample_policy_actions(
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
    raise TypeError("Policy must implement sample_torch_actions for OMD training.")


def _target_q(
    bundle: _CriticBundle,
    policy: TorchPolicy | GaussianLinearPolicy,
    next_states: torch.Tensor,
    action_samples: int,
    act_low: float,
    act_high: float,
) -> torch.Tensor:
    batch_size = next_states.size(0)
    actions = _sample_policy_actions(
        policy,
        next_states,
        repeats=action_samples,
        deterministic=False,
        act_low=act_low,
        act_high=act_high,
    )
    rep_states = next_states.repeat_interleave(action_samples, dim=0)

    q1 = bundle.target_q1(rep_states, actions)
    q2 = bundle.target_q2(rep_states, actions)
    q = torch.minimum(q1, q2).reshape(batch_size, action_samples).mean(dim=1)
    return q


def _critic_td_loss_with_params(
    bundle: _CriticBundle,
    params_q1: Mapping[str, torch.Tensor],
    params_q2: Mapping[str, torch.Tensor],
    policy: TorchPolicy | GaussianLinearPolicy,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    action_samples: int,
    act_low: float,
    act_high: float,
) -> torch.Tensor:
    target_q = _target_q(
        bundle=bundle,
        policy=policy,
        next_states=next_states,
        action_samples=action_samples,
        act_low=act_low,
        act_high=act_high,
    )
    targets = rewards + gamma * (1.0 - dones) * target_q

    q1_pred = functional_call(bundle.q1, params_q1, (states, actions))
    q2_pred = functional_call(bundle.q2, params_q2, (states, actions))

    return F.mse_loss(q1_pred, targets) + F.mse_loss(q2_pred, targets)


def _maybe_warm_start_from_q(
    critic: OmdCritic,
    source_q: Optional[nn.Module],
) -> None:
    if source_q is None:
        return
    src = getattr(source_q, "base", source_q)
    if not isinstance(src, nn.Module):
        return
    try:
        critic.load_state_dict(src.state_dict(), strict=False)
    except Exception:
        # Warm-start is optional; ignore incompatible architectures.
        return


def _soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p_src, p_tgt in zip(source.parameters(), target.parameters()):
            p_tgt.copy_(p_tgt * (1.0 - tau) + p_src * tau)


def train_omd_dynamics(
    dataset: OfflineDataset | Mapping[str, np.ndarray],
    policies: Mapping[str, TorchPolicy | GaussianLinearPolicy],
    init_q_models: Optional[Mapping[str, nn.Module]] = None,
    gamma: float = 0.99,
    outer_steps: int = 2000,
    batch_size: int = 1024,
    model_lr: float = 3e-4,
    inner_lr: float = 1e-3,
    inner_updates: int = 1,
    action_samples: int = 8,
    dynamics_hidden_dim: int = 256,
    critic_hidden_dim: int = 256,
    backbone: str = "mlp",
    reward_fn_torch: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    act_low: float = -1.0,
    act_high: float = 1.0,
    state_low: Optional[torch.Tensor] = None,
    state_high: Optional[torch.Tensor] = None,
    wrapped_dims: Optional[Sequence[int]] = None,
    target_tau: float = 0.01,
    val_fraction: float = 0.1,
    grad_clip: float = 10.0,
    seed: int = 0,
    device: Optional[torch.device] = None,
    log_hook: Optional[Callable[..., None]] = None,
) -> Tuple[DynamicsNet, OmdTrainingInfo]:
    """Train a dynamics model with OMD-style bilevel optimization on offline data.

    Practical approximations mirror the OMD function-approximation recipe:
    - K inner Q updates per outer model step.
    - Implicit term approximation via differentiable inner updates.
    - Warm-started Q parameters across outer steps.
    """
    if not policies:
        raise ValueError("At least one policy is required for OMD training.")

    set_seed(seed)
    device = device or DEVICE

    states, actions, rewards, next_states, dones = _dataset_tensors(dataset, device)
    num_samples = states.size(0)

    if reward_fn_torch is not None:
        with torch.no_grad():
            rewards = reward_fn_torch(states, actions).to(torch.float32)

    val_fraction = float(np.clip(val_fraction, 0.0, 0.9))
    if num_samples > 1 and val_fraction > 0.0:
        val_count = max(1, int(num_samples * val_fraction))
        val_count = min(num_samples - 1, val_count)
        perm = torch.randperm(num_samples, device=device)
        val_idx = perm[:val_count]
        train_idx = perm[val_count:]
    else:
        train_idx = torch.arange(num_samples, device=device)
        val_idx = None

    train_states = states[train_idx]
    train_actions = actions[train_idx]
    train_rewards = rewards[train_idx]
    train_next_states = next_states[train_idx]
    train_dones = dones[train_idx]

    val_tensors = None
    if val_idx is not None:
        val_tensors = (
            states[val_idx],
            actions[val_idx],
            rewards[val_idx],
            next_states[val_idx],
            dones[val_idx],
        )

    state_dim = int(train_states.shape[-1])
    act_dim = int(train_actions.shape[-1])

    dynamics = DynamicsNet(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden=dynamics_hidden_dim,
        state_low=state_low,
        state_upper=state_high,
        wrapped_dims=wrapped_dims,
        backbone=backbone,
    ).to(device)

    dynamics.fit_normalizer(train_states, train_actions, train_next_states)
    model_optimizer = torch.optim.AdamW(dynamics.parameters(), lr=float(model_lr))

    policy_order = list(policies.keys())
    critic_bundles: Dict[str, _CriticBundle] = {}
    for name in policy_order:
        q1 = OmdCritic(state_dim=state_dim, act_dim=act_dim, hidden=critic_hidden_dim).to(device)
        q2 = OmdCritic(state_dim=state_dim, act_dim=act_dim, hidden=critic_hidden_dim).to(device)

        if init_q_models is not None:
            _maybe_warm_start_from_q(q1, init_q_models.get(name))
            _maybe_warm_start_from_q(q2, init_q_models.get(name))

        target_q1 = copy.deepcopy(q1).to(device)
        target_q2 = copy.deepcopy(q2).to(device)
        for p in target_q1.parameters():
            p.requires_grad_(False)
        for p in target_q2.parameters():
            p.requires_grad_(False)

        critic_bundles[name] = _CriticBundle(q1=q1, q2=q2, target_q1=target_q1, target_q2=target_q2)

    inner_losses: List[float] = []
    outer_losses: List[float] = []
    val_outer_losses: List[Optional[float]] = []

    batch_size = int(max(1, batch_size))
    inner_updates = int(max(1, inner_updates))
    action_samples = int(max(1, action_samples))
    outer_steps = int(max(1, outer_steps))

    for step in range(outer_steps):
        # Virtual parameters for differentiable inner-loop updates.
        virtual_q1: Dict[str, Dict[str, torch.Tensor]] = {}
        virtual_q2: Dict[str, Dict[str, torch.Tensor]] = {}
        for name in policy_order:
            bundle = critic_bundles[name]
            virtual_q1[name] = {
                k: p.detach().requires_grad_(True) for k, p in bundle.q1.named_parameters()
            }
            virtual_q2[name] = {
                k: p.detach().requires_grad_(True) for k, p in bundle.q2.named_parameters()
            }

        last_inner_loss = 0.0
        for _ in range(inner_updates):
            idx = torch.randint(0, train_states.size(0), (batch_size,), device=device)
            sb = train_states[idx]
            ab = train_actions[idx]
            rb = train_rewards[idx]
            db = train_dones[idx]

            sn_model = dynamics.sample_next(sb, ab, deterministic=True)

            inner_total = torch.zeros((), device=device)
            flat_entries: List[Tuple[str, str, str, torch.Tensor]] = []

            for name in policy_order:
                bundle = critic_bundles[name]
                loss_i = _critic_td_loss_with_params(
                    bundle=bundle,
                    params_q1=virtual_q1[name],
                    params_q2=virtual_q2[name],
                    policy=policies[name],
                    states=sb,
                    actions=ab,
                    rewards=rb,
                    next_states=sn_model,
                    dones=db,
                    gamma=gamma,
                    action_samples=action_samples,
                    act_low=act_low,
                    act_high=act_high,
                )
                inner_total = inner_total + loss_i

                for p_name, tensor in virtual_q1[name].items():
                    flat_entries.append((name, "q1", p_name, tensor))
                for p_name, tensor in virtual_q2[name].items():
                    flat_entries.append((name, "q2", p_name, tensor))

            inner_total = inner_total / float(len(policy_order))
            grads = torch.autograd.grad(
                inner_total,
                [entry[3] for entry in flat_entries],
                create_graph=True,
                allow_unused=True,
            )

            next_virtual_q1 = {n: {} for n in policy_order}
            next_virtual_q2 = {n: {} for n in policy_order}
            for (name, which, p_name, tensor), grad in zip(flat_entries, grads):
                g = torch.zeros_like(tensor) if grad is None else grad
                updated = tensor - float(inner_lr) * g
                if which == "q1":
                    next_virtual_q1[name][p_name] = updated
                else:
                    next_virtual_q2[name][p_name] = updated

            virtual_q1 = next_virtual_q1
            virtual_q2 = next_virtual_q2
            last_inner_loss = float(inner_total.detach().item())

        idx_outer = torch.randint(0, train_states.size(0), (batch_size,), device=device)
        sb = train_states[idx_outer]
        ab = train_actions[idx_outer]
        rb = train_rewards[idx_outer]
        snb = train_next_states[idx_outer]
        db = train_dones[idx_outer]

        outer_total = torch.zeros((), device=device)
        for name in policy_order:
            bundle = critic_bundles[name]
            loss_i = _critic_td_loss_with_params(
                bundle=bundle,
                params_q1=virtual_q1[name],
                params_q2=virtual_q2[name],
                policy=policies[name],
                states=sb,
                actions=ab,
                rewards=rb,
                next_states=snb,
                dones=db,
                gamma=gamma,
                action_samples=action_samples,
                act_low=act_low,
                act_high=act_high,
            )
            outer_total = outer_total + loss_i

        outer_loss = outer_total / float(len(policy_order))

        model_optimizer.zero_grad(set_to_none=True)
        outer_loss.backward()
        torch.nn.utils.clip_grad_norm_(dynamics.parameters(), max_norm=float(grad_clip))
        model_optimizer.step()

        # Warm-start critics for the next outer step using the final virtual params.
        with torch.no_grad():
            for name in policy_order:
                bundle = critic_bundles[name]
                for p_name, p in bundle.q1.named_parameters():
                    p.copy_(virtual_q1[name][p_name].detach())
                for p_name, p in bundle.q2.named_parameters():
                    p.copy_(virtual_q2[name][p_name].detach())

                _soft_update(bundle.q1, bundle.target_q1, tau=float(target_tau))
                _soft_update(bundle.q2, bundle.target_q2, tau=float(target_tau))

        val_loss_value: Optional[float] = None
        if val_tensors is not None:
            vs, va, vr, vns, vd = val_tensors
            vidx = torch.randint(0, vs.size(0), (min(batch_size, vs.size(0)),), device=device)
            vsb = vs[vidx]
            vab = va[vidx]
            vrb = vr[vidx]
            vnb = vns[vidx]
            vdb = vd[vidx]
            with torch.no_grad():
                val_total = torch.zeros((), device=device)
                for name in policy_order:
                    bundle = critic_bundles[name]
                    q1_params = {k: p for k, p in bundle.q1.named_parameters()}
                    q2_params = {k: p for k, p in bundle.q2.named_parameters()}
                    val_total = val_total + _critic_td_loss_with_params(
                        bundle=bundle,
                        params_q1=q1_params,
                        params_q2=q2_params,
                        policy=policies[name],
                        states=vsb,
                        actions=vab,
                        rewards=vrb,
                        next_states=vnb,
                        dones=vdb,
                        gamma=gamma,
                        action_samples=action_samples,
                        act_low=act_low,
                        act_high=act_high,
                    )
                val_loss_value = float((val_total / float(len(policy_order))).item())

        inner_losses.append(last_inner_loss)
        outer_losses.append(float(outer_loss.detach().item()))
        val_outer_losses.append(val_loss_value)

        if log_hook is not None:
            log_hook(step, outer_losses[-1], val_loss_value)

    info = OmdTrainingInfo(
        inner_losses=inner_losses,
        outer_losses=outer_losses,
        val_outer_losses=val_outer_losses,
        outer_steps=outer_steps,
        inner_updates=inner_updates,
        action_samples=action_samples,
        inner_lr=float(inner_lr),
        model_lr=float(model_lr),
        target_tau=float(target_tau),
        num_policies=len(policy_order),
    )
    return dynamics, info
