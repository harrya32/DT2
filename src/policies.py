from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


class GaussianLinearPolicy:
    """Linear-Gaussian policy with optional cached torch parameters."""

    def __init__(self, W: np.ndarray, std: float, name: str):
        self.W = np.asarray(W, dtype=np.float32)
        self.std = float(std)
        self.name = name
        self._torch_cache: Dict[torch.device, Tuple[torch.Tensor, torch.Tensor]] = {}

    def mean_action(self, state: np.ndarray) -> np.ndarray:
        mu = self.W @ state
        return np.clip(mu, -1.0, 1.0)

    def sample(self, state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mu = self.mean_action(state)
        return np.clip(rng.normal(mu, self.std), -1.0, 1.0)

    def log_prob(self, actions: np.ndarray, states: np.ndarray) -> np.ndarray:
        mu = np.einsum("ij,bj->bi", self.W, states)
        var = self.std ** 2
        return -0.5 * np.sum(np.log(2.0 * np.pi * var) + ((actions - mu) ** 2) / var, axis=1)

    def torch_log_prob(self, actions: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        W_t, std_t = self._get_torch_params(actions.device)
        mu = states @ W_t.t()
        var = std_t * std_t
        log_term = (actions - mu) ** 2 / var + torch.log(2.0 * torch.pi * var)
        return -0.5 * torch.sum(log_term, dim=-1)

    def _get_torch_params(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if device not in self._torch_cache:
            W_t = torch.tensor(self.W, dtype=torch.float32, device=device)
            std_t = torch.tensor(self.std, dtype=torch.float32, device=device)
            self._torch_cache[device] = (W_t, std_t)
        return self._torch_cache[device]

    def __repr__(self) -> str:  # pragma: no cover - for debugging convenience only
        return f"GaussianLinearPolicy(name={self.name}, std={self.std})"
