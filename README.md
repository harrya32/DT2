# DT2: Offline Policy Evaluation (OPE) Research

Code for ICML submission.

## Overview

This repository implements various **Offline Policy Evaluation (OPE)** methods for reinforcement learning. The primary goal is to estimate the expected return of a target policy using only offline (pre-collected) data, without requiring additional environment interaction.

### Key Features

- **Model-based Monte Carlo estimation** — Train a dynamics model and perform rollouts
- **Fitted Q Evaluation (FQE)** — Learn Q-functions via temporal difference methods
- **Value Function Evaluation** — Direct state-value function learning with importance sampling
- **Value-Aware Model Learning** — Train dynamics models with value-function-aware objectives
- **Q-Aware Model Learning** — Dynamics models optimized for Q-function consistency
- **Ranking-Aware Model Learning** — Models that preserve policy ranking relationships

The codebase supports continuous action spaces and is primarily tested on the **LunarLanderContinuous-v3** environment, with additional experiments on healthcare domains (Sepsis treatment).

---

## Repository Structure

```
DT2/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation
│
├── src/                      # Core library modules
│   ├── __init__.py
│   ├── datasets.py           # OfflineDataset class and data collection utilities
│   ├── dynamics.py           # Dynamics model training (supervised learning)
│   ├── env_utils.py          # Environment utilities and reward functions
│   ├── fqe.py                # Fitted Q Evaluation and Value Function Evaluation
│   ├── networks.py           # Neural network architectures (ValueNet, QNet, DynamicsNet)
│   ├── ope_methods.py        # High-level OPE method orchestration
│   ├── policies.py           # Policy classes (GaussianLinearPolicy)
│   ├── utils.py              # General utilities (seeding, device management)
│   └── value_aware.py        # Value-aware and ranking-aware model training
│
├── exps/                     # Experiment scripts
│   ├── offline_ope_refactored.py   # Main OPE experiment runner
│   ├── offline_ope_poc.py          # Proof-of-concept OPE experiments
│   ├── generate_offline_data.py    # Data generation utilities
│   ├── sepsis.py                   # Sepsis environment experiments
│   ├── train_ppo_sepsis.py         # PPO training for Sepsis
│   ├── ope_test.py                 # OPE testing scripts
│   ├── scope_rl_test.py            # SCOPE-RL integration tests
│   ├── dmt_autoregressive_toy.py   # Autoregressive model experiments
│   ├── test.py                     # General testing
│   ├── scope_rl.ipynb              # Interactive SCOPE-RL notebook
│   ├── simple_env.ipynb            # Simple environment exploration
│   └── d3rlpy_logs/                # Experiment logs (d3rlpy format)
│
├── data/                     # Datasets (offline trajectories)
├── results/                  # Experiment results and outputs
└── runs/                     # Training run artifacts (PPO, etc.)
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/harrya32/DT2.git
cd DT2

# Install dependencies
pip install -e .
```

---

## Usage

### Running OPE Experiments

The main experiment script supports multiple OPE methods:

```bash
python exps/offline_ope_refactored.py \
    --methods model value qvalue value-aware q-aware ranking-aware ground-truth \
    --seed 1 \
    --num-seeds 5 \
    --gamma 0.97
```

### Available Methods

| Method | Description |
|--------|-------------|
| `model` | Naive model-based Monte Carlo with supervised dynamics |
| `value` | Fitted Value Evaluation (state-value function) |
| `qvalue` | Fitted Q Evaluation (action-value function) |
| `value-aware` | Value-aware dynamics model training |
| `q-aware` | Q-function-aware dynamics model training |
| `ranking-aware` | Ranking-preserving model training |
| `ground-truth` | Online policy evaluation (requires env access) |

---

## Key Components

### Policies
- **GaussianLinearPolicy**: Linear policy with Gaussian noise for continuous actions

### Networks
- **ValueNet**: MLP for state-value function approximation
- **QNet**: MLP for action-value function approximation  
- **DynamicsNet**: Probabilistic dynamics model (mean + log-variance outputs)

### Evaluation
- Importance sampling corrections for off-policy evaluation
- Reward normalization for stable training
- Support for custom reward functions

---

## License

See LICENSE file for details.

## TODO         

make models be environment agnostic (i.e. have bounds as inputs, fixed scaling, clipping, etc.)