# DT2: Decision-Targeted Digital Twins

Code for ICML submission.

---

## Repository Structure

```
DT2/
├── README.md                 # This file
├── environment.yml           # Conda env
├── setup.py                  # Package installation
│
├── src/                      # Core library modules
│   ├── __init__.py
│   ├── datasets.py           # OfflineDataset class and data collection utilities
│   ├── dynamics.py           # Dynamics model training
│   ├── env_utils.py          # Environment utilities and reward functions
│   ├── fqe.py                # Fitted Q Evaluation and Value Function Evaluation
│   ├── networks.py           # Neural network architectures
│   ├── ope_methods.py        # High-level OPE method orchestration
│   ├── policies.py           # Policy classes
│   ├── utils.py              # General utilities (seeding, device management)
│   └── value_aware.py        # Value-aware and ranking-aware model training
│
├── exps/                     # Experiment scripts
│   ├── base_pipeline.py            # Shared experimental fns for continuous controle exps
│   ├── <ENV>_runner.py             # Environment specific exp details
│   └── hypothesis_space_<EXP>      # Limited hypothesis space experiments
│
├── results/                  # Experiment results and outputs
└── scripts/                  # Scripts to reproduce paper experiments using files in exps/ dir
```