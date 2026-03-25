# Related Works: DT2 vs. MOREL

## 1) DT2 (Decision-Targeted Digital Twins)

### Method summary from the DT2 paper
DT2 starts from a standard dynamics objective (`L_sim`, e.g. NLL/MSE), then adds a decision-targeted ranking loss (`L_rank`) so the learned model preserves policy ordering rather than only one-step transition fidelity. In the paper this is:

- `L_DT2 = (1 - lambda) * L_sim + lambda * L_rank`.
- `L_rank` is a differentiable Kendall-style objective over pairwise policy value differences.
- Ground-truth policy values are unavailable offline, so pairwise targets are proxied with FQE values.
- To stabilize gradients, DT2 uses truncated model rollouts and bootstraps with a frozen Q-function at horizon `H`.

(See `papers/DT2_ICML_2026.txt`, around lines 172-243 for Sections 4.1-4.3 and Eq. 12/16/17.)

### How DT2-style dynamics learning is implemented in this repository
The code mirrors the paper closely:

- Dynamics model and losses: `src/networks.py`
- `DynamicsNet` with configurable backbones (`mlp`, `resnet`, `ode`, `transformer`, `gru`): lines 149-178.
- Gaussian next-state head (`mean`, `logvar`) with delta prediction + normalization + logvar clamp: lines 217-236.
- Simulation losses `nll`, `mse`, and `balanced`: lines 238-270.
- Data normalization (`state`, `action`, `delta`) fit once on train split: lines 364-397.

- Standard supervised dynamics training: `src/networks.py`
- `DynamicsNet.train(...)` optimizes one-step simulation loss with batching, validation split, early stopping, and gradient clipping: lines 434-532.

- Decision-targeted/ranking-aware training (DT2 core): `src/networks.py`
- `DynamicsNet.train_ranking_aware_model(...)`: lines 534-840.
- Builds FQE-based proxy targets with `estimate_V_from_Q_on_s0(...)`: lines 615-621.
- Computes model-implied policy values via rollout + bootstrap from Q at final step: lines 623-670.
- Ranking losses (`kendall`, `hinge`, `listnet`): lines 679-713.
- Uses combined objective `dyn_loss` plus `lambda_rank * rank_loss` (with per-batch dynamics updates and per-epoch ranking update): lines 725-769.

- FQE proxy estimation: `src/networks.py` + `src/fqe.py`
- `QNet.train(...)` learns per-policy Q with target network updates and clipped gradients: `src/networks.py` lines 1164-1253.
- Value extraction on initial states: `src/fqe.py` lines 78-85.

- Pipeline orchestration: `exps/base_pipeline.py`
- Trains per-policy Q models: lines 652-682.
- Trains supervised + ranking-aware dynamics variants: lines 689-826.
- Selects model family via CLI (`--dynamics-models supervised kendall hinge listnet`): lines 1165-1173.
- Integrates dynamics training/loading in pipeline step 4: lines 1395-1450.

## 2) MOREL (Model-Based Offline RL)

### Method summary from the MOREL paper
MOREL also learns a dynamics model from offline data (typically via maximum likelihood), but the key idea is pessimism under uncertainty rather than ranking alignment:

1. Learn approximate dynamics `P_hat` from offline data.
2. Build an unknown-state-action detector (USAD) that marks `(s,a)` as unknown when model reliability cannot be guaranteed.
3. Construct a pessimistic MDP (P-MDP): if unknown is hit, transition to an absorbing `HALT` state and incur large negative reward `-kappa`.
4. Plan a policy in this pessimistic model.

In practical implementation, MOREL uses Gaussian delta dynamics with normalized inputs and an ensemble of models; ensemble disagreement defines uncertainty/unknown regions.

(See `papers/morel.txt`, lines 182-253 for Algorithm 1 + P-MDP definition, and lines 383-412 for practical Gaussian ensemble + discrepancy-threshold USAD.)

For the implementation scope in this repository, we stop before MOREL's policy-learning step and only use the learned pessimistic MDP for fixed-policy rollout evaluation/ranking.

## 3) How DT2 and MOREL differ in dynamics-model training

| Aspect | DT2 | MOREL |
|---|---|---|
| Primary training signal | Hybrid: one-step simulation + policy-ranking consistency | One-step model likelihood (simulation fidelity), then uncertainty-based pessimism at planning time |
| Use of OPE/Q-functions | Central: FQE values supervise ranking targets used during dynamics training | Not used to rank policies during dynamics fitting |
| Objective coupling to policy quality | Direct: dynamics is optimized to preserve policy orderings | Indirect: dynamics used with conservative penalties to avoid unknown regions |
| Uncertainty handling | Not the main control signal; stability via truncated rollout + bootstrapped Q | Core mechanism: ensemble discrepancy defines unknown regions and pessimism |
| Output use-case | Decision support / ranking candidate policies (digital-twin framing) | Conservative control via pessimistic MDP (in this repo: conservative fixed-policy evaluation) |
| Failure mode addressed | Good one-step fit but wrong policy ranking | Model exploitation in OOD/poorly covered regions |

## 4) Where to implement MOREL in this codebase (fixed-policy evaluation only)

A clean integration path is:

### A) Add a dedicated MOREL module
Create `src/morel.py` containing:

- `train_dynamics_ensemble(...)`: train `N` `DynamicsNet` members on bootstrapped/minibatch-shuffled data.
- `ensemble_disagreement(s, a)`: compute max pairwise mean prediction distance across members.
- `usad_mask(s, a, threshold)`: known/unknown decision.
- `pessimistic_step(...)` (or `PessimisticModelEnv` step):
  - if unknown, transition to `HALT` and reward `-kappa`;
  - else transition via sampled/mean ensemble dynamics and environment reward.
- `rollout_in_pessimistic_mdp(policy, s0, horizon, gamma, ...)`:
  - unroll pre-defined policies in the pessimistic MDP and return value estimates.

`DynamicsNet` already has almost all primitives needed (Gaussian delta model, normalization, sampling), so this avoids touching too much existing DT2 code.

### B) Extend shared pipeline entry points
In `exps/base_pipeline.py`:

- CLI/API:
  - Extend `--dynamics-models` choices to include `morel`.
  - Add MOREL hyperparameters: e.g. `--morel-ensemble-size`, `--morel-unknown-threshold`, `--morel-kappa`.

- Training step:
  - Branch inside `train_dynamics_models(...)` (lines 689-826) to call `src.morel.train_dynamics_ensemble(...)` when `morel` is selected.

- Save/load:
  - Extend `save_dynamics_models(...)` / `load_dynamics_models(...)` (lines 321-384) with a `morel` manifest entry for ensemble checkpoints + threshold/kappa config.

- Step-4 orchestration:
  - Wire this branch where dynamics are trained/loaded in `run_pipeline(...)` (lines 1395-1450).

### C) Fixed-policy pessimistic rollout integration (no planner)
In `exps/base_pipeline.py` evaluation utilities (near `evaluate_in_dynamics(...)`), add:

- `evaluate_in_morel_pessimistic_mdp(...)` that:
  - samples `s0` from the dataset,
  - rolls out each pre-defined policy in the pessimistic MDP transition/reward model,
  - applies `HALT` transitions and `-kappa` penalty on unknown `(s,a)`,
  - returns discounted value estimates for ranking/comparison.

No policy optimization/planning call is needed.

### D) Evaluation and logging
In `exps/base_pipeline.py` evaluation section (after line ~1458):

- Add MOREL policy evaluation alongside current `q_est`, supervised dynamics, and ranking models.
- Log metrics under a separate namespace (e.g., `eval/morel_pessimistic_return`, `eval/morel_unknown_rate`).

## 5) Practical note on fit with current repository
Current DT2 code is optimized for ranking candidate policies using learned dynamics and FQE proxies. In your requested scope, MOREL is used as a conservative dynamics backend: train ensemble dynamics + define pessimistic MDP + unroll the existing fixed policy batch inside that pessimistic model for value comparison. No planner/policy-learning component is required.
