# Related Works: DT2 vs. MOReL vs. MOPO vs. ROMI

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

## 6) MOPO (Model-Based Offline Policy Optimization)

### Method summary from the MOPO paper
MOPO is also a model-based offline RL method, but unlike MOREL's hard unknown-region termination, it uses a soft uncertainty penalty in reward:

- Define an uncertainty-penalized reward `r_tilde(s,a) = r(s,a) - lambda * u(s,a)`.
- Optimize policy in the corresponding uncertainty-penalized MDP.
- In practice, train an ensemble of probabilistic Gaussian dynamics models and penalize reward with a conservative uncertainty estimate.
- The practical algorithm rolls out model trajectories, writes penalized rewards into model-generated replay, and updates policy with SAC on real + model data.

(See `papers/MOPO.txt`: lines 123-126 for MOPO vs MOReL hard-vs-soft distinction, 313-317 for uncertainty-penalized MDP, 411-440 for practical uncertainty estimator and penalty, and 959-980 for Algorithm 2 rollout/update loop.)

### How MOPO relates to DT2 and MOREL

| Aspect | DT2 | MOREL | MOPO |
|---|---|---|---|
| Core objective | Preserve policy ranking during dynamics training (`L_sim + lambda_rank * L_rank`) | Build pessimistic MDP with hard `HALT` on unknown `(s,a)` | Optimize in uncertainty-penalized MDP with soft reward penalty |
| Uncertainty role | Secondary (stability via truncated rollout + Q bootstrap) | Primary safety gate via thresholded USAD | Primary conservative signal via continuous penalty `lambda * u(s,a)` |
| Treatment of risky regions | No explicit pessimistic transition rule | Immediate absorb/terminate with large negative reward | Allow temporary risky steps but charge penalty; no forced termination |
| Coupling to OPE / ranking | Strong coupling via FQE targets | Not ranking-targeted | Not ranking-targeted; conservative policy optimization |
| Closest method | Distinct objective family | Closest to MOPO among current methods | Closest to MOREL in this repo (same model-based offline + uncertainty framing) |

### How to implement MOPO here (delta from current MOREL path)

`src/morel.py` and `exps/base_pipeline.py` already provide most of the scaffolding for a MOPO branch. The main required changes versus the current MOREL implementation are:

### A) Replace hard-threshold pessimism with soft reward penalty
In `src/morel.py`:

- Remove dependence on `unknown_mask(...)` thresholding (`MorelDynamicsEnsemble.unknown_mask`, line ~192) for MOPO execution.
- Replace `pessimistic_step(...)` (line ~466), which halts trajectories and emits `halt_reward`, with a `mopo_step(...)` that:
  - always transitions using the ensemble model,
  - computes uncertainty `u(s,a)`,
  - sets reward to `r_hat(s,a) - lambda_mopo * u(s,a)` (or `reward_fn_torch(s,a) - lambda_mopo * u(s,a)` for fixed-policy evaluation).
- Replace `rollout_in_pessimistic_mdp(...)` (line ~511) with `rollout_in_mopo_penalized_mdp(...)` that accumulates penalized returns and does not maintain `halted` state.

### B) Update uncertainty estimator to MOPO-style (paper-faithful)
Current MOREL code uses pairwise disagreement of ensemble mean next states (`disagreement(...)`, line ~184). MOPO practical implementation uses uncertainty from probabilistic model variance (max ensemble std norm).

So for paper-faithful MOPO:

- Extend the dynamics member to output probabilistic parameters (including covariance/variance) instead of only deterministic next-state mean (`MorelDynamicsModel`, line ~61).
- Add/retain ensemble training by MLE, then compute `u(s,a)` from predicted variance (max across members), matching MOPO Section 4.3.

### C) Pipeline/CLI branching changes
In `exps/base_pipeline.py`:

- Add `mopo` to `--dynamics-models` choices (near line ~1361).
- Add MOPO hyperparameters (`--mopo-lambda`, optionally MOPO-specific ensemble settings).
- In `train_dynamics_models(...)` (line ~785 onward), add a `mopo` branch parallel to `morel`.
- Add `evaluate_in_mopo_penalized_mdp(...)` parallel to `evaluate_in_morel_pessimistic_mdp(...)` (line ~1059).
- Log MOPO metrics separately (e.g., `eval/mopo_penalized_return`, `eval/mopo_uncertainty_mean`, `eval/mopo_penalty_mean`) alongside existing MOREL logs.

### D) Scope note: fixed-policy evaluation vs full MOPO
For this repository's current scope (policy ranking/evaluation of pre-trained policies), MOPO can be integrated as a conservative evaluation backend exactly like MOREL, but with soft penalized rewards instead of HALT transitions.

For full paper-faithful MOPO, an additional policy-learning stage is required: MBPO-style synthetic rollouts with penalized rewards and SAC updates on `D_env U D_model` (Algorithm 2), which is beyond the current fixed-policy-only MOREL evaluation path.

## 7) ROMI (Robust Value-Aware Model Learning with Implicitly Differentiable Adaptive Weighting)

### Method summary from the ROMI paper (transition-model learning)
ROMI is designed as a replacement for RAMBO-style adversarial model-gradient updates. For transition learning, ROMI combines:

1. A **robust value-aware loss** (`L_RVL`, Eq. 6): make model-predicted next states have values close to the *minimum* value within a state uncertainty set around dataset next states.
2. A **weighted supervised likelihood loss** (`L_WSL`, Eq. 8 inner objective): train dynamics with per-sample weights `w_nu(s,a,s')`.
3. A **bi-level update** (Eq. 9/10):  
   - Inner loop updates dynamics parameters `psi` by minimizing `L_WSL` (dynamics awareness).  
   - Outer loop updates weighting-network parameters `nu` using implicit differentiation through the inner update, to minimize `L_RVL` (value awareness).

The uncertainty set is derived from a Wasserstein dynamics uncertainty set (Section 4.1, Proposition 4.1), then implemented practically by perturbing dataset next states with scale `xi` and taking a sampled min-value surrogate (Eq. 6).

### Concrete appendix algorithm details (what is actually optimized)
Appendix D.2 + Algorithm 1 make the training recipe explicit:

- Dynamics model: probabilistic ensemble `T_psi(s'|s,a)=N(mu_psi, Sigma_psi)`, next-state prediction only (reward treated as known if available from environment/reward function).
- Pretraining: maximum-likelihood/supervised dynamics pretrain for fixed epochs (paper uses 50).
- Main training loop:
  - Roll out short model trajectories; at each step sample an ensemble member.
  - Add model transitions to a model buffer.
  - Update policy with SAC on mixed real/model data (`f` real-data ratio; paper base uses 0.5).
  - Inner update: optimize `L_WSL(psi,nu) = -E[w_nu(s,a,s') log T_psi(s'|s,a)]`.
  - Outer update: compute implicit gradient term (`g_RVL`) from Eq. 10 and update `nu`.
- Weighting network details: MLP over concatenated `(s,a,s')`, output squashed to bounded interval (paper: `[0.5, 2.0]`).
- Core hyperparameters from Appendix Table 3: ensemble size 7, dynamics MLP `(input,200,200,200,200,output)`, weighting MLP `(input,256,256,256,1)`, short rollout horizon `H`.

This is the key practical distinction: ROMI does **not** directly push model parameters with adversarial model gradients; it reweights supervised dynamics updates so that minimizing one-step fit also improves robust value behavior.

### How to implement ROMI in this codebase (paper-faithful training, model-only evaluation)
The current repository already has the right structural pieces (`src/mopo.py` probabilistic ensembles, `src/morel.py` conservative rollouts, `src/networks.py` value-aware training hooks, `exps/base_pipeline.py` multi-branch dynamics orchestration). For your requested protocol, ROMI should be integrated as follows:

### A) Add a dedicated `src/romi.py`
Implement:

- `RomiDynamicsModel` / `RomiDynamicsEnsemble`:
  - Reuse/adapt MOPO-style probabilistic Gaussian next-state modeling (can omit reward head for paper-faithful ROMI when reward is known).
- `AdaptiveWeightNet`:
  - MLP on concatenated `(s,a,s')`, bounded output range `[a,b]`.
- `robust_value_aware_loss(...)`:
  - Implement Eq. 6 with uncertainty-set sampling: perturb `s'` with scale `xi`, sample `N` candidates, take min predicted value.
- `train_romi_full(...)` (paper-faithful):
  - Pretrain dynamics.
  - Run the full Algorithm 1 loop with model rollouts, model-buffer construction, SAC policy updates, and bi-level inner/outer dynamics updates (Eq. 8/9/10).
  - Return `(romi_dynamics, romi_learned_policy, romi_info)`.
- `rollout_in_romi_model(...)`:
  - Evaluate arbitrary fixed policies inside the trained ROMI dynamics model.

### B) Wire ROMI into `exps/base_pipeline.py`
- Extend CLI `--dynamics-models` with `romi`.
- Add ROMI hyperparameters for both dynamics and policy loops (e.g., `--romi-xi`, `--romi-uncertainty-samples`, `--romi-rollout-horizon`, `--romi-weight-lr`, `--romi-weight-min`, `--romi-weight-max`, `--romi-pretrain-epochs`, `--romi-ensemble-size`, plus SAC/model-buffer settings used by ROMI training).
- In `train_dynamics_models(...)`, add a ROMI branch that calls `train_romi_full(...)`.
- Extend `save_dynamics_models(...)` / manifest / loaders with `romi` and `romi_info` (optionally save `romi_learned_policy` for reproducibility, even though it is not used in downstream comparison).

### C) Required evaluation protocol for this project goal
Use the following protocol to match your comparison objective:

1. Train ROMI **as written in the paper** (joint model + policy training).
2. After training, **discard ROMI’s learned policy for evaluation purposes**.
3. Keep only the trained ROMI dynamics model.
4. Unroll your existing fixed policy set in ROMI dynamics and compute value estimates/rankings.
5. Compare these fixed-policy estimates against MOReL and MOPO under matched rollout/evaluation settings.

This isolates whether ROMI’s robustness-oriented *training procedure* improves dynamics quality for your fixed-policy value-estimation task, even when ROMI’s own learned policy is not used.

## 8) DT2 vs. MOReL vs. MOPO vs. ROMI (transition-model perspective)

| Aspect | DT2 | MOReL | MOPO | ROMI |
|---|---|---|---|---|
| Dynamics training signal | `L_sim + lambda_rank * L_rank` (ranking-aware) | Supervised/MLE ensemble fit | Probabilistic ensemble MLE (plus holdout elite selection) | Bi-level: inner weighted MLE (`L_WSL`) + outer robust value-aware loss (`L_RVL`) |
| How conservatism enters | Indirect (ranking consistency via FQE targets) | Hard unknown-region rule (`HALT`, large negative reward) | Soft uncertainty penalty in reward (`-lambda*u`) | Robust min-value target within uncertainty set (`xi` controls strength) |
| Role of Q/value function in dynamics updates | Central for ranking loss targets | Not required for fitting dynamics | Not required for fitting dynamics | Central in outer loss (value-aware objective), but not as DT2 pairwise ranking target |
| Optimization structure | Single-level combined objective | Mostly single-level (fit model, then threshold-based pessimism) | Single-level model fit; conservatism used during policy optimization/eval | Explicit bi-level optimization with implicit differentiation |
| Project evaluation protocol | Train/evaluate dynamics directly on fixed policy set | Train model, evaluate fixed policy set in pessimistic MDP | Train model, evaluate fixed policy set in penalized MDP | Train ROMI with joint model+policy loop, then discard learned policy and evaluate fixed policy set with ROMI dynamics only |
| Main failure mode targeted | Wrong policy ordering despite good one-step prediction | OOD exploitation via uncertain transitions | Over-optimism from model bias, handled via soft penalty | Over-conservative/unstable adversarial model-gradient behavior (RAMBO issue) |
| Closest current module in this repo | `src/networks.py` ranking-aware training | `src/morel.py` | `src/mopo.py` | New `src/romi.py` built by combining MOPO-like probabilistic dynamics + value-aware/bi-level logic |
