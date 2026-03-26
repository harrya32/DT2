# Related Works: DT2 vs. MOREL vs. MOPO vs. POPCORN

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

## 7) POPCORN (Partially Observed Prediction Constrained RL)

### Method summary from the POPCORN paper
POPCORN is designed for **batch off-policy RL under partial observability**. Instead of only maximizing model likelihood (2-stage POMDP fitting) or only optimizing policy value (value-only), it learns a POMDP model that balances both:

- Learn a latent-state IO-HMM/POMDP model (`tau, mu, sigma, R`) from trajectories.
- Solve the current model with PBVI (point-based value iteration) to get policy `pi_theta`.
- Estimate policy value off-policy using CWPDIS under the behavior policy.
- Optimize a prediction-constrained objective:
  - constrained form: maximize `L_gen(theta)` subject to `V(pi_theta) >= epsilon`;
  - practical Lagrangian form: maximize `L_gen(theta) + lambda * V(pi_theta)`.
- In real-data settings, add OPE stabilization:
  - ESS regularization (`-lambda_ESS / ESS(theta)`),
  - support restriction so `pi_theta(a|h)` only uses actions with sufficient behavior-policy support.

Unlike MOReL/MOPO, POPCORN's core pessimism mechanism is **not** uncertainty penalties on rewards/transitions; it is a **joint model-fit + decision-value training objective** with explicit off-policy value control.

(See `papers/popcorn.pdf`: Section 1 for motivation, Section 4.1 for constrained/Lagrangian objective, Section 4.1-4.2 for differentiable PBVI + CWPDIS + ESS/support stabilization, and Section 7 for the high-level conclusion.)

### Algorithm specifics that matter for implementation
Key paper-level mechanics to preserve if you want POPCORN fidelity:

- **Latent POMDP parameterization** (`tau, mu, sigma, R`) with IO-HMM likelihood and forward-backward filtering.
- **Planner in the loop**: PBVI is part of training, not just post-hoc evaluation.
- **Differentiable PBVI relaxation**:
  - relax PBVI argmax operations to softmax (temperature-controlled),
  - use stochastic policies for more stable OPE support.
- **Continuous-observation PBVI approximation**:
  - cluster observations into meta-observations via sampled observation partitions and alpha-vector assignments.
- **Training loop engineering from appendix**:
  - cache PBVI value function/belief sets across gradient updates,
  - do a few backups per update (instead of full PBVI each step),
  - occasional hard reset of belief/value caches.
- **Reward-learning guardrail**:
  - do not backprop OPE value through reward parameters directly; fit reward separately (EM-style update) to avoid reward inflation artifacts.

(See `papers/popcorn.pdf`: Section 3 for model/planning setup, Appendix A.2-A.4 for cached-PBVI + meta-observation approximation + softmax PBVI relaxation, and Appendix B for separate reward fitting.)

### How POPCORN compares to DT2, MOREL, and MOPO

| Aspect | DT2 | MOREL | MOPO | POPCORN |
|---|---|---|---|---|
| Primary setting | Offline MDP policy ranking | Offline MDP conservative policy optimization/eval | Offline MDP conservative policy optimization | Offline **POMDP** policy learning in batch off-policy data |
| Core objective | `L_sim + lambda_rank * L_rank` | Model fit + hard pessimistic HALT MDP | Model fit + soft uncertainty-penalized rewards | `L_gen + lambda * V_OPE(pi_theta)` (or constrained `V >= epsilon`) |
| Decision signal during model training | Pairwise policy ordering proxies (FQE) | No direct value term in dynamics fitting | No direct value term in dynamics fitting | Direct off-policy value of solved policy is in training objective |
| Uncertainty usage | Secondary/stability-related | Central (USAD threshold + HALT) | Central (continuous uncertainty penalty) | OPE variance/safety control via ESS + support overlap, not HALT/uncertainty penalty |
| Partial observability handling | Not explicit | Not explicit | Not explicit | First-class (latent-state POMDP + belief-state planning) |

### How to implement POPCORN in this codebase
Because POPCORN is structurally different from MOReL/MOPO, a clean integration is a **new module** plus a new pipeline branch.

### A) Add a dedicated POPCORN module
Create `src/popcorn.py` with:

- `PopcornPOMDPModel`:
  - latent transition/emission/reward parameters (`tau, mu, sigma, R`),
  - forward filtering / sequence log-likelihood.
- `soft_pbvi(...)`:
  - differentiable PBVI backups (temperature softmax),
  - support for continuous observations using sampled meta-observations.
- `cwpdis_value(...)` + `effective_sample_size(...)`:
  - off-policy value and ESS computations from trajectory-level data.
- `train_popcorn(...)`:
  - objective `L_gen + lambda * V - lambda_ess / ESS`,
  - support restriction using `pi_beh`,
  - optional cached value-function/belief updates and periodic resets.

### B) Extend base pipeline entry points
In `exps/base_pipeline.py`:

- Add `popcorn` to `--dynamics-models` choices (currently includes `supervised`, `kendall`, `hinge`, `listnet`, `morel`, `mopo`; see lines 1537-1542).
- Add POPCORN hyperparameters, e.g.:
  - `--popcorn-num-latent-states`,
  - `--popcorn-lambda`,
  - `--popcorn-lambda-ess`,
  - `--popcorn-support-delta`,
  - `--popcorn-pbvi-temp`,
  - `--popcorn-pbvi-backups-per-step`.
- Branch inside `train_dynamics_models(...)` (line ~869 onward) to call `train_popcorn(...)`, parallel to existing MOReL/MOPO branches (lines 1042-1113).

### C) Save/load + evaluation plumbing
Follow existing MOReL/MOPO patterns:

- Extend manifest save/load in `save_dynamics_models(...)` / `load_dynamics_models(...)` (lines 323-406) with `popcorn` and `popcorn_info`.
- Add `evaluate_in_popcorn_*` helper(s) alongside `evaluate_in_morel_pessimistic_mdp(...)` and `evaluate_in_mopo_penalized_mdp(...)` (lines 1195-1250), depending on whether you evaluate:
  - learned `pi_theta` from PBVI directly, or
  - fixed candidate policies under POPCORN-trained model.
- Add logging in the evaluation/logging block (lines 2000-2126), e.g.:
  - `eval/popcorn_cwpdis_value`,
  - `eval/popcorn_ess`,
  - `eval/popcorn_loglik`,
  - `eval/popcorn_support_overlap`.

### D) Data/trajectory requirements (important)
POPCORN needs trajectory-level histories and behavior-policy probabilities for CWPDIS. Current `OfflineDataset` use in this repo is often transition-centric for MDP methods, so POPCORN integration likely requires:

- preserving sequence structure (`o_{0:t}, a_{0:t}`),
- behavior-policy estimation module (or known `pi_beh`) to compute importance ratios,
- policy support-masking utilities.

### E) Best-fit scope in this repository
POPCORN is most natural for the partially observed workflows (`exps/pomdp_pipeline.py` and clinical/sepsis-style trajectories), rather than fully observed MuJoCo-style branches where MOReL/MOPO are currently aligned.

For a lighter-weight first step in `exps/base_pipeline.py`, you can implement a **POPCORN-inspired** objective (`model-fit + OPE-valued policy term`) without full latent-state PBVI. But full paper-faithful POPCORN requires the POMDP planner-in-the-loop design above.

## 8) Decision-Focused RL for Reward Transfer (RDF-MBRL)

### Method summary from the reward-transfer paper
This paper proposes **Robust Decision-Focused (RDF) model learning** for settings where reward preferences can change between training and deployment. The reward is parameterized as a weighted combination of reward bases, and preferences are represented by a weight vector `w`.

Core objective (paper Eq. 8):

- learn model parameters `theta` that do well on average over deployment-time preferences `w ~ P(w)`,
- while keeping performance high on the known learning-phase preference `w_bar`.

Constrained form:

- minimize `E_{w~P(w)} [ J_{T*, R_w}(theta) ]`,
- subject to `J_{T*, R_{w_bar}}(pi*(theta, R_{w_bar})) >= delta`.

They optimize a Lagrangian relaxation (Eq. 12), approximate the expectation over `P(w)` with a finite grid `W` (Eq. 13), and compute gradients through the planning step with implicit differentiation (Eq. 14-15). Their Algorithm 1 is:

1. Build a preference grid `W` over deployment preferences.
2. For each `w in {w_bar} U W`, plan `pi*(theta, R_w)` under the current model.
3. Evaluate returns on the true environment for each planned policy.
4. Update `theta` using the averaged deployment objective plus learning-phase constraint term.

So, unlike plain DF, RDF is explicitly optimized for **reward-transfer robustness**.

### How RDF differs from current methods in this repo

| Aspect | DT2 | MOREL | MOPO | POPCORN | RDF (this paper) |
|---|---|---|---|---|---|
| Main training signal | `L_sim + lambda_rank * L_rank` on fixed candidate policies | Model fit + hard pessimism (HALT) | Model fit + soft uncertainty penalty | `L_gen + lambda * V_OPE` in latent POMDP | Decision-focused value objective averaged over reward preferences, with learning-task constraint |
| Reward-shift handling | Indirect via ranking supervision for one target reward setup | Not reward-transfer targeted | Not reward-transfer targeted | Handles changing objectives but in POMDP/OPE-planner setting | Explicitly targets transfer across `P(w)` |
| Uses uncertainty pessimism | No (primary signal is ranking) | Yes | Yes | OPE variance/support controls | No explicit pessimistic uncertainty mechanism |
| Planner in training loop | No (fixed policy set + FQE proxies) | Not required for fixed-policy eval branch | Not required for fixed-policy eval branch | Yes (PBVI) | Yes (differentiate through `pi*(theta, R_w)`) |
| Best match to DT2 question (policy ranking after model learning) | Native | Usable conservative evaluator | Usable conservative evaluator | Heavy mismatch for MDP ranking | Usable, but needs adaptation for fixed-policy ranking in this codebase |

### How to implement RDF in this codebase
There are two practical options:

### A) Paper-faithful RDF branch (harder)
Add `src/rdf.py` with:

- `build_preference_grid(...)`,
- `train_rdf_model(...)` implementing Eq. 12/13/15,
- planner adapters (tabular VI / LQR where differentiable policy gradients w.r.t. model parameters are available),
- optional implicit-diff utilities.

Then in `exps/base_pipeline.py`:

- add `rdf` to `--dynamics-models`,
- add RDF args (`--rdf-lambda`, `--rdf-delta`, `--rdf-w-min`, `--rdf-w-max`, `--rdf-num-w`),
- branch in `train_dynamics_models(...)` to call `train_rdf_model(...)`,
- extend save/load (`save_dynamics_models(...)`, `load_dynamics_models(...)`) with `rdf` + `rdf_info`,
- add evaluation keys (e.g. `eval/rdf_return`, `eval/rdf_avg_over_w`).

Important mismatch: paper-faithful RDF evaluates planned policies on the true simulator during learning. For fully offline branches, you would need an OPE proxy in place of true-environment returns.

### B) RDF-style fixed-policy variant (recommended for your DT2 comparison)
Because your main evaluation is ranking a **pre-defined set of policies**, implement an RDF-inspired objective that avoids differentiating through a planner:

- keep current dynamics parameterization (`DynamicsNet`),
- for each training step, evaluate candidate policies in the learned model across a preference grid `W`,
- optimize average deployment preference performance plus a learning-preference constraint/penalty term,
- continue to include simulation fit loss (`nll`/`mse`) for stability.

This gives a reward-transfer-aware model-learning baseline that is directly aligned with your fixed-policy ranking protocol.

### Can RDF be used as a baseline for DT2 in your ranking setup?
Yes, with a caveat.

- **Yes**: once the RDF (or RDF-style) transition model is learned, you can score each pre-defined policy by model rollouts and rank by predicted return (exactly like other dynamics evaluators in this repo).
- **Caveat**: paper-faithful RDF is planner-centric, not fixed-policy-ranking-centric. For fair DT2 comparison on policy ranking, use the fixed-policy RDF-style adaptation above (or clearly label the baseline as "RDF-planner-transfer" vs "DT2-ranking-transfer").

In short: RDF is a strong, conceptually relevant reward-transfer baseline, but for your exact ranking task it should be adapted to fixed-policy evaluation to avoid an apples-to-oranges comparison.
