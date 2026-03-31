Thank you for your thoughtful comments and suggestions. We give answers to each of your questions/concerns below.

---

**(A) Connection to decision-focused learning**

We thank the reviewer for pointing us toward the literature on decision-focused learning (DFL). We agree that these are relevant works, and we will extend our Related Work section with a detailed discussion of this literature to better contextualise our contribution.

While the foundational non-MDP methods (Donti et al., 2017; Wilder et al., 2019) do not apply to our sequential setting, we will directly address the MDP extensions you mentioned. Firstly, a fundamental distinction between our work and these DFL MDP works is the core objectives. The DFL MDP papers generally target learning a model *such that the optimal policy within it achieves high ground-truth value*. In contrast, our goal, in similar terms, is to construct a model *such that policies from a candidate set are well ranked within it*. Our objective is more relevant to human-in-the-loop decision-making with DTs. Consequently, the training of these DFL methods is quite different to ours.

As well as this shared distinction in problem setting, the specific works cited have technical constraints that make direct empirical comparison inappropriate or infeasible:
*   **Futoma et al. (2020):** Operates in a POMDP setting, with discrete actions only. Their method is not applicable to the continuous-action environments we consider.
*   **Wang et al. (2021):** Addresses a fundamentally different problem, attempting to predict MDP parameters (transition or reward parameters) from features of the MDP (e.g. external static descriptions of the MDP), rather than learning a simulator from observed state-action trajectory data.
*   **Sharma et al. (2024):** Focuses on reward transfer—learning a model that supports policy optimization when the reward function changes. They only consider simple linear and tabular MDP settings, leaving scalability to expressive model classes and continuous control open.

To our knowledge, $\text{DT}^2$ is the first method to directly optimise an expressive dynamics model for policy ranking amongst a candidate set, bringing the DFL literature to the DT setting. Furthermore, we also focus on maintaining simulation fidelity of the model, for human interpretability purposes, which these DFL MDP works are not generally concerned with—if their models are good in terms of their optimal policy, their objective is satisfied.

**Update:** We will extend our related works to incorporate this discussion of decision-focused literature.

---

**(B) Offline MBRL baselines**

We appreciate the suggestion to compare against offline MBRL. As you noted, offline MBRL also learns a dynamics model from offline trajectory data, and so, although its purpose (using the model to learn an optimal policy) differs from ours (using the model to rank a set of candidate policies), it is possible and worth comparing to empirically. Note that many offline MBRL works attempt to learn so-called 'pessimistic' dynamics, to avoid 'model exploitation', where the optimal policy explores regions that the model cannot accurately predict, and this inductive bias may not be optimal for ranking performance.

We have now run additional experiments comparing $\text{DT}^2$ against prominent offline MBRL methods. Specifically, we compare with the two foundational papers you referenced, **MOReL** (Kidambi et al., 2020) and **MOPO** (Yu et al., 2020), which both train neural ensembles of dynamics models and construct a 'pessimistic MDP', penalising the reward function in transitions where there is high uncertainty. We also compare to a very recent offline MBRL baseline, **ROMI** (Qiao et al., ICLR 2026), which alternates between training an ensemble of neural dynamics models and a SAC policy, and weights the dynamics training using a value-aware approach to incorporate pessimism adversarially, rather than based on uncertainty. Also note that we already included an adapted version of **VaGraM** (Voelcker et al., 2022) in our original submission. Because VaGraM is an *online* MBRL method, we adapted it to be capable offline, and compared with it to see if encoding policy values via value-function gradient weighting (rather than our ranking loss) was beneficial.

The table linked [here](https://anonymous.4open.science/r/DT2-r/mbrl-results.pdf) reports ranking performance of these new baselines and the best performing $\text{DT}^2$ and standard DT architectures, per environment, across the environments from $\S 6.2$. The new baselines are about as competitive as the simulation-loss DTs, and $\text{DT}^2$ outperforms in both regret and Spearman's correlation across almost all environments.


**Update:** We will update $\S 6.2$ to include these additional offline MBRL baselines.

---

Thank you once again. We hope that we have addressed all your comments, and we would greatly appreciate any further feedback.