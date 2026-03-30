Thank you for your thoughtful comments and suggestions. We give answers to each of your questions/concerns below.

---

**(A) Connection to decision-focused learning**

We thank the reviewer for pointing us toward the rich literature on decision-focused learning. We agree that these are highly relevant works, and we will extend our Related Work section to incorporate a detailed discussion of this stream of literature to better contextualize our contribution.

While the foundational non-MDP methods (Donti et al., 2017; Wilder et al., 2019) do not apply to our sequential trajectory setting, we directly address the MDP extensions. The fundamental distinction between $\text{DT}^2$ and the decision-focused MDP literature is our core objective. The DFL MDP literature broadly targets learning a model such that a policy optimized within it achieves high ground-truth value. In contrast, our goal is explicitly to *rank a fixed, known set of candidate policies*—the primary use case for human-in-the-loop Digital Twins. Consequently, DFL methods are not designed to allocate model capacity toward distinguishing between the values of a given candidate set. Furthermore, DFL methods typically assume access to ground-truth task losses during training, whereas we circumvent the lack of ground-truth rankings in offline settings by uniquely utilizing an OPE proxy (FQE) to guide the ranking loss.

Beyond this shared distinction in problem setting, the specific works cited have technical constraints that make direct empirical comparison inappropriate for our continuous-control setups:
*   **Futoma et al. (2020):** Operates specifically in a POMDP setting with discrete actions only. While they share our design choice of combining a decision-focused loss and a simulation loss via a hyperparameter, their method is not applicable to the continuous-action environments we consider.
*   **Wang et al. (2021):** Addresses a fundamentally different problem. They predict MDP parameters (transition or reward functions) from *features* that describe the MDP, rather than learning a simulator from observed state-action trajectory data.
*   **Sharma et al. (2024):** Focuses on reward transfer—learning a model that supports policy optimization when the reward function changes. They only demonstrate their approach on simple linear or tabular MDP settings, leaving scalability to expressive model classes and continuous control open.

To our knowledge, $\text{DT}^2$ is the first method to directly optimize an expressive generative dynamics model for pairwise policy ranking in the offline, continuous-action setting, while simultaneously maintaining simulation fidelity for human interpretability.

**Update:** We will extend our related works to incorporate this discussion of decision-focused literature.

---

**(B) Comparison with offlime MBRL baselines**

We appreciate the suggestion to compare against offline MBRL baselines. As you noted, while offline MBRL operates on offline data, its purpose (learning a single optimal policy via pessimism/uncertainty penalties) differs significantly from ours (evaluating/ranking a diverse set of candidate policies). Pessimism often distorts the learned dynamics in out-of-distribution regions, which actively harms the ability to accurately rank multiple distinct policies. 

To empirically validate this, we have run additional experiments comparing $\text{DT}^2$ against prominent offline MBRL models. Specifically, we compare to the dynamics models of the two papers you references, **MOReL** (Kidambi et al., 2020) and **MOPO** (Yu et al., 2020), as well as very recent offline MBRL baseline, **ROMI** (Qiao et al., ICLR 2026). Note that we already included an adapted version of **VaGraM** (Voelcker et al., 2022) in our original submission; because VaGraM is an online value-aware MBRL method, we adapted it to see if encoding pre-defined policy values via value-gradients (rather than our ranking loss) could benefit model training.

The table HERE(link) reports the **Regret (std) / Spearman's Rank (std)**. As shown, $\text{DT}^2$ significantly outperforms the offline MBRL models in both regret and ranking correlation across almost all environments.


**Update:** We will update our empirical section to include these additional offline MBRL baselines.

---

Thank you once again. We hope that we have addressed all your comments, and we would greatly appreciate any further feedback. 

---

[1] ...