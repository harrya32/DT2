Thank you for your thoughtful comments and suggestions. We give answers to each of your questions/concerns below.

---

**(A) Computational cost**

We agree that analyzing the computational trade-offs of $DT^2$ is important. Computational overhead compared to standard simulation-loss training is due to the necessary training of the FQE value functions and the $H$-step unrolling to obtain model-estimated policy values.

Concretely, for the continuous control experiments, training $DT^2$ took up to 30 minutes, while standard DTs took up to 10 minutes.

However, it is worth noting that $DT^2$ training overhead is comparable to other decision-aware modeling paradigms. For instance, the VaGraM baseline we have compared to also requires the training of value models for each policy $\pi \in \Pi$.

Furthermore, we now compare with further offline model-based RL baselines (please see point (B) of our response to reviewer `frev` for details). These new models also take longer than our simulation-loss baselines. Specifically, MOReL and MOPO involve training ensembles of dynamics models, and ROMI requires alternating between training a dynamics ensemble and a SAC policy. These models took up to 15, 42, and 90 minutes respectively on the continuous control environments.

**Update:** We will expand upon the computational costs of each method in our discussion section.

---

**(B) Interpretability**

To concretely compare interpretability between OPE methods and DTs is difficult, because their outputs are very different. Most OPE methods, e.g. FQE, would effectively be an entirely "black box" component when used in a decision-making process, outputing a single scalar value (expected return of policy $\pi$). While potentially accurate, this offers a human decision-maker no insight into why a policy is good or how the system is expected to behave, and it is difficult to know when to reject OPE estimates as inaccurate. 

A DT, conversely, outputs a multivariate simulation for each policy under consideration. This can be useful, addressing some of the drawbacks of OPE methods:
1. **Consideration of Subjective/Secondary Criteria:** Decisions may involve subjective factors that are not perfectly captured by a formally defined reward function. By generating rollouts, DTs allow practitioners to assess policies in terms of some subjective factors, alongside the more formal reward of interest, to enable a comprehensive decision.
2. **Human-in-the-loop "Rejection Sampling":** If a user observes a simulation that violates known constraints (e.g., negative blood pressure), they immediately know to trust the model's recommendation less for that specific policy. OPE does not provide this safety check.

**Update:** We will expand our discussion in $\S 1$ and $\S 5.2$ on the practical interpretability benefits of DTs.

---

**(C) Theoretical assumptions**

We appreciate your careful reading of Theorems $3.1$ and $3.3$. You are correct that the associated assumptions may not always hold in real-world scenarios. However, we have already empirically investigated some settings where these assumptions may be violated, in $\S 6.2$. Here, we used multiple expressive architectures (Transformers, ResNets, Neural ODEs, MLPs, RNNs) as the backbone of DT models, and we saw that $DT^2$ still showed a marked, consistent improvement over standard training. This demonstrates that the core takeaway of our theory—that simulation loss can misallocate capacity for decision-making purposes—remains practically relevant even when using expressive architectures.

---

**(D) Generalisation to unseen policies $\pi \notin \Pi$**

We agree that out-of-distribution (OOD) generalization is an important consideration for DTs. Please note that we have already shown some evidence for this in $\S6.3$, where we compare evaluate rankings of $11$ unseen policies. 

For further evidence, we have now run some additional OOD evaluations on the three smaller continuous control environments from $\S6.2$ (Pendulum, Lunarlander, Hopper). We compare standard and $DT^2$ policy rankings (using the ResNet architecture, for consistentency with $\S6.3$) for $5$ OOD policies: *Random policy, Constant 0 action, Constant Minimum Action, Constant Middle Action,* and *Constant Max Action*. These constant-action policies are less sophisticated than the PPO policies used in training, and their different visitation distributions and lower returns make them OOD.

From the table [here](https://anonymous.4open.science/r/DT2-r/OOD-results.pdf) we see a consistent improvement in ranking performance by $DT^2$. While the difference is, expectedly, more modest than for in-distribution policies, this provides further evidence that $DT^2$ is not overfitting to the training policies, but is learning general, decision-relevant dynamics.

**Update:** We will add these OOD evaluations to $\S 6.3$.

---

Thank you once again. We hope that we have addressed all your comments, and we would greatly appreciate any further feedback. 