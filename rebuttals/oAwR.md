Thank you for your thoughtful comments and suggestions. We give answers to each of your questions/concerns below.

---

**(A) Computational cost**

We agree that analyzing the computational trade-offs of $\text{DT}^2$ is important. Computational overhead compared to standard simulation-loss training is due to the necessary training of the FQE value functions and the $H$-step unrolling to obtain model-estimated policy values.

Concretely, across the continuous control experiments, the training time for $\text{DT}^2$ ranged averaged ..., while simulation-loss training averaged ...

However, it is worth noting that $\text{DT}^2$ training overhead is comparable to other decision-aware modeling paradigms. For instance, the **VaGraM** baseline we already compared to also required the training of value models for each policy $\pi \in \Pi$ to compute value-gradients during training.

Furthermore, we now compare with some sophisticated DT and offline model-based RL baselines (please see point (B) of our response to reviewer `frev` for details). These new models also take significantly longer than our simulation-loss baselines. Specifically, MOReL and MOPO involve training ensembles of dynamics models, ROMI requires alternating between training a dynamics ensemble and a SAC policy, and HDTwin requires repeated queries to an LLM to refine a hybrid mechanistic and neural model, and undergoing multiple training rounds to reach its final model. These models average ..., ..., ..., and ... training minutes on the continuous control environments.

**Update:** We will expand upon the computational costs of each method in our discussion section.

---

**(B) Interpretability**

To concretely compare interpretability between OPE methods and DTs is difficult, because their outputs are very different. Most OPE methods, e.g. FQE, would effectively be an entirely "black box" component when used in a decision-making process, as they output a single scalar value (expected return of policy $\pi$). While potentially accurate, this offers a human decision-maker no insight into why a policy is good or how the system should be expected to behave over time, and it is difficult to know when to reject OPE estimates as inaccurate. 

A DT, conversely, outputs a full, multivariate simulation of the unrolling of each policy under consideration. This can be very useful, as it addresses some of the drawbacks of OPE methods:
1. **Consideration of Subjective/Secondary Criteria:** Real-world decisions may involve subjective factors that are not perfectly captured by a formally defined reward function. By generating rollouts, DTs can allow practitioners to assess policy values in terms of these subjective factors, alongside the more formal reward of interest, to enable a more comprehensive decision.
2. **Human-in-the-loop "Rejection Sampling":** If a user observes a DT simulation that violates known constraints (e.g., negative blood pressure), they immediately know to trust the model's recommendation less for that specific policy. OPE does not provide this safety check.

**Update:** We will expand our discussion in $\S 1$ and $\S 5.2$ on the practical interpretability benefits of DTs over scalar OPE estimates.

---

**(C) Theoretical assumptions**

We appreciate your careful reading of Theorems $3.1$ and $3.3$. You are correct that the associated assumptions may not always hold in real-world scenarios, especially when expressive model classes are used to instantiate a DT.

However, we have already deliberately empirically investigated some settings where these assumptions may be violated, in $\S 6.2$. In the experiments, we used multiple expressive architectures (Transformers, ResNets, Neural ODEs, MLPS, RNNs) as the backbone of DT models, and we saw that $\text{DT}^2$ still showed a marked, consistent improvement in policy rankings over standard training. This empirically demonstrates that the core takeaways of our theory—that simulation losses can misallocate capacity for decision-making purposes—remains practically relevant even when using modern expressive architectures.

---

**(D) Generalisation to unseen policies $\pi \notin \Pi$**

We agree that out-of-distribution (OOD) generalization is an important consideration for DTs. Please note that we have already demonstrated some evidence for this in $\S6.3$, where we compare ranking performance between $\text{DT}^2$ and NLL DT on $11$ unseen policies. 

For further evidence, we have now run some additional OOD evaluations on the three smaller continuous control environments from $\S6.2$ (Pendulum, Lunarlander, Hopper). We compare NLL and $\text{DT}^2$ policy rankings (using the ResNet architecture, for consistentency with $\S6.3$) for $5$ OOD policies: *Random policy, Constant 0 action, Constant Minimum Action, Constant Middle Action,* and *Constant Max Action*. These constant-action policies are far less sophisticated than the PPO policies used in training, and their different state visitation distributions and generally lower returns make them fairly OOD.

From the table HERE(link), we see a consistent improvement in ranking performance by $\text{DT}^2$ compared to the NLL DT. While the difference is, expectedly, more modest than for in-distribution policy decisions, this provides further evidence that $\text{DT}^2$ is not overfitting to the training policies, but is learning more general, decision-relevant dynamics.

**Update:** We will include these new OOD evaluations alongside the existing OOD case study in $\S 6.3$.

---

Thank you once again. We hope that we have addressed all your comments, and we would greatly appreciate any further feedback. 