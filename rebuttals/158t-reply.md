Thank you for your continued engagement with our work. We appreciate the follow-up question, as it prompted us to formally benchmark FQE and clarify the motivation for our method. We address your comments below.

---

**(A) Empirical comparison with FQE**

We thank the reviewer for raising this point. Originally, we did not quantitatively compare against FQE because we viewed it as falling under a different decision-making paradigm (see point (B) for reasons). Our statement that "FQE generally had lower regret than the DT models" was made heuristically, based on judgement from early experimentation. However, this should be verified, as we agree that FQE can nevertheless serve as an useful baseline. Prompted by your feedback, we evaluated FQE rankings in our continuous control environments, and have constructed new figures with it included, linked [here](https://anonymous.4open.science/r/DT2-r-r/README.md).

We do indeed see that FQE performs best, in terms of regret and Spearman's correlation, in two environments  - Hopper and Ant. Interestingly, however, in the remaining four environments one or multiple $DT^2$ configurations actually outperform FQE, but FQE still performs relatively competitively in these. The results are largely favourable for our method, and show that it can outperform FQE in terms of ranking performance. Also, DTs have other benefits that make them particularly useful for human-in-the-loop decision-making processes, which is our concrete focus in this paper, discusssed below.

**Update:** We will include the FQE baseline in $\S 6.2$.

---

**(B) Decision-making with DTs vs. FQE**

Even in cases where FQE might have a lower regret, DTs possess several distinct, structural advantages that make them more useful for our overarching goal: **human-in-the-loop decision support**. 

FQE would effectively act as an entirely "black box" component when used in a decision-making process, outputting a single scalar value (expected return of $\pi$). While potentially accurate, this offers a human decision-maker no insight into why $\pi$ is good, or how the system is expected to behave, and it is difficult to know when to reject FQE estimates as inaccurate. 

A DT, conversely, outputs a multivariate simulation for each $\pi$ under consideration, uniquely permitting:
1. Consideration of Subjective/Secondary Criteria: Decisions may involve subjective factors that are not perfectly captured by a formally defined reward function. By generating rollouts of all variables, DTs allow practitioners to assess $\pi$ in terms of some subjective factors, alongside the more formal reward of interest, to enable a comprehensive decision.
2. Human-in-the-loop "Rejection Sampling": If a user observes a simulation that violates known constraints (e.g., negative blood pressure), they immediately know to trust the model's recommendation less for that specific $\pi$. FQE does not provide this safety check.


The above give us motivation in our work to ensure that, in our decision-targeted training method, the simulation fidelity of the DT remains strong, such that the DT can still be used for the above tasks. We do observe that the test-set MSE from $DT^2$ remains relatively good in $\S 6.2$ and $\S 6.3$, and we show that $\lambda$ can effectively be used to trade this off with decision quality ($\S 6.4$).

Furthermore, there are positives in terms of efficiency in decison-making for DTs over FQE.

3. Once trained, a DT can evaluate new, unseen policies, at inference time, just by generating rollouts that they induce. FQE, on the other hand, requires a full training loop to be run for each policy under consideration, which leads to considerably slower decision-making processes, and requires the user to maintain access to the offline dataset. Because DTs are much more efficient in this respect, they can be used more interactively to conduct "what-if" planning, where a user can tweak a policy and quickly observe the simulated outcomes.

**Update:** We will expand our discussion in $\S 5.2$ on the benefits of DTs over FQE.

---

Thank you once again. We believe these additions will strengthen the paper's narrative and thoroughness. We hope that we have addressed all your comments, and, if so, that you would consider raising your score.