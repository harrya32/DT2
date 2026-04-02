Thank you for your continued engagement with our work. We address your further questions below.

---

**(A) Empirical comparison with FQE**

Explain that we had not fully quantitively investigated FQE before this, since we consider it a different decision-making paradigm (see B).

We agree though that it is worth comparing to. We now have a new figure 4... link

Describe it wins some envs, doesnt win others. Describe avg. regret/spearmans if useful

We still see the best performing DT2 models lead to the outright best ranking in some environments, but are outperformed by FQE in Hopper and Ant, specifically. Nevertheless, DTs also have other benefits that make them particularly useful for human-in-the-loop decision-making processes, which discusssed below

**Update:** We will add these further OOD ranking experiments to $\S 6.3$.

---

**(B) DTs vs. FQE**

- Able to observe rollouts, for comparisons in more qualitative ways than just via a reward function. Allows capturing more subjective/personal factors that may not be well-defined in the formal reward function of interest.

- Rejection sampling if implausible.

The above form our motivation throughout the paper to ensure that, in decision-targeted training, the simulation loss is affected minimally (i.e. MSE does not drastically increase), such that the DT can still be useful for the above tasks.

Furthermore, there are positives in terms of efficiency in decison-making for DTs over FQE.

- Once trained, a DT can evaluate new, unseen policies, at inference time, by generating rollouts that it induces. FQE, on the other hand, requires training a value function for each policy under consideration, which leads to considerably slower decision-making processes, and requires constaconsistent access to the offline dataset. Because DTs are much more efficient in this respec, they can be used more easily for "what-if" planning, involving small changes to a policy under consideration, to see how that effects outcomes. FQE would require full retraining for each change in this scenario, whereas DTs just need to generate new rollouts.

**Update:** 

---

Thank you once again. We hope that we have addressed all your comments, and, if so, that you would consider raising your score.