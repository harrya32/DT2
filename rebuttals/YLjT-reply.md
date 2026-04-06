Thank you for your efforts during this review period, and your continued engagement with our work. We are glad that the new experiments are well received, and we will certainly add these new results into the final manuscript. We are also very grateful for the resulting score increase!

Regarding the Decision-Focused Learning (DFL) baseline:

While we believe that the specific DFL papers you included cannot be easily applied to the continuos control environments we consider (because they are designed for simple MDPs or discrete action spaces, which can make the ways that they differentiate through the policy optimisation process incompatible), we have identified a more compatible DFL MDP approach, termed *Optimal Model Design (OMD)* ("Control-Oriented Model-Based Reinforcement Learning with Implicit Differentiation", Nikishin et al., 2022). This work uses a method quite similar to the Sharma et al. (2024),  paper, but it is not concerned with generalising to unseen reward functions (which is unecessary for our setting), and it applies OMD to more complex environments and transition functions. OMD uses implicit differentiation to optimise the learned model such that the Q-function of the optimal policy within it matches the true Q-function of the optimal policy, and it is demonstrated on MuJoCo environments using deep learning models. We have adapted it to our offline setting, such that we can now compare with a DFL approach. Notably, OMD is an entirely DFL approach, in that it is not concerned with simulation fidelity at all. 

As with the other MBRL baselines, we see that OMD it underperforms our $\text{DT}^\text{2}$ method on our specific task of ranking candidate policies, performing relatively worse than most other methods evaluated. We have now created a Figure-4-style comparison of all of these loss functions across the 6 continuous control environments [here](https://anonymous.4open.science/r/DT2-rebuttals-reply-2-6D34/README.md). This includes comparisons between the following losses:

1. Standard NLL 
2. Value-function-gradient weighted MSE (VaGraM)
3. Pessimistic NLL/MSE via uncertainty (MOReL, MOPO)
4. Adversarial pessimistic MSE (ROMI)
5. DFL (OMD)
6. Our ranking loss ($\text{DT}^\text{2}$)

In most environments, $\text{DT}^\text{2}$ outperforms in terms of regret (best in 5/6 environments) and Spearman's correlation (best in 5/6 environments). We will add this into the empirical section of our final manuscript, to compare across loss functions, as done here, as well as our initial comparison across architectures.

Thank you once again for helping us improve our work, these additional results are certainly worthwhile comparisons and they strengthen the paper. Again, we greatly appreciate your time and effort, and are grateful for the resulting increase in score. We hope that we have now fully resolved your concerns.