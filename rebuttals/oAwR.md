- (A) Computational cost

We describe training times here now... Not that VaGraM also requires these value models that DT2 does... Also, please see our repsonse to reviewer YLjT, where we now compare to some more offline MBRL baselines. All of these use ensembles, and larger architectures (and ROMI trains a policy alongside the model, and they influence each other so that cannot be decoupled). There respective training times are...

- (B) Interpretability

OPE can only offer a scalar reward estimate. A DT also offers a simulation over the time scale and variables requested. Naturally, this can be used to do some 'rejection-sampling', where if a user sees the simulation violates known constraints (e.g. non-negativity, smoothness), this would be a signal to trust the DT less. Also, it is a direct expectation of what will happen under the given policy, so it gives a richer signal than just the OPE-estimated value. The simulation can be used to assess factors that are not well-captured by the reward function, like individualistic/subjective factors

- (C) Theoretical assumptions

Please note that we do investigate what happens when these assumptions likely do not hold, in our expressive hypothesis space examples. We see that, even in this less controlled case, DT2 training has a marked improvement over simulation-only training, and existine baselines.

- (D) Generalisation to unseen policies

... can we do some quick experiment here... like in the cont. control envs, with other PPO policies, or manually defined ones?