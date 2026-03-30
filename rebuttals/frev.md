- (A) Practical reasons for DTs

Why is having a simulator of a real-world system useful? For experiments/planning/decision-making in high-stakes domains, e.g. medicine, finance, business, where it may not be practical to try many different options. e.g. if we have a patient, we need to choose a treatment plan, it is not feasible to test many different treatments (as a form of experimenting), and then decide to continue with whichever had the best results. We want some way to expect the results/ranking of different options before deployment.

- (B) Baseline methods

We now compare to a larger suite of baselines, including more MBRL baselines (offline ones in this case), and a recent digital twin paper HDTwin. Please note that we see all of the simulation-trained models currently in the paper as useful DT baselines already, as this is the prevailing training paradigm in the majority of DT literature. Nevertheless, DT papers tend to be more applied, e.g. design architectures for their specific systems, so these can be unhelpful to compare to, hence we tried to make the comparison as broad as possible, across different systems and architectures, chaning just the loss.