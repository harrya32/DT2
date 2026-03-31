Thank you for your thoughtful comments and suggestions. We give answers to each of your questions below.

---

**(A) Practical reasons for DTs**

The core motivation for using a DT as a practical surrogate stems from the desire to make optimal decisions (i.e. choosing a policy $\pi \in \Pi$) in situations when the practical risks, costs, and ethical barriers associated with experimentation on the real-world system make trial-and-error infeasible. In fields like medicine, finance, climate policy, or manufacturing, executing policies in the real world can have irreversible consequences, and so there must be some way to deliberate over different options without executing each one.

For example, related to our medical case study ($\S 6.3$), consider how a DT could generally benefit a clinician treating a cancer patient. When choosing a treatment plan for a patient with cancer, it is usually impossible, both ethically and safety-wise, to test multiple different treatments in tandem/one after the other, to see which yields the best outcome, and then choose to continue with the optimal one. Instead, using a DT to simulate the effects of each policy can allow the clinician to rank the efficacy of different treatment options before deployment, and optimise a patient's treatment plan without risking their health with real-world experiments. DTs can serves as practical *in silico* testing grounds to simulate trajectories and establish preference orderings without risking real-world harm. Similar examples also can also be constructed in the other domains, beyond medicine, mentioned earlier.

**Update:** We will **expand the introduction** to clearly **motivate the use of DTs** in real-world settings.

---

**(B) Added baselines**

**Why VaGraM?** We chose to adapt VaGraM because it is a prominent value-aware Model-Based RL (MBRL) method. As we describe in $\S 5.3$, value-aware MBRL methods are related to our setting, as there share some common themes with our work in terms of motivation and algorithm. Through the comparison with VaGraM (appropriately adapted to our setting), we wanted to test how $\text{DT}^2$ compares to a representative method from this broad field, and also to see whether biasing training of a dynamics model based on some notion of the policy values for $\pi \in \Pi$, through an existing appraoch (VaGraM's value-function-gradient weighting), could achieve comparable benefits as our proposed ranking loss. Our results showed that our method, explicitly targeting pairwise rankings rather than value-function gradient weighting, is far more effective for this specific task.

**Added Baselines:** To provide a more complete comparison to existing works, we have now run additional experiments against a wider suite of baselines. This includes a recent hybrid digital twin baseline (HDTwin (Holt et al., 2024)) and several prominent offline MBRL methods (MOReL (Kidambi et al., 2020), MOPO (Yu et al., 2020), ROMI (Qiao et al., 2026)). As shown in the table linked [here](https://anonymous.4open.science/r/DT2-r/mbrl-and-hdtwin-results.pdf), where we compare the best performing $\text{DT}^2$ and standard DT architectures, per environment, against these new methods, $\text{DT}^2$ clearly outperforms all of these baselines in both decision regret and Spearman's correlation across the majority of the six continuous control environments we consider. 

Furthermore, we do think it is important to recognise that the standard DT models (using MSE/NLL losses) already presented in our paper can be seem as a representation of current DT literature, as indeed the vast majority of recent DT works do employ such simulation-based losses (see concrete citations in our introduction). We did demonstrate how many modern architectures (MLPs, Transformers, ResNets, Neural ODEs, RNNs), which are often used as baselines in DT papers (e.g. in (Holt et al., 2024)), can be improved upon using our loss rather than these typical simulation losses.

**Update:** We will **extend our empirical results section** to include these **additional DT and offline MBRL baselines** to better position our method against closely related literature.

---

**(C) Presentation and Figures**

We appreciate your constructive feedback regarding the presentation. We will make the requested changes:
* We will refine Figure 1 by simplifying the legend and adding subfigure titles to improve clarity. 
* We will make the colours in Figure 2 more distinct, and replace the "X" notation denoting the position of the true $\Phi$ with a circular marker.
* We will bring the Related Works section forward to $\S 2$, and expand it to more clearly highlight the gaps in the current DT  literature that our work addresses.

---

Thank you once again. We hope that we have addressed all your comments, and we would greatly appreciate any further feedback.