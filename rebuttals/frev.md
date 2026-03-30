Thank you for your thoughtful comments and suggestions. We give answers to each of your questions/concerns below.

---

**(A) Practical reasons for DTs**

The core motivation for using a DT as a practical surrogate stems from the risks, costs, and ethical barriers associated with trial-and-error in high-stakes domains. In fields like medicine, finance, climate policy, or manufacturing, executing a policies in the real world can have irreversible consequences. 

For example, in our medical case study ($\S 6.3$), consider how a DT could benefit a clinician treating a cancer patient. When a clinician needs to choose a treatment plan for a patient with cancer, it is ethically and practically impossible to test multiple different treatments on the real patient to see which yields the best outcome. 

Instead, by simulating the outcomes and ranking the efficacy of different treatment options *before* deployment, we can optimise a patient's treatment plan without risking their health with real experiments. DTs can serves as *in silico* testing grounds to simulate trajectories and establish preference orderings without risking real-world harm.

**Update:** We will expand the introduction to clearly motivate DTs in real-world settings.


---

**(B) Added baselines**

**Why VaGraM?** We chose to adapt VaGraM because it is a prominent value-aware Model-Based RL (MBRL) method. As we describe in $\S 5.3$, value-aware MBRL methods are related to our setting, as there share some common themes with our work in terms of motivation and requirements. By comparing to VaGraM (appropriately adapted to our setting), we wanted to test whether encoding some notion of the policy values within $\pi \in \Pi$ into the dynamics model, through an existing appraoch (VaGraM's value-gradient weighting), could achieve the comparable benefits as our proposed ranking loss. Our results showed that our method of explicitly targeting pairwise rankings is far more effective for this specific task.

**Added Baselines:** To provide a more exhaustive comparison to existing works, we have run additional experiments against a wider suite of baselines. This includes a recent general digital twin baseline (**HDTwin**) and several prominent offline MBRL methods (**MOReL, MOPO, ROMI**), which, again, also attempt to learn models from offline data and are therefore a related literature stream. As shown in the table [here](https://anonymous.4open.science/r/DT2-r/mbrl-and-hdtwin-results.pdf), $\text{DT}^2$ significantly outperforms all of these baselines in both decision regret and ranking correlation across the six continuous control environments we consider. 

Nevertheless, we do think it is important to recognise that the simulation-trained models (using MSE/NLL losses) already presented in our paper can be seem as a representation of many current DT works, as indeed the vast majority of recent DT literature do employ such simulation-based losses (see concrete citations in our introduction).

However, many recent DT papers are highly applied—proposing specific neural architectures or physics-informed equations for individual, niche systems. Therefore, rather than comparing against specialized domain architectures, we evaluated our general training paradigm across a wide variety of standard architectures (MLPs, Transformers, ResNets, Neural ODEs, RNNs) and in modelling different systems.


**Update:** We will extend our empirical results section to include these additional baselines to better position our method against closely related literature.

---

**(C) Presentation and Figures**

We appreciate your constructive feedback regarding the presentation. We will make the requested changes:
* We will refine Figure 1 by simplifying the legend and adding subfigure titles to improve clarity. 
* We will make the colours in Figure 2 more distinct, and replace the "X" notation denoting the position of the true $\Phi$ with a circular marker.
* We will bring the **Related Works** section forward to $\S 2$, and expand it to more clearly highlight the gaps in the current DT and decision-focused literature that our work addresses.

---

Thank you once again. We hope that we have addressed all your comments, and we would greatly appreciate any further feedback. 