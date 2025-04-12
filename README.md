# Flow Matching for Human Face Generation

This repository explores using Flow Matching to generate new human faces based on the CelebFaces (CelebA) dataset.

The training procedure for this particular flow matching model is as follows:

1. **Require**: A dataset of samples $z \sim p_{\text{data}}$, neural network $u_t^\theta$
   
2. **For each mini-batch of data**:

   - Sample a data example $z$ from the dataset.
   - Sample a random time $t \sim \text{Unif}_{[0,1]}$.
   - Sample $x \sim p_t(\cdot \mid z)$.

3. **Compute loss**:

   $$
   \mathcal{L}(\theta) = \left\| u_t^\theta(x) - u_t^{\text{target}}(x \mid z) \right\|^2
   $$

4. **Update model parameters** $ \theta $ via gradient descent on $\mathcal{L}(\theta) $.

5. **End for**


Once we have learned a good $u_t^\theta$, the next step to generate new samples that follow the $p_{\text{data}}$ distribution is to simulate the solution of an ODE or an SDE. However, in this repository, I will focus only on the ODE, even though they are equivalent in a certain sense (since we are in the conditional probability path and $p_{\text{init}}$ is Gaussian, we already have a formula linking them).

