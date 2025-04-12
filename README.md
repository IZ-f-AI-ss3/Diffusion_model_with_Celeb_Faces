# Flow Matching for Human Face Generation

This repository explores using Flow Matching to generate new human faces based on the CelebFaces (CelebA) dataset.

The training procedure for this particular flow matching model is as follows:

$$
\begin{aligned}
\text{ Flow Matching Training Procedure (General)} \\
    \text{Require: A dataset of samples } z \sim p_{\text{data}}, \text{ neural network } u_t^\theta \\
    \text{For each mini-batch of data:} \\
      \quad \text{Sample a data example } z \text{ from the dataset.} \\
      \quad \text{Sample a random time } t \sim \text{Unif}_{[0,1]}. \\
      \quad \text{Sample } x \sim p_t(\cdot \mid z) \\
      \quad \text{Compute loss:} \\
      \quad \quad \mathcal{L}(\theta) = \left\| u_t^\theta(x) - u_t^{\text{target}}(x \mid z) \right\|^2 \\
      \quad \text{Update the model parameters } \theta \text{ via gradient descent on } \mathcal{L}(\theta) \\
\text{End for}
\end{aligned}
$$

Once we have learned a good $u_t^\theta$, the next step to generate new samples that follow the $p_{\text{data}}$ distribution is to simulate the solution of an ODE or an SDE. However, in this repository, I will focus only on the ODE, even though they are equivalent in a certain sense (since we are in the conditional probability path and $p_{\text{init}}$ is Gaussian, we already have a formula linking them).

