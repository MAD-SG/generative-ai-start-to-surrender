# Score Function

In the context of the energy based model, we have the two types of score function:

1. **Stein's score function**

    $$\tag{1}
    s_\theta(x) \overset{\text{def}}{=} \nabla_x \log p_\theta(x).
    $$

2. **ordinary score function**

    $$
    \tag{2}
    s_x(\theta) \overset{\text{def}}{=} \nabla_\theta \log p_\theta(x).
    $$

In the context of Langevin dynamics and the underlying Fokker-Planck equation, a key component is the gradient of the log-likelihood function \(\nabla_x \log p(x)\), which is called **ordinary score function** in equation (2).

On the other hand, the derivative of the log density with respect to the sample variable $x$ is called **Stein's score function** in equation (1).

Despite the naming conventions, in diffusion literature, Stein's score function is often simply referred to as the score function, and we will adopt this terminology here.

In the context of contrastive divergence (CD), the ordinary score function is used to directly calculate the derivative of $p_\theta$ and update the network parameters.

In score matching methods, Stein's score function is employed, where the loss is defined as the divergence between the gradient of the learned score function and the true gradient of the log-likelihood function.

Suppose $p_\theta$ is a mixture of two Gaussian distributions in one and two dimensions. We can visualize the score function of $p_\theta$ in the following figures:

![1D Gaussian Mixture Score Function](../../images/image-24.png)

![2D Gaussian Mixture Score Function](../../images/image-23.png)

In two dimensions, the score function can be visualized as a vector field of the log-likelihood function. The arrow direction represents the gradient, and the arrow length indicates the gradient's magnitude. By employing Langevin dynamics, we can randomly choose a point and follow the arrows to sample from the log-likelihood function, ultimately tracing the trajectory of Langevin dynamics. In physics, the score function is analogous to "drift," indicating how diffusion particles should flow towards the lowest energy state.

### Reference

- code about the score function of gaussian mixture: experiment/score_function_gaussian.ipynb
