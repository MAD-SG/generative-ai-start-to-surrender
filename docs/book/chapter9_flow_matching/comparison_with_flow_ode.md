# Comparison with Diffusion Flow ODE
We can show that both flow matching and Diffusion Flow ODE can arrive in same formula.

In order to be conistent, we assume the forward process is both from $p_0$ to $p_1$, where in practive, the $p_0$ is the data distribution, $p_1$ is the noise distribution (simple distribution). In the flow matching, we don't need require that $p_1$ is the Gaussian distribution.

Recall in [diffusion unified representation](../chapter7_diffusion/sde_diffusion_unified_representation.md) and [Affine Conditional Flow](./affine_conditional_flows.md)

## Numerical Experiments

To have a similar compare with what we done in the SDE diffusion model, we take the distribution $p_0$ be the gaussian distribution and $p_1$ be the funnel distribution in the two dimensional space.

The Funnel distribution is defined as follows:

- \( v \sim \mathcal{N}(0, 3^2) \)
- \( x \mid v \sim \mathcal{N}\bigl(0, \exp(v)\bigr) \)

|gaussian distribution $\epsilon$|funnel distribution $x_{data}$|
|---|---|
|![](../../images/image-112.png)|![](../../images/image-79.png)|

Thus, the joint density is given by:

$$
q(x,v) = p_1(x,v) = \frac{1}{3\sqrt{2\pi}} \exp\left(-\frac{v^2}{18}\right)
\cdot \frac{1}{\sqrt{2\pi\,\exp(v)}} \exp\left(-\frac{x^2}{2\exp(v)}\right)
$$

