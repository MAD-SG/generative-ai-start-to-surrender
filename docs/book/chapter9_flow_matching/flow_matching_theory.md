# Flow Matching Theorem
> In this article, we establish the theroy foundation of the flow matching method
> Please read the section introduction for common knowledges of the flow matching and standard notations

## Formula of flow matching

Suppose $v(x,t)$ is the velocity field from push the probability $p$ to $q$. Our aim is to estimate the delocity field

$$L_{FM}=E_{X_t\sim p_t} ||v(x,t) - v_\theta(x,t)||$$

Here $||$ could be any distance metric besides the normal mean square error.

For simplicity, we denote the $\pi_{0,1}(X_0,X_1)$ be the joint distribution of the data coupling

$$(X_0,X_1)\sim \pi_{0,1}(X_0,X_1)$$

Although, the common used distirbution of $X_0$ is the Gaussian noise, we just consider the general distribution in this section.

### Conditional probability path

Let $p_{t|1}(x|x_1)$ be the consitional probability path.

Then the marginal probability path (responde to the joint distribution $\pi_{0,1}$) $p_t$

$$p_t(x) = \int p_{t|1}(x|x_1) q(x_1)d x_1.$$

and $p_t$ satisfied the boundary condition

$$p_0 = p, \qquad p_1 = q$$

which required the conditional probability path satisfy

$$
p_{0|1}(x|x_1) = \pi_{0|1}(x|x_1), and \; p_{1|1} (x|x_1) = \delta_{x_1}(x)
$$

where $\delta_{x_1}$ is the delta measure centered at $x_1$.

If we consider the independent data coupling, then

$$\pi_{0|1}(x_0|x_1) = \pi_{0,1}(x_0,x_1)/q(x_1)$$

The constrains becomes $p_{0|1}(x|x_1) = p(x)$.

The second condition could also be written as

$$\int p_{t|1}(x|y) f(y) d y \rightarrow f(x)$$

as $t\rightarrow 1$ for any continuous function $f$ since $\delta$ has no density function.

### Conditional Velocity Field

Let $u_t(\cdot|x_1)$ generates $p_{t|1}(\cdot | x_1)$

Thus we have

$$u_t(x) = \int u_t(x|x_1) p_{x_1|x} d x_1$$

This can be viewed as a weighted average of the conditional velocities.
