# Relationship bettween probability path, flow, and velocity field

In this article, we illustrate the relationship between the probability density path and the continuity equation.

## What is probability path

A probability path is a path that the probability density function is gradually transformed from one distribution to another.

In the flow model, we also have the probability path. Suppose $X_t = \psi(X_0,t)$, and $X_t\sim p_t$, the $p_t$ defines a probability path from $X_0\sim p$ to $X_1\sim q$.

Recall the push forward function in the normalize flow method,

$$
 p_x(\mathbf{x}) = p_z(f^{-1}(\mathbf{x})) \left| \det J_{f^{-1}}(\mathbf{x}) \right|
$$

where $f$ is the flow function that maps from $z$ to $x$ and $f^{-1}$ is its inverse.

Denote

$$\psi_{\#}P_z = p_x(\mathbf{X})$$

Thus we have

$$ p_{t}(x)   = \psi_{\#} p(x) $$

We say $v(x,t)$ generates the probability path from $x_0\sim p$ to $x_1\sim q$ if

> $$X_t = \psi(X_0,t)\sim p_t \; \forall t\in [0,1)$$

## Continuity Equation

<div class="theorem">
<strong>Theorem: Continuity Equation </strong>
Let $p_t$ be a probability path and $v(x,t)$ be a vector field. The continuity equation states that the probability density function $p(x)$ satisfies

$$\frac{d p_t(x)}{dt} = -\nabla \cdot\big[ v(x,t) p_t(x)\big]$$
</div>
