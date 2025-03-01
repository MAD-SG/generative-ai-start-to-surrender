# Relationship bettween probability path, flow, and velocity field
[TOC]
In this article, we illustrate the relationship between the probability density path and the continuity equation.

## What is probability path

A ==probability path== is a path that the probability density function is gradually transformed from one distribution to another.

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

We say $v(x,t)$ ==generates== the probability path from $x_0\sim p$ to $x_1\sim q$ if

> $$\tag{1} X_t = \psi(X_0,t)\sim p_t \; \forall t\in [0,1)$$

## Velocity Field

Given the definition of $\psi(x,t)$, we have the velocity field $v(x,t)$

$$\tag{2}\frac{\text{d} \psi(x,t)}{\text{d} t} = v(\psi(x,t),t)$$

!!! thm "Flow Existence"
    ODE (2) has a unique solution in $C^r (\Omega,R^d)$ diffeomorphism if $u$ is $C^r([0,1]\times R^d, R^d)$
??? proof "Solution Existence"
    Proof omitted. It is easy to prove by constructing the integral function.

## Continuity Equation
Continuity equation establishes the relation ship between probability path and velocity fields which is come from the mass conversation.

!!! thm "Continuity Equation"

    Let $p_t$ be a probability path and $v(x,t)$ be a vector field. The continuity equation states that the probability density function $p(x)$ satisfies

    $$\frac{d p_t(x)}{dt} = -\nabla \cdot\big[ v(x,t) p_t(x)\big]$$

??? proof  "Continuity Equation"
    Here's a concise proof using $\rho \cdot \mathbf{v}$ directly:

    1. **Probability Conservation:**
    For a probability density $\rho(\mathbf{x}, t)$, the total probability in any volume $V$ is
    $$
    P_V(t) = \int_V \rho(\mathbf{x}, t)\, d\mathbf{x}.
    $$
    Conservation of probability means that any change in $P_V(t)$ comes solely from the flow of probability across the boundary $\partial V$.

    2. **Expressing the Rate of Change:**
    If probability moves with a velocity field $\mathbf{v}(\mathbf{x}, t)$, then the probability flux (flow per unit area) is $\rho(\mathbf{x}, t)\mathbf{v}(\mathbf{x}, t)$. Thus, the rate at which probability leaves the volume $V$ is
    $$
    \frac{d}{dt}P_V(t) = -\int_{\partial V} \rho(\mathbf{x}, t) \,\mathbf{v}(\mathbf{x}, t)\cdot d\mathbf{S},
    $$
    where $d\mathbf{S}$ is the outward-pointing surface element.

    3. **Applying the Divergence Theorem:**
    The divergence theorem converts the surface integral to a volume integral:
    $$
    \int_{\partial V} \rho \,\mathbf{v}\cdot d\mathbf{S} = \int_V \nabla \cdot (\rho\, \mathbf{v})\, d\mathbf{x}.
    $$
    Thus, we have
    $$
    \frac{d}{dt}P_V(t) = -\int_V \nabla \cdot (\rho\, \mathbf{v})\, d\mathbf{x}.
    $$

    4. **Equating the Two Expressions:**
    On the one hand, directly differentiating $P_V(t)$ gives
    $$
    \frac{d}{dt}P_V(t) = \int_V \frac{\partial \rho}{\partial t}\, d\mathbf{x}.
    $$
    Equating the two expressions:
    $$
    \int_V \frac{\partial \rho}{\partial t}\, d\mathbf{x} = -\int_V \nabla \cdot (\rho\, \mathbf{v})\, d\mathbf{x}.
    $$

    5. **Local Form of the Continuity Equation:**
    Since the volume $V$ is arbitrary, the integrands must be equal at every point:
    $$
    \frac{\partial \rho}{\partial t} + \nabla \cdot (\rho\, \mathbf{v}) = 0.
    $$
    This is the continuity equation expressed directly in terms of $\rho$ and $\mathbf{v}$.

!!! thm "Uniquness of $p_t$"
    Given the continuity equation and $v(x,t)$, there must exist a unique solution $p_t$ which is the probability path generated by $v(x,t)$.

    Below is a simpler proof based on an energy estimate (using the $L^2$ norm) and Gronwall’s inequality. We assume that $r(t,x)$ is sufficiently smooth and decays at infinity so that all integrations by parts are justified, and that $v(x,t)$ is locally Lipschitz with bounded divergence.

??? proof "Uniqueness of the solution"

    ### The Setting
    let $p_t$ and $q_t$ both satisfied the continuity equation with same initial condition. Let $r(t,x) = p_t(x) - q_t(x)$.
    Then $r(t,x)$ satisfy

    $$
    \partial_t r(t,x) = -\nabla\cdot\big(r(t,x)\,v(x,t)\big), \quad r(0,x) = 0.
    $$

    It is enough to show that  $r(t,x)\equiv 0$ for all $t\ge0$.

    ### Step 1. Multiply by $r$ and Integrate

    Multiply the equation by $r(t,x)$ and integrate over $\mathbb{R}^d$:

    $$
    \int_{\mathbb{R}^d} r\,\partial_t r\,dx = -\int_{\mathbb{R}^d} r\,\nabla\cdot\big(r\,v\big)\,dx.
    $$


    ### Step 2. Rewrite the Left-Hand Side

    Notice that
    $$
    \int_{\mathbb{R}^d} r\,\partial_t r\,dx = \frac{1}{2}\frac{d}{dt}\int_{\mathbb{R}^d} r^2\,dx.
    $$


    ### Step 3. Rewrite the Right-Hand Side by Expanding the Divergence

    Expand the divergence:
    $$
    \nabla\cdot\big(r\,v\big) = v\cdot\nabla r + r\,\nabla\cdot v.
    $$
    Thus,
    $$
    -\int r\,\nabla\cdot\big(r\,v\big)\,dx = -\int r\,(v\cdot\nabla r)\,dx - \int r^2\,\nabla\cdot v\,dx.
    $$


    ### Step 4. Handle the First Term on the Right

    Consider the term
    $$
    -\int r\,(v\cdot\nabla r)\,dx.
    $$
    Notice that
    $$
    -\int r\,(v\cdot\nabla r)\,dx = -\frac{1}{2}\int v\cdot \nabla (r^2)\,dx.
    $$
    Integrate by parts (using that $r$ decays sufficiently at infinity so that the boundary term vanishes):
    $$
    -\frac{1}{2}\int v\cdot \nabla (r^2)\,dx = \frac{1}{2}\int r^2\,\nabla\cdot v\,dx.
    $$



    ### Step 5. Combine the Terms

    The two terms on the right-hand side become:
    $$
    -\frac{1}{2}\int r^2\,\nabla\cdot v\,dx - \int r^2\,\nabla\cdot v\,dx = -\frac{1}{2}\int r^2\,\nabla\cdot v\,dx.
    $$
    Thus, we obtain
    $$
    \frac{1}{2}\frac{d}{dt}\int r^2\,dx = -\frac{1}{2}\int r^2\,\nabla\cdot v\,dx.
    $$
    Multiplying both sides by 2 yields
    $$
    \frac{d}{dt}\int r^2\,dx = -\int r^2\,\nabla\cdot v\,dx.
    $$



    ### Step 6. Use Boundedness of $\nabla\cdot v$ and Apply Gronwall’s Inequality

    Assume that the divergence of $v$ is bounded in absolute value; that is, there exists a constant $M$ such that
    $$
    \bigl|\nabla\cdot v(x,t)\bigr| \le M \quad \text{for all } x,t.
    $$
    Then,
    $$
    \frac{d}{dt}\int r^2\,dx \le M \int r^2\,dx.
    $$
    Let
    $$
    E(t) = \int_{\mathbb{R}^d} r(t,x)^2\,dx.
    $$
    We then have
    $$
    \frac{d}{dt}E(t) \le M\, E(t).
    $$
    Since $r(0,x)=0$, it follows that $E(0)=0$.

    By Gronwall’s inequality, we obtain
    $$
    E(t) \le E(0) e^{Mt} = 0.
    $$
    Thus, for all $t\ge0$,
    $$
    \int_{\mathbb{R}^d} r(t,x)^2\,dx = 0.
    $$
    Since the $L^2$ norm of $r(t,\cdot)$ is zero, we conclude that
    $$
    r(t,x) = 0 \quad \text{for almost every } x \text{ and for all } t\ge0.
    $$

!!! thm "Equivalence of probability path and velocity field"
    Let $p_t$ be a probability path and $v(x,t)$ a locally Lipchitz integrable vector field, then the following are equivalent

    1. Cotinuity Equation holds for $t\in[0,1)]$

    2. $v(x,t)$ generates $p_t$

??? proof "velocity generated probability path"

    ### 1. Flow Map
    Let $v: \mathbb{R}^d \times [0,T] \to \mathbb{R}^d$ be a smooth (or Lipschitz) velocity field. Define the flow map $X(t,x)$ as the unique solution of the ordinary differential equation (ODE)

    $$
    \begin{cases}
    \displaystyle \frac{d}{dt} X(t,x) = v\big(X(t,x),t\big),\$1mm]
    X(0,x) = x.
    \end{cases}
    $$

    Under these conditions, for each fixed $t$, the map $X(t,\cdot): \mathbb{R}^d \to \mathbb{R}^d$ is a diffeomorphism.

    ### 2 Pushforward of Measures
    Given a probability measure $p_0$ (absolutely continuous with respect to the Lebesgue measure, with density $p_0(x)$), the pushforward of $p_0$ under $X(t,\cdot)$ is defined by

    $$
    p_t = \bigl(X(t,\cdot)\bigr)_\# p_0,
    $$

    which means that for any measurable set $A \subset \mathbb{R}^d$,

    $$
    p_t(A) = p_0\Bigl(X(t,\cdot)^{-1}(A)\Bigr).
    $$

    If $X(t,\cdot)$ is a diffeomorphism, the density $p_t(x)$ is given by the change of variables formula:
    $$
    p_t(x) = \frac{p_0(y)}{\bigl|\det\bigl(D_yX(t,y)\bigr)\bigr|},\quad\text{with } x=X(t,y).
    $$


    ### 3. Weak Formulation of the Continuity Equation

     and integrating over $\mathbb{R}^d$ yields:
    $$
    \int_{\mathbb{R}^d} \partial_t p_t(x) \, f(x) \,dx + \int_{\mathbb{R}^d} \nabla\cdot\big(v(x,t)p_t(x)\big)\, f(x) \,dx = 0.
    $$

    Applying integration by parts (using the compact support of $f$) to the second term gives

    $$
    \int_{\mathbb{R}^d} \nabla\cdot\big(v(x,t)p_t(x)\big)\, f(x) \,dx = - \int_{\mathbb{R}^d} v(x,t)p_t(x)\cdot\nabla f(x)\,dx.
    $$

    Thus, the weak formulation becomes: for all $f\in C_c^\infty(\mathbb{R}^d)$,

    $$
    \frac{d}{dt}\int_{\mathbb{R}^d} p_t(x)\, f(x)\, dx = \int_{\mathbb{R}^d} v(x,t)p_t(x)\cdot\nabla f(x)\, dx.
    $$

    We denote the pairing by $\langle p_t, f\rangle$, so that

    $$
    \frac{d}{dt}\langle p_t, f\rangle = \langle p_t, v(\cdot,t)\cdot\nabla f \rangle.
    $$

    ### 4. Pushforward Density Satisfies the Weak Formulation

    Define the candidate solution by the pushforward:

    $$
    \tilde{p}_t = \bigl(X(t,\cdot)\bigr)_\# p_0.
    $$

    For any test function $f\in C_c^\infty(\mathbb{R}^d)$, by the definition of pushforward we have

    $$
    \int_{\mathbb{R}^d} f(x)\,\tilde{p}_t(x)\,dx = \int_{\mathbb{R}^d} f\bigl(X(t,x)\bigr)\,p_0(x)\,dx.
    $$

    Differentiate with respect to $t$ and apply the chain rule:

    $$
    \frac{d}{dt} f\bigl(X(t,x)\bigr) = \nabla f\bigl(X(t,x)\bigr) \cdot \frac{d}{dt} X(t,x) = \nabla f\bigl(X(t,x)\bigr)\cdot v\bigl(X(t,x),t\bigr).
    $$

    Thus,

    $$
    \frac{d}{dt}\int f(x)\,\tilde{p}_t(x)\,dx = \int_{\mathbb{R}^d} \nabla f\bigl(X(t,x)\bigr)\cdot v\bigl(X(t,x),t\bigr)\, p_0(x)\,dx.
    $$

    Changing variables using $y = X(t,x)$ (with the corresponding Jacobian) yields

    $$
    \frac{d}{dt}\int f(x)\,\tilde{p}_t(x)\,dx = \int_{\mathbb{R}^d} \nabla f(y)\cdot v(y,t)\, \tilde{p}_t(y)\,dy.
    $$

    This is exactly the weak formulation of the continuity equation:

    $$
    \frac{d}{dt}\langle \tilde{p}_t, f\rangle = \langle \tilde{p}_t, v(\cdot,t)\cdot\nabla f \rangle.
    $$


    Assuming the continuity equation has a unique solution (under suitable regularity conditions on $v$ and $p_t$), and noting that both $p_t$ and $\tilde{p}_t$ satisfy the weak formulation with the same initial condition $p_0$, we conclude that
    $$
    p_t = \tilde{p}_t \quad \text{for all } t\in[0,T].
    $$
    That is, the evolution of the probability density under the velocity field $v(x,t)$ is exactly given by the pushforward of $p_0$ under the flow $X(t,\cdot)$.

## Log Likelihood path

### **1. Introduction**
One of the advantages of using normalizing flows as generative models is that they allow the computation of exact likelihoods $\log p_1(x)$ for any $ x \in \mathbb{R}^d $. This is made possible by the **Continuity Equation**, leading to the **Instantaneous Change of Variables** formulation.

The governing ordinary differential equation (ODE) for the change in log-likelihood along a trajectory $\psi_t(x)$ is given by:

$$
\frac{d}{dt} \log p (\psi_t (x),t) = - \nabla \cdot u_t (\psi_t (x)).
$$

### **2. Derivation**

Let $p_t = p(x,t)$ incase confusino about the partial derivative and derivateive.

#### **Step 1: The Continuity Equation**
The probability density function $ p_t(x) $ evolves over time according to the **Continuity Equation**:

$$
\frac{\partial p(x,t)}{\partial t} + \nabla \cdot (p_t(x) u_t(x)) = 0.
$$

Expanding the divergence term using the product rule:

$$
\nabla \cdot (p_t(x) u_t(x)) = (\nabla p_t(x)) \cdot u_t(x) + p_t(x) (\nabla \cdot u_t(x)).
$$

Substituting this into the Continuity Equation:

$$
\frac{\partial p_t(x)}{\partial t} + (\nabla p_t(x)) \cdot u_t(x) + p_t(x) (\nabla \cdot u_t(x)) = 0.
$$

#### **Step 2: Expressing in Terms of Log Probability**
Define the **log-probability** function:

$$
\log p_t(x).
$$

Taking the total derivative along the flow:

$$
\frac{d}{dt} p_t(\psi_t(x)) = \frac{\partial p_t}{\partial t} + (\nabla p_t) \cdot \frac{d \psi_t}{dt}.
$$

Since $ x = \psi_t(x) $ follows the deterministic transport equation:

$$
\frac{d \psi_t}{dt} = u_t(\psi_t(x)),
$$

substituting gives:

$$
\frac{d}{dt} p_t(\psi_t(x)) = \frac{\partial p_t}{\partial t} + (\nabla p_t) \cdot u_t.
$$

Dividing by $ p_t(x) $ and substituting the continuity equation:

$$
\frac{d}{dt} \log p_t(\psi_t(x)) = - \nabla \cdot u_t(x).
$$

### **3. Hutchinson’s Trace Estimator for Efficient Computation**
Directly computing $ \nabla \cdot u_t(x) $, which requires evaluating the trace of the Jacobian matrix, is expensive for large dimensions. Hutchinson’s trace estimator provides an efficient approximation:

$$
\nabla \cdot u_t(x) = \mathbb{E}_Z \left[ Z^T \partial_x u_t(x) Z \right],
$$

where $ Z \sim \mathcal{N}(0, I) $. Plugging this into the integral form of log-likelihood evolution:

$$
\log p_1 (\psi_1 (x)) = \log p_0 (\psi_0 (x)) - \mathbb{E}_Z \int_0^1 \text{tr} \left[ Z^T \partial_x u_t (\psi_t (x)) Z \right] dt.
$$

This allows efficient computation using vector-Jacobian products (VJP) in autodiff frameworks.
