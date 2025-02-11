# Flow Matching Theorem

[toc]

!!! quote "Note"
      In this article, we establish the theroy foundation of the flow matching method

      Please read the section introduction for common knowledges of the flow matching and standard notations

## Formula of flow matching

Suppose $v(x,t)$ is the velocity field from push the probability $p$ to $q$. Our aim is to estimate the delocity field

!!! def "Flow Matching Loss"
      The flow matching loss is defined as

      $$L_{F M}=E_{X_t\sim p_t} ||v(x,t) - v_\theta(x,t)||$$

Here **$||$** could be any distance metric besides the normal mean square error.

For simplicity, we denote the $\pi_{0,1}(X_0,X_1)$ be the joint distribution of the data coupling

$$(X_0,X_1)\sim \pi_{0,1}(X_0,X_1)$$

Although, the common used distirbution of $X_0$ is the Gaussian noise, we just consider the general distribution in this section.

### Conditional probability path

Let $p_{t|1}(x|x_1)$ be the consitional probability path.

Then the marginal probability path (responde to the joint distribution $\pi_{0,1}$) $p_t$

$$\tag{1}p_t(x) = \int p_{t|1}(x|x_1) q(x_1)d x_1.$$

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

$$\tag{2}u_t(x) = \int u_t(x|x_1) p_{x_1|x} d x_1$$

This can be viewed as a weighted average of the conditional velocities or it can be regared as the conditional expectation

$$u_t(x) = \mathbf{E}[u_t(X_t|X_1)|X_t=x]$$

!!! thm "Expectation of conditional velocity"
      Given any Random Variable $Z$, $p_Z$ has bounded support,

      $$p_{t|Z}(x|z)\in C^1([0,1]\times R^d)$$

      $$u_t(x|z) \in C^1([0,1]\times R^d,R^d), p_t(x)>0 \; \forall x \in R^d, t\in[0,1)$$

      If $u_t(x|z)$ is conditional integrable and generates the conditional probability path $p_t(\cdot|z)$, then the marginal velocity field $u_t$ generates the marginal probability path $p_t$ for all $t\in [0,1)$

??? proof "Expectation of conditional velocity"

      Step 1: Differentiating $ p_t(x) $
      By the law of total probability:

      $$
      p_t(x) = \int p_{t|Z}(x|z) p_Z(z) dz.
      $$

      Differentiating both sides:

      $$
      \frac{d}{dt} p_t(x) = \int \frac{d}{dt} p_{t|Z}(x|z) p_Z(z) dz.
      $$

      Since $ u_t(x|z) $ generates $ p_{t|Z}(x|z) $, we have:

      $$
      \frac{d}{dt} p_{t|Z}(x|z) = -\nabla_x \cdot [ u_t(x|z) p_{t|Z}(x|z) ].
      $$

      Thus,

      $$
      \frac{d}{dt} p_t(x) = -\int \nabla_x \cdot\left[ u_t(x|z) p_{t|Z}(x|z) \right] p_Z(z) dz.
      $$

      Using the Leibniz rule to move differentiation outside the integral:

      $$
      \frac{d}{dt} p_t(x) = -\nabla_x \cdot \int u_t(x|z) p_{t|Z}(x|z) p_Z(z) dz.
      $$

      Step 2: Expressing $ u_t(x) $ Using **Bayes' Formula**
      From **Bayes' rule**:

      $$
      p_{t|Z}(x|z) p_Z(z) = p_{Z|t}(z|x) p_t(x),
      $$

      we substitute:

      $$
      \frac{d}{dt} p_t(x) = -\nabla_x \cdot \int u_t(x|z) p_{Z|t}(z|x) p_t(x) dz.
      $$

      Since the definition of $ u_t(x) $ is:

      $$
      u_t(x) = \int u_t(x|z) p_{Z|t}(z|x) dz,
      $$

!!! def "Contitional Flow Matching"
      Define the conditional flow matching loss as
      $$
      L_{CF M}(\theta) = E_{t,z,x_t\sim p_{t|Z}(\cdot | Z)} || u_t(x_t|Z)-v_\theta(x_t,t)||
      $$

!!! thm "Equivalent of gradient of FM and CFM loss"
      The gradients of the Flow Matching loss and the Conditional Flow Matching loss coincide. In particular, the minimizer of the Conditional Flow Matching loss is the marginal velocity.

      $$\nabla_\theta L_{F M}(\theta)  = \nabla_\theta L_{CF M}(\theta)$$

??? proof "Proof"
      To prove that the **gradients of the Flow Matching (FM) loss and the Conditional Flow Matching (CFM) loss coincide**, i.e.,

      We differentiate the CFM loss:

      $$ \nabla_\theta L_{CF M}(\theta) =E_{t,x,x_t} [ 2 (u_t(x_t | x) - v_\theta(x_t, t)) \cdot \nabla_\theta v_\theta(x_t, t) ].
      $$

      Using the **law of total expectation**, we can rewrite this expectation over the joint distribution:

      $$
      E_{t, x_t} \left[ 2 \mathbb{E}_{x | x_t} [u_t(x_t | x)] - v_\theta(x_t, t) \cdot \nabla_\theta v_\theta(x_t, t) \right].
      $$

      By the **definition of marginal velocity**,

      $$
      u_t(x_t) = \mathbb{E}_{x | x_t} [u_t(x_t | x)],
      $$

      this simplifies to:

      $$
      \nabla_\theta L_{CF M}(\theta) = \mathbb{E}_{t, x_t} \left[ 2 (u_t(x_t) - v_\theta(x_t, t)) \cdot \nabla_\theta v_\theta(x_t, t) \right].
      $$

      which is **exactly the gradient of the FM loss**:

      $$
      \nabla_\theta L(\theta) = \nabla_\theta L(\theta).
      $$

      $$\nabla_\theta L_{\text{F M}}(\theta)  = \nabla_\theta L_{\text{CF M}}(\theta)$$
