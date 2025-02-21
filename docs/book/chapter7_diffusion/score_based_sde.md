# Score-Based SDEs

## **Introduction**
Score-based Stochastic Differential Equations (SDEs) provide a **continuous-time framework** for diffusion models. They define a **forward diffusion process** (adding noise) and a **reverse process** (denoising) using **score functions**.

In this guide, we will:

- Explain different **SDE designs** including **VPSDE, VESDE, and Sub-VPSDE**.
- Show **how to construct \( x_t \) given \( x_0 \)**.
- Derive **conditional SDE score functions**.
- Implement **training, reverse sampling, and probability flow ODE sampling**.

---

## **Forward Process (Adding Noise)**
In **Score-based SDEs**, the forward process **gradually adds noise** to data:

$$
dx = f(x, t) dt + g(t) dw
$$

where:

- \( f(x, t) \) is the **drift term** (controls decay).
- \( g(t) \) is the **diffusion term** (controls noise strength).
- \( dw \) is the **Wiener process** (random noise).

At time \( t \), the noisy version of \( x_0 \) is denoted as **\( x_t \)**.

---

## **Constructing \( x_t \) from \( x_0 \)**
For any SDE, we can write the transition distribution \( p_t(x_t | x_0) \) as:

$$
p_t(x_t | x_0) = \mathcal{N}(\mu_t(x_0), \Sigma_t)
$$

where:

- **\( \mu_t(x_0) \)** is the mean (drifted input).
- **\( \Sigma_t \)** is the variance (accumulated noise).

### **VPSDE (Variance Preserving SDE)**
$$
dx = -\frac{1}{2} \beta(t) x dt + \sqrt{\beta(t)} dw
$$

Solution:
$$
x_t = x_0 e^{-\frac{1}{2} \int_0^t \beta(s) ds} + \sqrt{1 - e^{-\int_0^t \beta(s) ds}} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

### **VESDE (Variance Exploding SDE)**
$$
dx = \sigma(t) dw
$$

Solution:
$$
x_t = x_0 + \sqrt{\int_0^t \sigma^2(s) ds} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

### **Sub-VPSDE**
Sub-VPSDE is a modification of VPSDE that **controls the noise level more finely**:

$$
dx = -\frac{1}{2} \beta(t) x dt + \sqrt{\beta(t) (1 - e^{-2 \int_0^t \beta(s) ds})} dw
$$

Solution:
$$
x_t = x_0 e^{-\frac{1}{2} \int_0^t \beta(s) ds} + \sqrt{(1 - e^{-\int_0^t \beta(s) ds})} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

- **This keeps the noise level lower than VPSDE, helping preserve some structure in the data.**

---

## **Conditional Score Function**
The **score function** is:

$$
\nabla_x \log p_t(x_t | x_0) = -\Sigma_t^{-1} (x_t - \mu_t(x_0))
$$

### **For VPSDE:**
$$
\nabla_x \log p_t(x_t | x_0) = -\frac{x_t - x_0 e^{-\frac{1}{2} \int_0^t \beta(s) ds}}{(1 - e^{-\int_0^t \beta(s) ds})}
$$

### **For VESDE:**
$$
\nabla_x \log p_t(x_t | x_0) = -\frac{x_t - x_0}{\int_0^t \sigma^2(s) ds}
$$

### **For Sub-VPSDE:**
$$
\nabla_x \log p_t(x_t | x_0) = -\frac{x_t - x_0 e^{-\frac{1}{2} \int_0^t \beta(s) ds}}{(1 - e^{-\int_0^t \beta(s) ds})}
$$

- The score function is similar to VPSDE but adapted to Sub-VPSDE's noise control.

---

## **Training Loss (Score Matching)**
We train a **score network** \( s_\theta(x, t) \) to approximate the **true score function**:

$$
s_\theta(x, t) \approx \nabla_x \log p_t(x)
$$

The training loss is:

$$
\mathcal{L}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \lambda(t) \| s_\theta(x_t, t) - \nabla_x \log p_t(x_t | x_0) \|^2 \right]
$$

where \( \lambda(t) \) is a weighting function.

### **Code for Constructing \( x_t \) and Training**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define the score network
class ScoreNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x, t):
        return self.net(x)

# Construct x_t for Sub-VPSDE
def sample_xt_sub_vpsde(x0, t):
    beta_int = torch.exp(-0.5 * beta_t(t))
    noise = torch.randn_like(x0)
    return beta_int * x0 + torch.sqrt((1 - beta_int**2)) * noise, noise

# Define loss function
def score_matching_loss(model, x0):
    t = torch.rand((x0.shape[0], 1))  # Random time
    xt, noise = sample_xt_sub_vpsde(x0, t)
    score = model(xt, t)
    return ((score + noise)**2).mean()

# Training loop
model = ScoreNetwork(input_dim=2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for step in range(10000):
    x0 = torch.randn((128, 2))  # Sample data
    loss = score_matching_loss(model, x0)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 1000 == 0:
        print(f"Step {step}, Loss: {loss.item()}")
```

---

## **Reverse Sampling**
Once trained, we use **reverse SDE**:

$$
dx = \left[f(x, t) - g(t)^2 s_\theta(x, t) \right] dt + g(t) d\bar{w}
$$

### **Sampling Code (Euler-Maruyama)**

```python
def reverse_sde(model, xT, steps=1000):
    dt = -1.0 / steps
    x = xT
    for i in range(steps):
        t = torch.ones_like(x[:, :1]) * (1.0 - i / steps)
        drift = -0.5 * beta_t(t) * x - beta_t(t) * model(x, t)
        diffusion = torch.sqrt(beta_t(t)) * torch.randn_like(x)
        x = x + drift * dt + diffusion * torch.sqrt(-dt)
    return x
```

## Score function
> Understanding Score Functions and Gaussian Transition Probabilities in Score-Based SDE Models

### The Forward SDE and Its Score Function

A generalized SDE is often written as:

$$
dx = f(x,t)\,dt + g(x,t)\,dw,
$$

where \(w\) is a standard Brownian motion and \(p_t(x)\) is the marginal density at time \(t\). The score function is defined as the gradient of the log probability density:

$$
s(x,t) = \nabla_x \log p_t(x) = \frac{\nabla_x p_t(x)}{p_t(x)}.
$$

In practice, a neural network \(s_\theta(x,t)\) approximates this score function to guide the reverse diffusion process in generative models.

### Deriving the Score Function under Gaussian Transition Assumptions

Assume that the transition probability from an initial state \(x_0\) to \(x\) at time \(t\) is Gaussian:

$$
p_{t|0}(x \mid x_0) = \frac{1}{\sqrt{2\pi\,\sigma_t^2}} \exp\left(-\frac{\left(x - m(x_0,t)\right)^2}{2\sigma_t^2}\right).
$$

Given an initial density \(p_0(x_0)\), the marginal density is

$$
p_t(x) = \int p_{t|0}(x \mid x_0) \, p_0(x_0) \, dx_0.
$$

#### Marginal Score Function

Differentiating \(p_t(x)\) with respect to \(x\) and using properties of the Gaussian yields

$$
\nabla_x p_t(x) = -\frac{1}{\sigma_t^2}\int \bigl(x - m(x_0,t)\bigr)\, p_{t|0}(x \mid x_0)\, p_0(x_0) \, dx_0.
$$

Recognizing the conditional expectation

$$
\mathbb{E}[m(x_0,t) \mid x] = \frac{\int m(x_0,t) \, p_{t|0}(x \mid x_0)\, p_0(x_0) \, dx_0}{p_t(x)},
$$

the **marginal score function** becomes

$$
s(x,t) = \nabla_x \log p_t(x) = -\frac{x - \mathbb{E}[m(x_0,t) \mid x]}{\sigma_t^2}.
$$

A notable special case is when \(m(x_0,t) = x_0\), which leads to the well-known Tweedie's formula:

$$
s(x,t) = -\frac{x - \mathbb{E}[x_0 \mid x]}{\sigma_t^2}.
$$

#### Conditional Score Function

In many scenarios, it is useful to consider the score function for the **conditional probability** \(p_{t|0}(x \mid x_0)\) directly. We define the **conditional score function** as:

$$
s(x,t\mid x_0) = \nabla_x \log p_{t|0}(x \mid x_0).
$$

Since the conditional density is Gaussian,

$$
p_{t|0}(x \mid x_0) = \frac{1}{\sqrt{2\pi\,\sigma_t^2}} \exp\left(-\frac{\left(x - m(x_0,t)\right)^2}{2\sigma_t^2}\right),
$$

its log-density is

$$
\log p_{t|0}(x \mid x_0) = -\frac{1}{2}\log(2\pi\,\sigma_t^2) - \frac{\left(x - m(x_0,t)\right)^2}{2\sigma_t^2}.
$$

Taking the gradient with respect to \(x\) yields

$$
s(x,t\mid x_0) = \nabla_x \log p_{t|0}(x \mid x_0) = -\frac{x - m(x_0,t)}{\sigma_t^2}.
$$

This expression explicitly quantifies how the log-probability of \(x\) given \(x_0\) changes with \(x\) under the Gaussian assumption.

---

### Conditions for Gaussian Transition Probabilities from the Fokker–Planck Perspective

The evolution of the probability density \(p(x,t)\) is governed by the Fokker–Planck equation:

$$
\frac{\partial p(x,t)}{\partial t} = -\frac{\partial}{\partial x}\Bigl(f(x,t)\,p(x,t)\Bigr) + \frac{1}{2}\frac{\partial^2}{\partial x^2}\Bigl(g(x,t)^2\,p(x,t)\Bigr).
$$

For the Gaussian form of \(p(x,t)\) to be preserved over time, the following conditions are necessary:

1. **Linear (or Affine) Drift:**
   The drift function must be linear in \(x\):

   $$
   f(x,t) = A(t)x + b(t),
   $$

   where \(A(t)\) is a matrix (or scalar) and \(b(t)\) is a bias term. This ensures that applying the drift to a Gaussian density results in another Gaussian (or an affine-transformed Gaussian).

2. **State-Independent Diffusion:**
   The diffusion function must be independent of \(x\):

   $$
   g(x,t) = g(t).
   $$

   When the noise is additive (i.e., \(g(x,t)\) does not depend on \(x\)), the diffusion term in the Fokker–Planck equation preserves the quadratic form in \(x\) and, therefore, the Gaussian shape of the density.

For example, the Ornstein–Uhlenbeck process

$$
dx = -\lambda x\,dt + \sigma\,dw,
$$

satisfies these conditions, resulting in a Gaussian transition probability.

### Relationship Between the State Transition Matrix \(\Psi(t)\) and \(A(t)\)

For linear systems, the state transition matrix \(\Psi(t)\) (often denoted as \(\Phi(t)\) in some literature) is defined as the solution to the differential equation

$$
\frac{d}{dt}\Psi(t) = A(t)\,\Psi(t), \quad \Psi(0)=I,
$$

where \(I\) is the identity matrix. This matrix propagates the initial state \(x_0\) to the state at time \(t\) through the relation:

$$
x(t) = \Psi(t)x_0 + \int_0^t \Psi(t,s)\,b(s)\,ds + \int_0^t \Psi(t,s)\,g(s)\,dw(s).
$$

#### Closed-Forms Expression for \(\Psi(t)\)

Since the ODE for \(\Psi(t)\) is linear, it is often possible to obtain a closed-form expression for \(\Psi(t)\) under certain conditions. For example, if \(A(t)\) is time-invariant, i.e., \(A(t) = A\) for all \(t\), then the solution is given by the matrix exponential:

$$
\Psi(t) = e^{At}.
$$

Even if \(A(t)\) is time-dependent, if it commutes with itself at different times (i.e., \([A(t_1), A(t_2)] = 0\) for all \(t_1, t_2\)), the closed-form solution can be written as:

$$
\Psi(t) = \exp\left(\int_0^t A(s)\,ds\right).
$$

In cases where \(A(t)\) does not commute at different times, the closed-form expression might not be available, and one must resort to numerical integration or approximation methods.

---

### Explicit Expression for the Conditional Score Function

Under the assumptions that the drift is linear and the diffusion is state-independent, the SDE becomes

$$
dx = \bigl(A(t)x + b(t)\bigr)dt + g(t)\,dw.
$$

Its solution can be written as:

$$
x(t) = \Psi(t)x_0 + \mu(t) + \int_0^t \Psi(t,s) g(s)\, dw(s),
$$

where:

- \(\Psi(t)\) is the state transition matrix defined above,
- \(\mu(t) = \int_0^t \Psi(t,s)\,b(s)\, ds\),
- The noise integral is Gaussian with covariance

  $$
  \Sigma(t) = \int_0^t \Psi(t,s)\,g(s)^2\,\Psi(t,s)^\top ds.
  $$

Thus, the conditional (or transition) probability is given by

$$
p_{t|0}(x \mid x_0) = \mathcal{N}\Bigl(x; \Psi(t)x_0 + \mu(t),\, \Sigma(t)\Bigr).
$$

Assuming the initial distribution \(p_0(x_0)\) is also Gaussian, the marginal distribution \(p_t(x)\) remains Gaussian:

$$
p_t(x) = \mathcal{N}\Bigl(x; m(t),\, \Sigma(t)\Bigr),
$$

with mean

$$
m(t) = \Psi(t)m_0 + \mu(t),
$$

where \(m_0\) is the mean of \(p_0(x_0)\).

The **marginal score function** is computed as the gradient of the log density of a Gaussian:

$$
s(x,t) = \nabla_x \log p_t(x) = -\Sigma(t)^{-1}\bigl(x - m(t)\bigr).
$$

Recall that the **conditional score function** for \(p_{t|0}(x \mid x_0)\) is

$$
s(x,t\mid x_0) = \nabla_x \log p_{t|0}(x \mid x_0).
$$

Given the Gaussian form of \(p_{t|0}(x \mid x_0)\), we obtain

$$
s(x,t\mid x_0) = -\frac{x - (\Psi(t)x_0 + \mu(t))}{\sigma_t^2},
$$

where, in this context, \(\sigma_t^2\) relates to the covariance \(\Sigma(t)\) (or is a scalar if the state is one-dimensional). This expression quantifies how the log-probability of \(x\) given \(x_0\) changes with \(x\) under the Gaussian assumption.

---

### Example: Non-Gaussian Transition Probability

When the conditions for Gaussian transitions are not met, the SDE may yield a non-Gaussian transition probability. A classic example is **geometric Brownian motion**, where the SDE is given by

$$
dx = \mu x\,dt + \sigma x\,dw.
$$

Here, both the drift \(f(x,t) = \mu x\) and the diffusion \(g(x,t)=\sigma x\) depend linearly on \(x\). Although the drift is linear, the diffusion is state-dependent (multiplicative noise). The solution to this SDE is

$$
x(t)= x_0\,\exp\Bigl\{\left(\mu - \frac{1}{2}\sigma^2\right)t + \sigma w_t\Bigr\},
$$

and the resulting distribution of \(x(t)\) is **log-normal**, not Gaussian. This deviation occurs because the multiplicative nature of the noise distorts the Gaussian structure through a nonlinear transformation, resulting in a distribution with asymmetry (skewness) and a long tail.

### When the score function is $-\frac{x-x_0}{\sigma_t^2}$
When we say that the conditional mean is preserved, we mean that for a sample starting at \(x_0\), the mean of the transition density remains \(m(x_0,t)=x_0\) for all \(t\). In terms of the SDE,

this property requires that the drift term does not “push” the process away from its initial value in expectation. Here are several common cases with specific forms for \(f(x,t)\):

#### Zero Drift

The simplest case is when there is no drift at all. That is, set

$$
f(x,t) = 0.
$$

Then the SDE becomes a pure diffusion process

$$
dx = g(t)\,dw,
$$

and since there is no deterministic shift, we have

$$
E[x(t) \mid x_0] = x_0.
$$

#### Centered Linear Drift

Another case is to use a drift that is linear and “centered” at the initial condition. For the conditional process (i.e. given \(x(0)=x_0\)), one can choose a drift of the form

$$
f(x,t) = -a(t)\bigl(x - x_0\bigr),
$$

where \(a(t)\) is a nonnegative function (or a positive function) of time. To see why this preserves the conditional mean, define

$$
y(t)=x(t)-x_0.
$$

Then the SDE for \(y(t)\) becomes

$$
dy = -a(t)y\,dt + g(t)\,dw,
$$

with initial condition \(y(0)=0\). Since the drift term in \(y(t)\) is proportional to \(y\) and \(y(0)=0\), it follows by uniqueness and linearity of expectation that

$$
E[y(t)] = 0,
$$

which implies

$$
E[x(t) \mid x_0] = x_0.
$$

#### Symmetric (Odd) Drift Functions Around \(x_0\)

More generally, any drift function that satisfies

$$
f(x_0,t)=0 \quad \text{and} \quad f(x_0+\delta,t)=-f(x_0-\delta,t)
$$

for all small \(\delta\) and for all \(t\) will not induce a bias in the conditional mean. For example, one might choose

$$
f(x,t) = -a(t)\,\tanh\bigl(x - x_0\bigr),
$$

where \(\tanh\) is an odd function. Near \(x=x_0\) (where \(\tanh(z) \approx z\) for small \(z\)), this behaves similarly to the linear case, ensuring that \(f(x_0,t)=0\) and that the “push” is symmetric about \(x_0\). Hence, the conditional mean remains unchanged.

In summary, the conditional mean \(m(x_0,t)=x_0\) is preserved if the drift \(f(x,t)\) is chosen such that it does not introduce a net shift away from the initial condition \(x_0\). Common choices include:

- **Zero drift:** \(f(x,t)=0\).
- **Centered linear drift:** \(f(x,t) = -a(t)(x-x_0)\).
- **Symmetric (odd) drift:** For instance, \(f(x,t) = -a(t)\,\tanh(x-x_0)\).

#### VP, VE,sub-VE SDEs
Below are the answers regarding the three SDE types and their conditional score functions:

##### VP SDE (Variance Preserving SDE)

**Definition:**
The VP SDE is typically defined as

$$
dx = -\frac{1}{2}\beta(t)x\,dt + \sqrt{\beta(t)}\,dw.
$$

**Mean Behavior:**
Its solution is

$$
x(t) = e^{-\frac{1}{2}\int_0^t \beta(s)\,ds}\,x_0 + \sqrt{1-e^{-\int_0^t \beta(s)\,ds}}\,z,
$$

where \(z\sim\mathcal{N}(0,I)\). Therefore, the conditional mean is

$$
m(x_0,t) = e^{-\frac{1}{2}\int_0^t \beta(s)\,ds}\,x_0,
$$

which is not equal to \(x_0\) unless \(x_0=0\) or \(\beta(t)\) is zero. Hence, VP SDE does not preserve the mean.

**Conditional Score Function:**
Since the conditional distribution is

$$
p_{t|0}(x\mid x_0)=\mathcal{N}\Bigl(x;\, e^{-\frac{1}{2}\int_0^t \beta(s)\,ds}\,x_0,\; 1-e^{-\int_0^t \beta(s)\,ds}\Bigr),
$$

its conditional score function is

$$
s(x,t\mid x_0) = \nabla_x \log p_{t|0}(x\mid x_0) = -\frac{x - e^{-\frac{1}{2}\int_0^t \beta(s)\,ds}\,x_0}{\,1-e^{-\int_0^t \beta(s)\,ds}\,}.
$$

---

##### VE SDE (Variance Exploding SDE)

**Definition:**
The VE SDE is usually written as

$$
dx = \sqrt{\frac{d\sigma^2(t)}{dt}}\,dw.
$$

**Mean Behavior:**
Because there is no drift term, the solution is

$$
x(t) = x_0 + \sigma(t)\,z,
$$

with \(z\sim\mathcal{N}(0,I)\). Thus, the conditional mean is

$$
m(x_0,t)= x_0,
$$

i.e. the mean is preserved.

**Conditional Score Function:**
Since

$$
p_{t|0}(x\mid x_0)=\mathcal{N}\Bigl(x;\, x_0,\; \sigma^2(t)\Bigr),
$$

the conditional score function becomes

$$
s(x,t\mid x_0) = -\frac{x-x_0}{\sigma^2(t)}.
$$

##### sub-VP SDE (Sub-Variance Preserving SDE)

**Definition and Mean Behavior:**
The sub-VP SDE is designed as a reparameterization of the VP SDE to cancel the exponential decay factor in the mean. By construction, its dynamics are modified so that the conditional mean is preserved:

$$
m(x_0,t)= x_0.
$$

Although several equivalent formulations exist, a common interpretation is that the reparameterized process has a conditional distribution

$$
p_{t|0}(x\mid x_0)=\mathcal{N}\Bigl(x;\, x_0,\; \tilde{\sigma}^2(t)\Bigr),
$$

with a suitably defined variance schedule \(\tilde{\sigma}^2(t)\).

**Conditional Score Function:**
Then the conditional score function for the sub-VP SDE is

$$
s(x,t\mid x_0) = -\frac{x-x_0}{\tilde{\sigma}^2(t)}.
$$

### Summary

- **VP SDE:**
  - **Mean:** \(m(x_0,t)= e^{-\frac{1}{2}\int_0^t \beta(s)\,ds}\,x_0\) (not preserved)
  - **Conditional Score:**
    $$
    s(x,t\mid x_0) = -\frac{x - e^{-\frac{1}{2}\int_0^t \beta(s)\,ds}\,x_0}{\,1-e^{-\int_0^t \beta(s)\,ds}\,}.
    $$

- **VE SDE:**
  - **Mean:** \(m(x_0,t)= x_0\) (preserved)
  - **Conditional Score:**
    $$
    s(x,t\mid x_0) = -\frac{x-x_0}{\sigma^2(t)}.
    $$

- **sub-VP SDE:**
  - **Mean:** \(m(x_0,t)= x_0\) (by design)
  - **Conditional Score:**
    $$
    s(x,t\mid x_0) = -\frac{x-x_0}{\tilde{\sigma}^2(t)}.
    $$

### Conclusion

To summarize:

- **Score Functions:**
  - The **marginal score function** is defined as \(\nabla_x \log p_t(x)\). Under Gaussian assumptions, we derived
    $$
    s(x,t) = -\frac{x - \mathbb{E}[m(x_0,t) \mid x]}{\sigma_t^2} \quad \text{or} \quad s(x,t) = -\Sigma(t)^{-1}\bigl(x - m(t)\bigr).
    $$
  - The **conditional score function** for the transition density \(p_{t|0}(x \mid x_0)\) is
    $$
    s(x,t\mid x_0) = -\frac{x - m(x_0,t)}{\sigma_t^2},
    $$
    and under linear drift and state-independent diffusion, this becomes
    $$
    s(x,t\mid x_0) = -\frac{x - (\Psi(t)x_0 + \mu(t))}{\sigma_t^2}.
    $$

- **Gaussian Transition Probabilities:**
  The transition probability remains Gaussian if the drift is linear (or affine), \(f(x,t)=A(t)x+b(t)\), and the diffusion is state-independent, \(g(x,t)=g(t)\).

- **State Transition Matrix \(\Psi(t)\) and \(A(t)\):**
  \(\Psi(t)\) satisfies
  $$
  \frac{d}{dt}\Psi(t) = A(t)\,\Psi(t) \quad \text{with} \quad \Psi(0)=I.
  $$
  When \(A(t)\) is time-invariant, \(\Psi(t) = e^{At}\). More generally, if \(A(t)\) commutes with itself at different times, then
  $$
  \Psi(t) = \exp\left(\int_0^t A(s)\,ds\right),
  $$
  providing a closed-form expression for the state transition matrix.

- **Non-Gaussian Example:**
  When \(g(x,t)\) depends on \(x\), as in geometric Brownian motion (\(dx=\mu x\,dt + \sigma x\,dw\)), the resulting transition probability becomes log-normal rather than Gaussian.

## Extenstion Types of Score Based SDE

Beyond **VPSDE, VESDE, and Sub-VPSDE**, there are several other types of **Score-based SDEs** that modify the drift and diffusion terms to improve generation quality, stability, or computational efficiency.

Here are some additional **Score-based SDEs**:

### Critically Damped Langevin Diffusion (CLD-SDE)
This method introduces **momentum variables** to improve sampling efficiency. Unlike VPSDE/VESDE, which use only position updates, CLD-SDE includes velocity to achieve faster convergence.

#### SDE Formulation
$$
\begin{cases}
dx = v dt \\
dv = -\gamma v dt - \lambda^2 x dt + \sigma dw
\end{cases}
$$
where:

- \( x \) is the **position**.
- \( v \) is the **velocity**.
- \( \gamma \) is the **friction coefficient** (controls how fast momentum dissipates).
- \( \lambda \) is the **spring constant** (pulls data towards the center).
- \( \sigma \) is the **noise strength**.

- score function

$$
\nabla_{v_t} \log p(v_t | x_t) = -\frac{v_t + \lambda^2 x_t}{\sigma^2}
$$

- training loss
    $$|| \nabla_{v_t} \log p(v_t | x_t) - s_\theta(x,v)||^2$$
- initial condition
  - $v=0$
  - $x \sim p_{data}(x)$
- training data construction
  - use the SDE discrete formula to estimate $x_t, v_t$ and calculate the score function accordingly

#### Key Features

- **Faster sampling**: Uses both position and momentum to traverse the data manifold efficiently.
- **Inspired by Langevin dynamics**, used in Hamiltonian Monte Carlo (HMC).

### Rectified Flow SDE
Instead of traditional diffusion, **Rectified Flow SDE** designs a flow field where trajectories follow a **straight-line path** from data to noise.

#### SDE Formulation
$$
dx = (x_T - x) dt + g(t) dw
$$
where:

- \( x_T \) is the terminal (noise) state.
- \( g(t) \) controls the noise schedule.

#### Key Features

- **Deterministic reverse process**: Paths are approximately straight, reducing error in reverse sampling.
- **Faster convergence**: Uses ODE-based sampling efficiently.

#### Training Details

- Loss: MSE loss on score function
- data construction
    $$x_t =(1-t) x_0 + t x_T$$
- score function
    $$\nabla_x \log p_t (x_t|x_0) = - \frac{x_t - x_0}{\sigma_t}$$

#### Formula for \( \sigma_t \)
The variance of the noise term in the Rectified Flow SDE can be written as:

$$
\sigma_t^2 = \int_0^t g(s)^2 ds
$$

The function \( g(t) \) is designed to **minimize unnecessary randomness**, leading to **more deterministic trajectories**.

A common choice for \( g(t) \) is:

$$
g(t) = \sigma_0 \sqrt{1 - t}
$$

where \( \sigma_0 \) is a constant that determines the initial noise scale.

Thus, the variance accumulates as:

$$
\sigma_t^2 = \int_0^t \sigma_0^2 (1 - s) ds = \sigma_0^2 \left( t - \frac{t^2}{2} \right)
$$

which gives:

$$
\sigma_t = \sigma_0 \sqrt{t - \frac{t^2}{2}}
$$

This ensures that noise starts **large** at \( t=0 \) and gradually decreases to **zero** as \( t \to 1 \), making the flow almost deterministic near the final state.

Using the definition:

$$
\nabla_x \log p_t(x_t | x_0) = -\frac{x_t - x_0}{\sigma_t^2}
$$

we get:

$$
\nabla_x \log p_t(x_t | x_0) = -\frac{x_t - x_0}{\sigma_0^2 (t - \frac{t^2}{2})}
$$

which reduces to:

$$
\nabla_x \log p_t(x_t | x_0) = -\frac{x_t - x_0}{\sigma_0^2 t (1 - \frac{t}{2})}
$$

- **Noise scaling function:**
  $$
  g(t) = \sigma_0 \sqrt{1 - t}
  $$
- **Variance accumulation:**
  $$
  \sigma_t^2 = \sigma_0^2 (t - \frac{t^2}{2})
  $$
- **Score function:**
  $$
  \nabla_x \log p_t(x_t | x_0) = -\frac{x_t - x_0}{\sigma_0^2 t (1 - \frac{t}{2})}
  $$

### Continuous-Time Normalizing Flows (CTNF-SDE)

Continuous-Time Normalizing Flows (CTNF) combine **normalizing flows** with **stochastic differential equations (SDEs)**. Unlike traditional diffusion models, CTNF explicitly models the log-likelihood of the data, making it a **likelihood-based generative model**.

#### SDE Formulation
The **CTNF-SDE** is defined as:

$$
dx = f(x, t) dt + g(x, t) dw
$$

where:

- \( f(x, t) \) is a **learnable drift function**.
- \( g(x, t) \) is a **learnable diffusion function**.
- \( dw \) is a **Wiener process** (Brownian motion).
- The drift \( f(x, t) \) and diffusion \( g(x, t) \) are parameterized using neural networks.

This SDE can be **interpreted as a normalizing flow in continuous time**, where we transform a simple base distribution (e.g., Gaussian) into the data distribution.

---

#### Variance Function \( \sigma_t \)
For CTNF, the variance function is **learned** rather than fixed. It follows:

$$
\sigma_t^2 = \int_0^t g(x, s)^2 ds
$$

This means:

- \( \sigma_t \) is **data-dependent**.
- The noise schedule adapts based on the dataset.

#### Score Function
The score function is derived as:

$$
\nabla_x \log p_t(x_t | x_0) = -\frac{x_t - \mu_t}{\sigma_t^2}
$$

where:

- \( \mu_t \) and \( \sigma_t^2 \) are estimated using the learned drift and diffusion functions.

Since \( g(x, t) \) is **learned**, the score function is **not fixed** like in traditional diffusion models.

---

#### Training Loss
CTNF optimizes a **log-likelihood loss** based on the probability flow ODE:

$$
\mathcal{L}(\theta) = \mathbb{E}_{x_t} \left[ -\log p_t(x_t) \right]
$$

Alternatively, we can use **score matching**:

$$
\mathbb{E}_{x_t} \left[ || \nabla_x \log p_t(x_t | x_0) - s_\theta(x, t) ||^2 \right]
$$

---

#### Initial Condition

- \( x \sim p_{data}(x) \) (samples from the data distribution).

### Training Data Construction
Since \( x_t \) does **not** have an analytical solution, we must **numerically estimate it**:

- **Use SDE discretization**:
  $$
  x_{t+\Delta t} = x_t + f(x_t, t) \Delta t + g(x_t, t) \sqrt{\Delta t} \eta_t, \quad \eta_t \sim \mathcal{N}(0, I)
  $$
- Compute the **score function** numerically.

---

#### Summary

| **Property** | **CTNF-SDE** |
|-------------|-------------|
| **Equation** | \( dx = f(x, t) dt + g(x, t) dw \) |
| **\( \sigma_t \)** | \( \sigma_t^2 = \int_0^t g(x, s)^2 ds \) |
| **Score Function** | \( \nabla_x \log p_t(x_t\| x_0) = -\frac{x_t - \mu_t}{\sigma_t^2} \) |
| **Training Loss** | \( -\mathbb{E}_{x_t} \log p_t(x_t) \) or score matching |
| **Training Data Construction** | SDE discretization |

### Score-Based SDEs with Adaptive Noise (AN-SDE)
Instead of fixing a noise schedule, **Adaptive Noise SDE** dynamically adjusts \( g(t) \) based on data properties.

$$
dx = f(x, t) dt + \sigma(x, t) dw
$$

where:

- \( \sigma(x, t) \) is **data-dependent** noise.

### **Key Features**

- **Adapts to dataset complexity** (e.g., higher noise for high-frequency details).
- **Better preservation of structure** in images and 3D modeling.

---

## **6. Fractional Brownian Motion SDE (FBM-SDE)**
Instead of using standard **Brownian motion**, FBM-SDE incorporates **long-range dependencies**.

$$
dx = -\alpha x dt + g(t) dB^H_t
$$

where:

- \( B^H_t \) is a **fractional Brownian motion** with **Hurst parameter \( H \)**.
- \( H \) controls **memory effects** (larger \( H \) → more persistent motion).

### **Key Features**

- **Models long-range dependencies** (useful in speech, financial modeling).
- **Better generation for sequential data**.

---

## **7. Hybrid SDE-ODE Models**
Some models **combine SDE and ODE approaches** to get the best of both:

$$
dx = f(x, t) dt + g(t) dw \quad \text{for } t < T_1
$$
$$
dx = f(x, t) dt \quad \text{for } t \geq T_1
$$

where:

- The system **follows an SDE** initially (better exploration).
- The system **switches to an ODE** at a later stage (better precision).

### **Key Features**

- **Combines SDE exploration with ODE stability**.
- **More efficient sampling** compared to full SDE models.

---

## **8. Summary of Score-Based SDEs**

| **SDE Type** | **Equation** | **Key Features** |
|-------------|-------------|------------------|
| **VPSDE** | \( dx = -\frac{1}{2} \beta(t) x dt + \sqrt{\beta(t)} dw \) | Standard variance-preserving diffusion |
| **VESDE** | \( dx = \sigma(t) dw \) | Large-scale noise growth (variance exploding) |
| **Sub-VPSDE** | \( dx = -\frac{1}{2} \beta(t) x dt + \sqrt{\beta(t)(1 - e^{-2\int_0^t \beta(s) ds})} dw \) | Controlled noise decay |
| **CLD-SDE** | \( dx = v dt, \quad dv = -\gamma v dt - \lambda^2 x dt + \sigma dw \) | Faster convergence with momentum |
| **Rectified Flow SDE** | \( dx = (x_T - x) dt + g(t) dw \) | Near-deterministic straight-line flow |
| **CTNF-SDE** | \( dx = f(x, t) dt + g(x, t) dw \) | Normalizing flows + diffusion |
| **Generalized SDE** | \( dx = -f(x, t) dt + g(t) dw \) | Customizable drift and noise schedules |
| **AN-SDE** | \( dx = f(x, t) dt + \sigma(x, t) dw \) | Adaptive noise for structured data |
| **FBM-SDE** | \( dx = -\alpha x dt + g(t) dB^H_t \) | Models long-range dependencies |
| **Hybrid SDE-ODE** | \( dx = f(x, t) dt + g(t) dw \) for early \( t \), \( dx = f(x, t) dt \) later | Mixes SDE and ODE for stability |

---

## **Conclusion**
While **VPSDE and VESDE** are the most widely used Score-based SDEs, many variations **introduce optimizations** for different tasks.

- **Momentum-based SDEs (CLD-SDE)** → **Faster sampling**.
- **Straight-line diffusion (Rectified Flow)** → **Better sample paths**.
- **Hybrid SDE-ODE models** → **Efficient sampling**.
- **Adaptive SDEs (AN-SDE)** → **Noise adjustment based on data**.

!!! note "Score Based SDE vs SDE diffusion"

    SDE Diffusion (Stochastic Differential Equation-based diffusion models) and Score-based SDE (Score-based Stochastic Differential Equations) are closely related in the field of generative models, but they are not completely equivalent. Most SDE Diffusion models involve the estimation of score functions and therefore fall under the category of Score-based SDE. However, there are still some SDE Diffusion models that do not directly rely on the estimation of score functions.

    For example, the Fractional SDE-Net is a generative model for time series data with long-term dependencies. This model is based on fractional Brownian motion and captures the long-range dependency characteristics in time series by introducing fractional-order stochastic differential equations. In this approach, the model focuses on simulating the temporal dependency structure of the data rather than directly estimating the score function of the data distribution.

    Additionally, the Diffusion-Model-Assisted Supervised Learning method uses diffusion models to generate labeled data to assist in density estimation tasks in supervised learning. This method directly approximates the score function in the reverse-time SDE through a training-free score estimation method, thereby improving sampling efficiency and model performance. Although this method involves the estimation of score functions, its primary goal is to generate auxiliary data through diffusion models to enhance the supervised learning process.

    In summary, while most SDE Diffusion models fall under the category of Score-based SDE, there are still some models, such as the Fractional SDE-Net and Diffusion-Model-Assisted Supervised Learning method, that focus on other aspects, such as modeling temporal dependency structures or assisting supervised learning, without directly relying on score function estimation.
