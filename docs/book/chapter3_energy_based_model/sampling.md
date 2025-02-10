# Sampling from a Distribution

Understanding the process of sampling is crucial in probability and statistics. Sampling is not merely picking random points in space; it involves generating samples that statistically resemble a given probability distribution. This is essential for tasks such as Monte Carlo simulations, statistical modeling, and machine learning.

## Key Concepts in Sampling

1. **Representation of the Distribution**: The samples should reflect the underlying probability distribution. For example, if a distribution has a high density in a specific region, more samples should appear in that region.

2. **Convergence to the Distribution**: As the number of samples increases, the empirical distribution (e.g., a histogram) should converge to the theoretical probability distribution.

3. **Applications**: Accurate sampling is vital for simulations and models where results depend on how well samples represent the true distribution.

![image](https://cdn.cognitiveseo.com/blog/wp-content/uploads/2014/03/b79459f12030f5efb4c24f44ab1178db2.png)

That is to say, we want to give higher priority to sample that have higher probability.

Next, if given a density function, how to sample it.
Unlike the VAE, Gan which directed generated an example. In the energy based model, we only have the probability density function (precisely, we have the unnormalized probability density function), how can we sample from it?

Generally, sample from a probability distribution can be done in this way

1. sample from a uniform distribution
2. map the sample to the inverse of the CDF function

than we obtained the samples that follow the given probability distribution.
Here CDF is the cumulative distribution function, that is

$$ CDF(x) = \int_{-\infty}^x p(z) \, dz $$

## sample from a simple gaussian distribution

In python, sample the gussian distribution maybe simple as the following.

```python
import torch
import torch.distributions as dist

# Sample from a standard normal distribution
sample = dist.Normal(0, 1).sample()
print(sample)
```

### Transformations from Uniform Distributions

While Gaussian distributions can be sampled directly, they can also be derived from uniform distributions using transformations like the Box-Muller method:

```python
import numpy as np

# Box-Muller transform: sampling normal distribution from uniform
u1 = np.random.uniform(0, 1)
u2 = np.random.uniform(0, 1)
z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
print(f"Samples: {z1}, {z2}")
```

This method is efficient because it avoids the need for the inverse CDF, which does not have a closed form for Gaussian distributions.

### Sampling from Complex Distributions

For complex distributions, especially with unnormalized probability density functions (as in Energy-Based Models), sophisticated methods are needed:

1. **Markov Chain Monte Carlo (MCMC)**: A broad class of algorithms for sampling from complex distributions.
2. **Langevin Dynamics**: Combines gradient descent with noise to explore probability spaces.
3. **Hamiltonian Monte Carlo (HMC)**: Uses physical dynamics to efficiently sample from high-dimensional distributions.

## Markov Chain Monte Carlo (MCMC)

MCMC is a broad class of algorithms used to draw samples from complex probability distributions, especially when direct sampling or classical numerical integration is difficult. The main idea behind MCMC is:

1. **We want samples from a target distribution**
Suppose we have a probability distribution $\pi(x)$ (often given up to a normalization constant, e.g., $\pi(x)\propto e^{-E(x)}$), and we want to estimate expectations like

$$
 \mathbb{E}_{x\sim \pi}[f(x)] \;=\; \int f(x)\,\pi(x)\,dx.
$$

In many applications (e.g., Bayesian inference), $\pi$ may be high‐dimensional or have no closed‐form normalizing constant, making direct sampling infeasible.

2. **Construct a Markov Chain whose stationary distribution is $\pi$**
MCMC methods build a Markov chain $X_0, X_1, X_2,\dots$ with a *transition rule* $X_{t+1}\sim T(\cdot\mid X_t)$. The key is to design $T$ so that if $X_t$ *is distributed* according to $\pi$, then $X_{t+1}$ is also distributed according to $\pi$. Under suitable conditions (ergodicity), the chain then *converges* to $\pi$ from a wide range of initial states, and the samples $X_0, X_1, \dots$ “mix” throughout the support of $\pi$.

3. **Samples from the chain approximate samples from $\pi$**

If the Markov chain is *ergodic* and *aperiodic*, then for large $t$, the distribution of $X_t$ is close to $\pi$. We can compute empirical averages using $\frac{1}{N}\sum_{t=1}^N f(X_t)$
to estimate $\mathbb{E}_{\pi}[f]$. The law of large numbers for Markov chains implies that, as $N\to\infty$, these empirical averages converge to the true expectation (under mild regularity conditions).

Popular MCMC approaches include:

- **Metropolis–Hastings (MH)** : Propose a new sample from a proposal distribution $q(\cdot\mid X_t)$ and accept or reject it based on a *Metropolis acceptance probability* that ensures $\pi$ is the stationary distribution.

- **Gibbs sampling** : Update each component in turn from its conditional distribution, often used when conditionals of $\pi$ are simpler than the joint.

## Hamiltonian Monte Carlo (HMC)

**Hamiltonian Monte Carlo (HMC)**  is a specialized MCMC method designed to tackle high‐dimensional sampling problems more efficiently than basic Metropolis–Hastings or Gibbs sampling, especially when $\pi(x)\propto e^{-E(x)}$ for some smooth potential $E(x)$. Its key ingredients:

1. **Incorporate “physical” dynamics**
HMC treats the target variable $x$ as a *position* in a physical system and introduces an auxiliary *momentum* variable $p$. Together, $(x,p)$ evolve according to (fictitious) Hamiltonian dynamics governed by a Hamiltonian function

$$H(x,p) = E(x) + \frac{1}{2}p^\top M^{-1} p$$

where $M$ is a mass matrix (often the identity).

2. **Hamiltonian flow**
Starting from $(x,p)$, HMC simulates the continuous‐time Hamiltonian equations:

$$
 \begin{cases}
\dot{x} \;=\; M^{-1} p,\\
\dot{p} \;=\; -\,\nabla E(x).
\end{cases}
$$

These flow equations conserve the Hamiltonian $H(x,p)$. In practice, one discretizes this flow via a *symplectic integrator* (e.g., leapfrog method), which approximates the true continuous trajectory but still preserves many beneficial geometry properties.

3. **Metropolis correction**
After simulating the Hamiltonian system for a certain number of leapfrog steps, HMC performs a *Metropolis acceptance/rejection step*:

- Propose a new state $(x^\star,p^\star)$ by integrating from $(x,p)$.

- Accept or reject based on the Metropolis probability involving the change in Hamiltonian:

$$
 \alpha \;=\; \min\Bigl(1,\;\exp\bigl[-(H(x^\star,p^\star)-H(x,p))\bigr]\Bigr).
$$

This ensures the Markov chain has $\pi(x)\cdot\mathcal{N}(p\mid 0,M)$ as its invariant distribution in the extended space.

4. **Efficient exploration**
Because Hamiltonian trajectories can travel *long distances* in the state space without random walk behavior, HMC reduces the *random walk* inefficiency often seen in simpler MCMC methods. This often leads to better *mixing* and more *decorrelated* samples, especially in high dimensions.

**Summary of HMC steps**

1. **Sample momentum**  $p\sim \mathcal{N}(0,M)$ to get $(x,p)$.

2. **Simulate Hamiltonian flow**  with a symplectic integrator (e.g., leapfrog) for a chosen number of steps $L$ and step size $\epsilon$. This yields a proposal $(x^\star, p^\star)$.

3. **Accept/Reject**  $(x^\star,p^\star)$ using the Metropolis probability $\alpha$. If accepted, set $X_{t+1}=x^\star$; else remain at $X_{t+1}=x$.

4. **Repeat**  for many iterations.

### References & Further Reading

- **MCMC in general** :
  - Chib, S. and Greenberg, E. (1995). *Understanding the Metropolis–Hastings Algorithm.* The American Statistician.

  - Gilks, W. R., Richardson, S., & Spiegelhalter, D. J. (1995). *Markov Chain Monte Carlo in Practice.* Chapman & Hall.

- **Hamiltonian Monte Carlo** :
  - Duane, S., Kennedy, A. D., Pendleton, B. J., & Roweth, D. (1987). *Hybrid Monte Carlo.* Physics Letters B.

  - Neal, R. M. (2011). *MCMC Using Hamiltonian Dynamics.* In *Handbook of Markov Chain Monte Carlo* (eds S. Brooks, et al.).

  - Betancourt, M. (2018). *A Conceptual Introduction to Hamiltonian Monte Carlo.*

In short, MCMC is the backbone of *sampling from complicated distributions* when direct sampling is infeasible. Hamiltonian Monte Carlo refines this idea by incorporating physical dynamics to *move quickly* through the space, often yielding more efficient sampling in high‐dimensional problems.

## Langevin Dynamics

#### 1. Definition

Langevin Dynamics Sampling is a sampling method based on Stochastic Differential Equations (SDE). It is used to sample from high-dimensional probability distributions $p(x) \propto e^{-E(x)}$. The core idea is to add stochastic noise to the deterministic gradient descent process, ensuring a balance between exploration and exploitation, ultimately achieving the desired distribution.

---

#### 2. Core Principles

- **Physical Perspective** : Derived from particle motion, where $E(x)$ represents the particle's potential energy (drift term), and random thermal noise (diffusion term) drives its movement.

- **Mathematical Perspective** : Designed as a random process to enable samples to converge to the target distribution $p(x)$.

- **Key Components** :

  1. **Gradient Descent** : Following the negative gradient of $E(x)$, which guides the particle toward areas of lower energy (higher probability density).

  2. **Random Noise** : Adding stochastic perturbations to prevent the algorithm from getting stuck and ensure sufficient exploration.

One also need to note that the converge to the final target distribution is not depended on the initial status, that means we can sample the initial status from any distribution.

#### 3. Discrete Formulation

**Continuous Langevin Dynamics**

$$
 dx_t = -\nabla E(x_t)dt + \sqrt{2}dW_t
$$

Where:

- $x_t$: Position of the particle at time $t$.

- $W_t$: Standard Wiener process (Brownian motion), satisfying $W_t \sim \mathcal{N}(0, t)$.

**Discrete Update Rule**  (Practical Implementation):

$$
 x_{k+1} = x_k - \epsilon \nabla E(x_k) + \sqrt{2\epsilon} \xi_k, \, \xi_k \sim \mathcal{N}(0, I)
$$

Where $\epsilon$ is the step size, controlling the trade-off between convergence speed and accuracy.

---

#### 4. Proof: Target Distribution as the Stationary Distribution

**Goal** : Prove that the stationary distribution of Langevin Dynamics is $p(x) \propto e^{-E(x)}$.

**Tool** : Fokker-Planck Equation (Describes the evolution of probability density under a stochastic process).

1. $p(x)$ is a stationary solution

According to Fokker-Planck Equation:

$$
\begin{aligned}
\frac{\partial p(x, t)}{\partial t} |_{p(x,t)=p(x)}  &= \nabla \cdot \left( p \nabla E(x) \right) + \nabla^2 p \\
& = \nabla \cdot ( p \nabla E + \nabla p) \\
& = \nabla \cdot ( p \nabla E - p \nabla E)\\
& = 0
\end{aligned}
$$

the last second equation comes from

$$
\nabla p = \nabla_x \frac{e^{-E}}{Z}= \frac{1}{Z} \nabla e^{-E} = - \frac{e^-E}{Z} \nabla E= -p\nabla E
$$

2. Prove $p(x)$ is a unique stationary distribution

Define the relative entropy

$$
H(t) = \int p(x,t) \log \frac{p(x,t)}{p(x)} d x
$$

where $p(x,t)$ is the density function in time $t$ according to the Langevin Dynamics.

Compute its time derivative, we have

$$\frac{d H}{d t} = - \int p \left| \nabla log \frac{p(x,t)}{p(x)}\right|^2 d x \leq 0$$

This shows $p(x,t)$ decreases monotonically and is zero only when $p(x,t) = p(x)$.

#### 5. Application Scenarios

1. **Statistical Sampling** : Simulates sampling from high-dimensional distributions.

2. **Bayesian Inference** : Samples from posterior distributions $p(\theta|D)$ in Bayesian models.

3. **Generative Models** : Used in methods like DDPM (Denoising Diffusion Probabilistic Models) for image synthesis.

4. **Optimization** : Combines stochastic noise with gradient-based optimization (e.g., Stochastic Gradient Langevin Dynamics, SGLD).

#### 6. Implementation Steps and Hyperparameter Tuning

**Implementation Process** :

1. **Initialization** : Start with a random initial point $x_0$.

2. **Iterative Updates** :

$$
 x_{k+1} = x_k - \epsilon \nabla \log p(x) + \sqrt{2\epsilon} \xi_k
$$

3. **Stopping Criterion** : Stop when the algorithm converges or sufficient samples are collected.

**Hyperparameter Tuning** :

1. **Step Size ($\epsilon$)** : Should ensure convergence while balancing efficiency and stability (commonly uses an annealing strategy).

2. **Noise Level** : Increase $\sqrt{2\epsilon}$ to improve exploration but avoid excessively large variances.

3. **Regularization** : Use adaptive methods (e.g., SGLD) to accelerate convergence.

**Notes** :

- May fail for ill-conditioned cases; alternative methods like Metropolis-Hastings or MALA can be used.

- Works under smoothness assumptions of the target distribution $E(x)$ (requires Lipschitz continuous gradients).

#### 7. Example Code (Python Code)

```python
import numpy as np

def langevin_sampling(grad_log_p, x0, epsilon, n_steps):
    samples = [x0]
    x = x0.copy()
    for _ in range(n_steps):
        noise = np.random.randn(*x.shape) * np.sqrt(2 * epsilon)
        x += -epsilon * grad_log_p(x) + noise
        samples.append(x)
    return np.array(samples)

# Example: Sampling from \( p(x) \propto e^{-x^2/2}, \, \text{grad_log_p}(x) = -x \)
grad_log_p = lambda x: -x  # Gradient of log-p(x)
samples = langevin_sampling(grad_log_p, x0=np.random.randn(), epsilon=0.01, n_steps=1000)
```

- Code ```experiment/langevin_dynamics_simulation.ipynb```
![LD sampling process](../../images/image-34.png)

#### 8. Comparison with Other Sampling Methods

| Method | Requires Gradients | Accept/Reject Step | Applicable Scenarios |
| --- | --- | --- | --- |
| Metropolis-Hastings | No | Yes | General-purpose, less efficient |
| HMC (Hamiltonian MC) | Yes | Yes | High efficiency for correlated distributions |
| Langevin Dynamics | Yes (Pure SDE) | No | Efficient for high-dimensional sampling |

#### Conclusion

Langevin Dynamics Sampling introduces stochasticity guided by gradients, enabling efficient sampling from high-dimensional distributions. It is a powerful tool in generative modeling, Bayesian inference, and optimization algorithms. The theoretical guarantee relies on the Fokker-Planck Equation ensuring the target distribution, while practical tuning of step size and noise level balances convergence and efficiency.

This simulation starts from a uniform distribution and converges to a Gaussian mixture, illustrating the effectiveness of Langevin dynamics in sampling from complex distributions.

## Alternative Sampling Dynamics

A question is that, why should we sample from $p(x)$ should follow LD ? Is there any other way?

He we introduce various stochastic differential equations (SDEs) designed to generate the steady-state distribution $p(x) \propto e^{-E(x)}$. By learning the appropriate potential $E(x)$, these SDE-based methods can model the target distribution, making them suitable for generative tasks, optimization, and more. Below are common dynamics and SDE models:

### 1. Underdamped Langevin Dynamics

**Definition** :

Introduces momentum $v$, modeling the dynamics of particles in a potential field. The equations are:

$$
 dx = v dt, \quad dv = -\eta v dt - \nabla E(x) dt + \sqrt{2\eta} dW_v,
$$

where $\eta > 0$ is the friction coefficient.

**Features** :

- Momentum introduces inertia, counteracting the "random walk" effect of overdamped Langevin dynamics.

- Converges to the stationary distribution $p(x) \propto e^{-E(x)}$.

**Applications** :

- Sampling methods like Hamiltonian Monte Carlo (HMC).

- Simulating particle dynamics in physical systems.

### 2. Regularized X-SDE

**Definition** :

For a smooth potential $E(x)$, introduces regularization terms. The SDE is:

$$
dx = b(x) dt + g(x) dW_x,
$$

where $p(x) \propto e^{-E(x)}$.

In the **Regularized X-SDE**  framework, the drift term $b(x)$ and diffusion term $g(x)$ are typically designed based on the energy function $E(x)$ to ensure that the process samples from the desired probability distribution $p(x) \propto e^{-E(x)}$. Their relationships are as follows:

#### 2.1. Drift Term $b(x)$

- The drift term $b(x)$ represents the deterministic component that drives the system towards lower-energy regions.

- In standard **Langevin Dynamics** , the drift is given by:

$$
 b(x) = -\nabla E(x)
$$

- However, in **Regularized X-SDE** , $b(x)$ can be modified to incorporate additional prior knowledge, regularization, or auxiliary variables to improve sampling efficiency.

#### 2.2. Diffusion Term $g(x)$

- The diffusion term $g(x)$ controls the stochasticity of the process.

- In standard Langevin Dynamics, the noise term follows:

$$
 g(x) = \sqrt{2} I
$$

where $I$ is the identity matrix.

- In **Regularized X-SDE** , $g(x)$ can be spatially varying, meaning:

$$
 g(x) = g(x)
$$

where $g(x)$ is a function that may depend on the energy landscape $E(x)$, allowing for **adaptive noise scaling** .

### 2.3. Generalized Regularized X-SDE Form

 A common form of the **Regularized X-SDE**  is

$$
 dx = -\nabla E(x) dt + g(x) dW_x
$$

where:

- $-\nabla E(x)$ ensures the process moves towards lower-energy regions.

- $g(x)$ can be tuned to balance exploration and exploitation, helping to escape local minima.

### 2.4. **Practical Adjustments**

- **Preconditioned Langevin Dynamics (PLD):**  $b(x)$ is modified as:

$$
 b(x) = - M(x) \nabla E(x)
$$

where $M(x)$ is a preconditioning matrix, improving convergence speed.

- **Adaptive Noise Scaling:**  $g(x)$ can be chosen to control the amount of stochasticity, improving mixing properties.

There are few main popular choise of $g(x)$,

#### 2.4.1 Constant Diffusion (Standard Langevin Dynamics)

$$g(x) = \sqrt(2)I$$

#### 2.4.2 Preconditioned Diffusion

$$ g(x) = \sqrt{2P(x)}$$

where $P(x)$ is a preconditioning matrix (positive definite). Some choices of $P(x)$

- Diagonal Preconditioning $P(x) = \text{diag} (\gamma_1, \gamma_2, ...,\gamma_d)$ depends on the local curvature of $E(x)$
- Inverse Hession Approximation: $P(x) \approx \left( \nabla^2 E(x) + \epsilon I \right)^{-1}$,(helps navigate high-curvature regions).

#### 2.4.3  Energy-Based Diffusion (Gradient Magnitude Scaling)

$$
 g(x) = \alpha \cdot (1 + \|\nabla E(x)\|^\beta)
$$

where $\alpha$ and $\beta$ are hyperparameters.

- **Intuition:**  Increases noise in high-energy regions, encouraging exploration.

- **Use Case:**  Helps escape local minima in multi-modal distributions.

#### 2.4.4 Annealed Langevin Dynamics (Temperature-Based Diffusion)

$$
g(x) = \sqrt{2 T(x)}
$$

where $T(x)$ is a temperature schedule such as:

$$
T(x) = \frac{T_0}{1 + \lambda t}
$$

- **Effect:**  Decreases noise over time, allowing for exploration in early iterations and exploitation in later iterations.

- **Use Case:**  Useful in Bayesian inference and generative modeling.

#### 2.4.5. Score-Based SDE (Diffusion Models)

$$
g(x) = \sigma(t)
$$

where $$\sigma(t)$$ follows a decreasing function, such as:

$$
 \sigma(t) = \sigma_{\max} \left( \frac{t}{T} \right)^{\gamma}
$$

- **Use Case:**  Used in diffusion models for generative modeling.

**Summary Table**

| Term | Standard Langevin Dynamics | Regularized X-SDE |
| --- | --- | --- |
| b(x) | $-\nabla E(x)$ | Can include regularization, preconditioning |
| $g(x)$ | $\sqrt{2} I$ | Adaptive function $b(x)$, may depend on xxx |

### 3. Time-Reversal SDE (Reverse Diffusion Process)

**Definition** :

Given a forward SDE:

$$
 dx = f(x, t) dt + g(x) dW_x,
$$

the reverse-time SDE has the form:

$$
 dx_r = f(x, t) - g(x)^2 \nabla \log p(x, t) + g(x) dW_x,
$$

where $p(x, t)$ is the time-dependent probability density.

**Applications** :

- Generative models (e.g., DDPM, score-based models).

- Denoising diffusion models for approximating $\log p(x)$.

In this method, we need to construct a path that maps from the original data distribution to some easy distribution, and then use the revserse process (as defined in the SDE) to sample from target data distribution.

### 4. Preconditioned Langevin Dynamics

**Definition** :

Introduces a preconditioning matrix $P(x)$ to accelerate convergence which is a special case of regularized X-SDE. The SDE is:

$$
 dx = -P(x) \nabla E(x) dt + \sqrt{2P(x)} dW_x,
$$

where $P(x)$ is symmetric and positive definite.

**Features** :

- Efficient sampling when $P(x)$ adapts to the geometry of $E(x)$.

- Extends to include Riemannian manifold-based methods.

Preconditioned SDE (or Preconditioned Langevin Dynamics) is used to improve **sampling efficiency**  and **convergence speed**  in stochastic optimization and generative modeling. The main idea is to introduce a preconditioning matrix $P(x)$  that adapts the sampling dynamics to the local geometry of the energy function $E(x)$. This helps navigate complex landscapes more effectively, especially when different directions in the space have different scales of variation.

**Where Does the Idea Come From?** The idea comes from **natural gradient methods**  and **Riemannian manifold-based optimization** , where the **metric tensor**  is used to adapt learning rates per dimension. In standard Langevin dynamics:

$$
dx = -\nabla E(x) dt + \sqrt{2} dW_x
$$

- The step size in each direction is **uniform** , which can lead to slow mixing if the energy landscape has anisotropic (different scaling in different directions) curvature.

- In high-dimensional spaces, some directions have strong gradients, while others are almost flat, causing inefficient exploration.
By introducing $P(x)$ , we scale the gradient and noise adaptively:

$$
dx = -P(x) \nabla E(x) dt + \sqrt{2P(x)} dW_x
$$

where $P(x)$ is a positive definite symmetric matrix that **scales updates differently in each direction** .

**Usage and Simple Example**

Let’s consider a simple case where:

$$
 P(x) = \text{diag}(p_1, p_2, ..., p_n)
$$

This means that we scale each coordinate **independently** .

**Example: Sampling from an Anisotropic Gaussian**

Consider an energy function for a Gaussian distribution:

$$
 E(x) = \frac{1}{2} x^T A x
$$

where $A$ is a diagonal matrix:

$$
A = \begin{bmatrix} 10 & 0 \\ 0 & 1 \end{bmatrix}
$$

This means:

- The first dimension has **stronger curvature**  (steeper gradient).

- The second dimension has **weaker curvature**  (flatter gradient).

Using standard Langevin dynamics:

$$
dx = -A x dt + \sqrt{2} dW_x
$$

- The first component moves very slowly due to large gradients.

- The second component moves faster, leading to inefficient exploration.

**Applying Preconditioning**

Now, we apply a preconditioner:

$$
P(x) = A^{-1} = \begin{bmatrix} 0.1 & 0 \\ 0 & 1 \end{bmatrix}
$$

which leads to:

$$
dx = - P(x) A x dt + \sqrt{2P(x)} dW_x
$$

Since $P(x) A = I$, the dynamics become:

$$
dx = - x dt + \sqrt{2P(x)} dW_x
$$

which **normalizes the scales in all directions** , making the sampling **more efficient**

### Summary of SDE Models

| SDE Type | Core Concept | Advantages | Applications |
| --- | --- | --- | --- |
| General Langevin Dynamics | Basic diffusion model | Theoretical clarity | Generative modeling |
| Underdamped Langevin Dynamics | Adds momentum | Faster convergence | Sampling, optimization |
| Preconditioned Langevin Dynamics | Preconditioning matrix for geometry | Accelerates convergence | Bayesian inference |
| Coupled SDE-ODE | Balance of stochastic/deterministic | Stable dynamics | Scientific computing |
| Reverse-Time SDE | Diffusion process reversal | High-quality generation | Generative models |

### Learning Objectives

Why do different SDE formulations yield the same equilibrium distribution?

**Mathematical Tool** :

**Fokker-Planck equation**

For the SDE $dx = b(x) dt + \sigma(x) dW_x$, the Fokker-Planck equation is:

$$
 \frac{\partial p}{\partial t} = -\nabla \cdot (p b) + \frac{1}{2} \nabla^2(p \sigma \sigma^\top),
$$

where $p(x, t)$ is the probability density, $b(x)$ the drift term, and $\sigma(x)$ the diffusion term. Stationary solution $p(x)$ satisfies:

$$
 \nabla \cdot (p b) = \frac{1}{2} \nabla^2(p \sigma \sigma^\top).
$$

By adjusting $b(x)$ and $\sigma(x)$, one can achieve the desired stationary distribution.

### Practical Recommendations

1. Choose the SDE based on your specific problem and computational constraints.

2. For generative tasks, consider time-reversal SDEs or preconditioned Langevin dynamics.

3. When optimizing, balance between exploration (stochastic) and exploitation (deterministic).

## Details of Annealed Langevin Dynamics Sampling

### 1. Definition

Annealed Langevin Dynamics Sampling is a stochastic sampling method that incorporates temperature annealing. It enhances the system's ability to explore the target space, balancing between exploration (searching globally) and exploitation (refining locally).

### 2. Core Principles

- **Annealing** : Introduces a temperature schedule $T(t)$, which controls the degree of randomness (higher temperatures lead to larger noise, encouraging exploration; lower temperatures reduce noise for refined optimization).

- **Stationary Distribution** : Gradually transitions the sampling distribution towards the target distribution $p(x) \propto e^{-E(x)}$ by reducing the temperature.

### 3. Mathematical Formulation and Derivation

The dynamic form of Annealed Langevin Dynamics is:

$$
 dx_t = -\nabla U(x_t) dt + \sqrt{2 T(t)} dW_t,
$$

where:

- $T(t)$: Temperature schedule (e.g., exponential decay $T(t) = T_0 e^{-\beta t}$, or other schedules like polynomial decay $T_k = T_0 / (1 + k)$).

- Noise term: $\sqrt{2 T(t)} dW_t$ ensures randomness at each step.

The discrete iteration formula becomes:

$$
 x_{k+1} = x_k - \epsilon \nabla U(x_k) + \sqrt{2 \epsilon T_k} \xi_k, \quad \xi_k \sim \mathcal{N}(0, I),
$$

where $\epsilon$ is the step size, and $\xi_k$ is Gaussian noise.

### 4. Temperature Scheduling Design

- **Exponential Decay** : $T_k = T_0 \cdot \gamma^k$, where $0 < \gamma < 1$.

- **Polynomial Decay** : $T_k = T_0 / (1 + k)^\alpha$, where $\alpha > 0$.

- **Adaptive Scheduling** : Adjust $T_k$ based on the sampling performance (e.g., adaptively reduce noise if the acceptance rate decreases).

Key parameters:

- **Initial Temperature**  $T_0$ : Chosen high enough to explore the search space widely.

- **Final Temperature** : Chosen low enough to focus on the target distribution.

- **Step Size**  $\epsilon$ : Adjusted to balance accuracy and computational cost.

---

### 5. Theoretical Analysis
In the presence of temperature $T$, the stationary distribution of Langevin Dynamics becomes:

$$
 p_T(x) \propto e^{-E(x)/T}.
$$

As $T \to 0$, the distribution transitions to $p(x) \propto e^{-E(x)}$ (i.e., the target distribution).

**Convergence Analysis** :

- **Exploration** : High temperatures help explore distant regions of the space.

- **Exploitation** : Gradual cooling refines the search towards the target distribution.

### 6. Applications

- **Generative Models** : Used in training and optimization tasks for high-dimensional data.

- **Non-Convex Optimization** : Combines annealing and dynamics for escaping local minima.

### 7. Practical Steps

1. **Initialization** :

- Set the initial temperature $T_0$, annealing schedule, step size $\epsilon$, and random initialization of $x_0$.

2. **Iteration** :

- Compute $\nabla U(x_k)$.

- Update $T_k$ based on the annealing schedule.

- Perform the update:

$$
 x_{k+1} = x_k - \epsilon \nabla U(x_k) + \sqrt{2 \epsilon T_k} \xi_k.
$$

3. **Convergence** :

- Stop when the temperature reaches a specified threshold or convergence criteria are met.

**Python Code Example** :

```python
import numpy as np

def annealed_langevin(grad_U, x0, T0, epsilon, n_steps, schedule='exponential', gamma=0.99):
    x = x0.copy()
    samples = [x0]
    T = T0
    for k in range(n_steps):
        if schedule == 'exponential':
            T = T0 * gamma ** k
        elif schedule == 'polynomial':
            T = T0 / (1 + k)
        noise = np.random.randn(x.shape) * np.sqrt(2 * epsilon * T)
        x = x - epsilon * grad_E(x) + noise
        samples.append(x)
    return np.array(samples)
```

### 8. Comparison with Other Methods

| Method | Annealing Strategy | Noise Term | Suitable Scenarios |
| --- | --- | --- | --- |
| Standard Langevin Dynamics | None | Yes | Single-mode stationary distributions |
| Annealed Langevin Dynamics | Yes | Yes | Multi-modal distributions, generative models |
| Simulated Annealing | Yes | No | Combinatorial optimization, discrete space |
| Hamiltonian Monte Carlo (HMC) | None | No | High-dimensional, parametric distributions |

### 9. Advanced Extensions

- **SGLD with Annealing** :
Combines stochastic gradient Langevin dynamics with annealing for large-scale datasets.

- **Adaptive Annealing** :
Dynamically adjusts the annealing schedule based on feedback (e.g., acceptance rate, energy variance).

---

### 10. Key Considerations

- **Temperature Schedule** : Improper schedules (e.g., too rapid cooling) may lead to poor convergence.

- **Step Size Selection** : Balances convergence speed with numerical stability.
