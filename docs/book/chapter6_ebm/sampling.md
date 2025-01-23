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

#### Langevin Dynamics

Langevin dynamics is an iterative process that uses the following equation to sample from a known distribution $p(\mathbf{x})$:

$$
\mathbf{x}_{t+1} = \mathbf{x}_t + \tau \nabla_x \log p(\mathbf{x}_t) + \sqrt{2\tau}\mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(0, \mathbf{I}),
$$

where $\tau$ is the step size, and $\mathbf{x}_0$ is initialized with white noise.

```python
import torch
import torch.nn as nn

# Langevin dynamics sampling function
def langevin_dynamics(energy_function, initial_samples, n_steps=1000, step_size=0.1):
    x = initial_samples.clone().requires_grad_(True)
    for i in range(n_steps):
        energy = energy_function(x)
        grad = torch.autograd.grad(energy.sum(), x)[0]
        noise = torch.randn_like(x) * np.sqrt(2 * step_size)
        x.data += -step_size * grad + noise
    return x.detach()

# Example energy function
def example_energy_function(x):
    return 0.5 * ((x - 2)**2) + 0.5 * ((x + 2)**2)

# Initialize samples
initial_x = torch.zeros(1000, 1)
# Sample using Langevin dynamics
samples = langevin_dynamics(example_energy_function, initial_x)
```

### Intuition Behind Langevin Dynamics

1. **Drift Term $ \nabla_x \log p(\mathbf{x}) $**: Acts as a force guiding particles toward high-probability regions.
2. **Noise Term**: Introduces randomness, allowing exploration and preventing particles from getting stuck.
3. **Balance Between Drift and Noise**: Ensures thorough exploration of the distribution space.
4. **Convergence**: Over time, the particle distribution converges to the target distribution.

In the following sections, we will discuss the Langevin dynamics in continous form and the proof that is converged to the unerline distribution.

### Simulation Example

The following Python code simulates Langevin dynamics for a Gaussian mixture:

```python
import numpy as np
import matplotlib.pyplot as plt

# Mixture parameters
pi1, mu1, sigma1 = 0.6, 2.0, 0.5
pi2, mu2, sigma2 = 0.4, -2.0, 0.2

# PDF and gradient functions
def gaussian_pdf(x, mu, sigma):
    return (1.0 / (np.sqrt(2.0 * np.pi) * sigma) * np.exp(-0.5 * ((x - mu)/sigma)**2))

def mixture_pdf(x):
    return pi1 * gaussian_pdf(x, mu1, sigma1) + pi2 * gaussian_pdf(x, mu2, sigma2)

def grad_log_mixture_pdf(x):
    n1 = gaussian_pdf(x, mu1, sigma1)
    n2 = gaussian_pdf(x, mu2, sigma2)
    numerator = (pi1 * n1 * (-(x - mu1) / sigma1**2) + pi2 * n2 * (-(x - mu2) / sigma2**2))
    denominator = pi1 * n1 + pi2 * n2 + 1e-16
    return numerator / denominator

# Langevin dynamics parameters
M = 10_000
np.random.seed(1234)
x_min, x_max = -3.0, 3.0
x = np.random.uniform(x_min, x_max, size=M)
eta = 0.01
n_steps = 100
plot_times = [0, 1, 10, 100]
sample_snapshots = {0: x.copy()}

# Perform Langevin dynamics
for t in range(1, n_steps+1):
    grad = grad_log_mixture_pdf(x)
    x = x + eta*grad + np.sqrt(2*eta)*np.random.randn(M)
    if t in plot_times:
        sample_snapshots[t] = x.copy()

# Plot results
plt.figure(figsize=(12, 3))
for i, t in enumerate(plot_times):
    plt.subplot(1, len(plot_times), i+1)
    plt.hist(sample_snapshots[t], bins=50, density=True, alpha=0.7, color='orange')
    grid = np.linspace(-4, 5, 400)
    pdf_vals = mixture_pdf(grid)
    plt.plot(grid, pdf_vals, 'r-', lw=2)
    plt.title(f"t = {t}")
    plt.ylim(0, 0.9)
plt.suptitle("Langevin dynamics sampling from a 1D Gaussian mixture")
plt.tight_layout()
plt.show()
```
- Code ```experiment/langevin_dynamics_simulation.ipynb```

This simulation starts from a uniform distribution and converges to a Gaussian mixture, illustrating the effectiveness of Langevin dynamics in sampling from complex distributions.

## Markov Chain Monte Carlo (MCMC)

MCMC is a broad class of algorithms used to draw samples from complex probability distributions, especially when direct sampling or classical numerical integration is difficult. The main idea behind MCMC is:

1. **We want samples from a target distribution**
Suppose we have a probability distribution $\pi(x)$ (often given up to a normalization constant, e.g., $\pi(x)\propto e^{-U(x)}$), and we want to estimate expectations like

$$
 \mathbb{E}_{x\sim \pi}[f(x)] \;=\; \int f(x)\,\pi(x)\,dx.
$$

In many applications (e.g., Bayesian inference), $\pi$ may be high‐dimensional or have no closed‐form normalizing constant, making direct sampling infeasible.

2. **Construct a Markov Chain whose stationary distribution is $\pi$**
MCMC methods build a Markov chain $X_0, X_1, X_2,\dots$ with a *transition rule* $X_{t+1}\sim T(\cdot\mid X_t)$. The key is to design $T$ so that if $X_t$ *is distributed* according to $\pi$, then $X_{t+1}$ is also distributed according to $\pi$. Under suitable conditions (ergodicity), the chain then *converges* to $\pi$ from a wide range of initial states, and the samples $X_0, X_1, \dots$ “mix” throughout the support of $\pi$.

3. **Samples from the chain approximate samples from $\pi$**
If the Markov chain is *ergodic* and *aperiodic*, then for large $t$, the distribution of $X_t$ is close to $\pi$. We can compute empirical averages using$
 \frac{1}{N}\sum_{t=1}^N f(X_t)
$
to estimate $\mathbb{E}_{\pi}[f]$. The law of large numbers for Markov chains implies that, as $N\to\infty$, these empirical averages converge to the true expectation (under mild regularity conditions).

Popular MCMC approaches include:

- **Metropolis–Hastings (MH)** : Propose a new sample from a proposal distribution $q(\cdot\mid X_t)$ and accept or reject it based on a *Metropolis acceptance probability* that ensures $\pi$ is the stationary distribution.

- **Gibbs sampling** : Update each component in turn from its conditional distribution, often used when conditionals of $\pi$ are simpler than the joint.


## Hamiltonian Monte Carlo (HMC)

**Hamiltonian Monte Carlo (HMC)**  is a specialized MCMC method designed to tackle high‐dimensional sampling problems more efficiently than basic Metropolis–Hastings or Gibbs sampling, especially when $\pi(x)\propto e^{-U(x)}$ for some smooth potential $U(x)$. Its key ingredients:

1. **Incorporate “physical” dynamics**
HMC treats the target variable $x$ as a *position* in a physical system and introduces an auxiliary *momentum* variable $p$. Together, $(x,p)$ evolve according to (fictitious) Hamiltonian dynamics governed by a Hamiltonian function

$$H(x,p) = U(x) + \frac{1}{2}p^\top M^{-1} p$$

where $M$ is a mass matrix (often the identity).

2. **Hamiltonian flow**
Starting from $(x,p)$, HMC simulates the continuous‐time Hamiltonian equations:

$$
 \begin{cases}
\dot{x} \;=\; M^{-1} p,\\
\dot{p} \;=\; -\,\nabla U(x).
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
