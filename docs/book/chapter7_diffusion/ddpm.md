# Diffusion Models

!!! note "preliminary"
    In order to have a fully understand of the diffusion model, it is recommend to have the knowledges of the following
    - Math
      - derivatives
      - partial derivatives
      - chain rule
      - multi variable function optimization
    -  Diffential Equation
       -  Ordinary Differential Equation
       -  SDE
       -  Itô Integra and Itô Lemma

## Understanding Denoising Diffusion Probabilistic Models (DDPM)

Generative models have revolutionized machine learning by enabling the creation of realistic data, from images to audio. Among these, **Denoising Diffusion Probabilistic Models (DDPM)** stand out for their simplicity, stability, and high-quality outputs. In this blog, we’ll break down the theory behind DDPMs, their training process, and the intuition that makes them work.


### What is a Diffusion Model?

Diffusion models are inspired by non-equilibrium thermodynamics. The core idea is simple: gradually destroy data by adding noise (forward process), then learn to reverse this process to generate new data (reverse process). DDPMs formalize this intuition into a probabilistic framework.

#### Explain in distribution aspect

At initial state, we have a very complex distribution $p_{data}$.

When doing the diffusion forward process (adding noise), we are actually do convolution with Guassian kernels such that the distribution becomes **infinite mixture of Gaussian distribution)
$$q(x_t)=\int q(x_t|x_0) p_{data} d x_{0}$$

We make the noise being small such that the distribution changes is small.

And gradually, the distribution is becomes more smoothly and finally converges to normal gaussian distribution.


* The goal is to learn a mapping from a **simple distribution (e.g., Gaussian)** → **complex real distribution (p_{\text{data}})**.
* Directly learning this **large distributional transformation** requires capturing extremely high-dimensional, non-linear, non-Gaussian relationships, making **stable convergence nearly impossible**.
* It's like trying to "turn white noise into a clear cat photo in one step" - too difficult.

---


* We break this large jump into **many tiny reversible steps**:

$$
p_{\text{data}}(x_0) \leftrightarrow q(x_t) \leftrightarrow \mathcal{N}(0, I)
$$

* Each small step only needs to learn "how to remove a bit of noise," making it locally linear, stable, and differentiable.
* Ultimately, this "nearly continuous" path forms a **smooth, computable manifold path** in distribution space,
  similar to an **integrable stochastic differential equation (SDE) or ODE flow**.


Diffusion models approximate a **probability flow**:

$$
\frac{dx}{dt} = f_\theta(x, t)
$$
This describes how to gradually flow from a Gaussian to the data distribution over time.
Because this flow is continuous, we can:

* **Train** using stochastic processes (SDE);
* **Generate samples** using deterministic ODEs;
* Obtain **computable likelihood estimates (score matching)**.



### The Forward Process: Gradually Adding Noise

The forward process is a fixed Markov chain that slowly corrupts data over \( T \) timesteps. Given an input \( x_0 \) (e.g., an image), we define a sequence \( x_1, x_2, \dots, x_T \), where each step adds Gaussian noise according to a schedule \( \beta_1, \beta_2, \dots, \beta_T \).

#### Mathematically

At each timestep \( t \),
$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t \mathbf{I})
$$
This means \( x_t \) is a noisy version of \( x_{t-1} \), with variance \( \beta_t \).

#### Reparameterization Trick

We can directly compute \( x_t \) from \( x_0 \) in closed form:
$$
x_t = \sqrt{\alpha_t} x_0 + \sqrt{1 - \alpha_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
$$
where \( \alpha_t = \prod_{i=1}^t (1 - \beta_i) \). This accelerates training by avoiding iterative sampling.

### The Reverse Process: Learning to Denoise

The goal is to learn a neural network \( \theta \) that reverses the forward process. Starting from noise \( x_T \sim \mathcal{N}(0, \mathbf{I}) \), the model iteratively denoises \( x_t \) to \( x_{t-1} \).

#### Reverse Distribution

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$
In practice, \( \Sigma_\theta \) is often fixed to \( \sigma_t^2 \mathbf{I} \), and the network predicts the mean \( \mu_\theta \).

#### Key Insight

Instead of directly predicting \( x_{t-1} \), the network predicts the noise \( \epsilon \) added at each step. This simplifies training and improves stability.

### Training Objective: Variational Bound

The loss function is derived from the variational lower bound (ELBO) of the log-likelihood:
$$
\mathbb{E}_{q(x_{1:T}|x_0)} \left[ \log \frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})} \right]
$$
After simplification, the objective reduces to:
$$
\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$
where \( \epsilon_\theta \) is the network predicting noise. Minimizing this loss trains the model to reverse the diffusion process.

### Intuition Behind DDPM

#### Noise as a Scaffold

By iteratively removing noise, the model refines its output from coarse to fine details, akin to an artist sketching and refining a painting.

#### Stability via Small Steps

Unlike GANs, which can suffer from mode collapse, DDPMs break generation into many small, stable denoising steps.

#### Connection to Score Matching

DDPMs implicitly learn the gradient of the data distribution (score function), guiding the denoising process toward high-probability regions.

### Sampling (Generation)

To generate data:

1. Sample \( x_T \sim \mathcal{N}(0, \mathbf{I}) \).
2. For \( t = T, \dots, 1 \):
   * Predict \( \epsilon_\theta(x_t, t) \).
   * Compute \( x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \alpha_t}} \epsilon_\theta \right) + \sigma_t z \), where \( z \sim \mathcal{N}(0, \mathbf{I}) \).

### Why DDPMs Work

* **Robust Training:** The per-timestep noise prediction task is easier to learn than modeling the full data distribution at once.
* **High Flexibility:** The iterative process allows capturing complex dependencies in data.
* **No Adversarial Training:** Unlike GANs, DDPMs avoid unstable min-max optimization.

### Limitations

* **Slow Sampling:** Generating samples requires \( T \) steps (often \( T \sim 1000 \)).
* **Fixed Noise Schedule:** Poorly chosen \( \beta_t \) can harm performance.

### Summarization
Here’s a concise summary of the **key formulas** in Denoising Diffusion Probabilistic Models (DDPM), focusing on the forward/reverse processes, noise schedules, and critical relationships:

#### **1. Forward Process (Noise Addition)**
Gradually corrupts data \( x_0 \) over \( T \) steps via a fixed Markov chain.

* **Conditional distribution**:

  $$
  q(x_t | x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t \mathbf{I}\right)
  $$

  * \( \beta_t \): Predefined noise schedule (small values increasing with \( t \)).

* **Reparameterization (direct sampling from \( x_0 \))**:
  $$
  x_t = \sqrt{\alpha_t} x_0 + \sqrt{1 - \alpha_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
  $$
  * \( \alpha_t = \prod_{i=1}^t (1-\beta_i) \): Cumulative product of \( 1-\beta_i \).
  * \( \sqrt{\alpha_t} \): Signal retention coefficient.
  * \( \sqrt{1-\alpha_t} \): Noise scaling coefficient.

#### **2. Reverse Process (Denoising)**
Learns to invert the forward process using a neural network \( \epsilon_\theta \).

* **Reverse distribution**:

  $$
  p_\theta(x_{t-1} | x_t) = \mathcal{N}\left(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 \mathbf{I}\right)
  $$

  * \( \mu_\theta \): Predicted mean, derived from noise estimate \( \epsilon_\theta \).
  * \( \sigma_t^2 \): Fixed to \( \beta_t \) (original DDPM) or \( \frac{1-\alpha_{t-1}}{1-\alpha_t}\beta_t \).

* **Mean prediction**:

  $$
  \mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\alpha_t}} \epsilon_\theta(x_t, t) \right)
  $$

#### **3. Training Objective**
Simplified loss derived from variational bound (ELBO):

$$
\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$

* \( \epsilon \sim \mathcal{N}(0, \mathbf{I}) \): Ground-truth noise added at step \( t \).
* \( \epsilon_\theta \): Neural network predicting noise in \( x_t \).

#### **4. Sampling (Generation)**
Starts from \( x_T \sim \mathcal{N}(0, \mathbf{I}) \) and iteratively denoises:
$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\alpha_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z, \quad z \sim \mathcal{N}(0, \mathbf{I})
$$

* \( \sigma_t = \sqrt{\beta_t} \): Controls stochasticity (set to 0 for deterministic sampling in DDIM).
* At \( t=1 \), the \( \sigma_t z \) term is omitted.

---

#### **5. Key Relationships**

* **Cumulative noise**: \( \alpha_t = \prod_{i=1}^t (1-\beta_i) \).
* **Signal-to-noise ratio (SNR)**: \( \text{SNR}_t = \frac{\alpha_t}{1-\alpha_t} \).
* **Noise schedule**: \( \beta_t \) is often linear or cosine-based (e.g., \( \beta_t \in [10^{-4}, 0.02] \)).

---

#### **6. Intuition**

* \( \alpha_t \): Decays monotonically, controlling how much signal remains at step \( t \).
* \( \beta_t \): Balances noise addition speed (small \( \beta_t \) = slow corruption).
* \( \epsilon_\theta \): Predicts the noise to subtract, steering \( x_t \) toward high-probability regions.

DDPMs hinge on:

1. Forward process equations for efficient training.
2. Noise prediction (\( \epsilon_\theta \)) and simplified loss.
3. Sampling via iterative denoising with \( \alpha_t, \beta_t, \sigma_t \).

DDPMs offer a elegant framework for generation by iteratively denoising data. Their simplicity, stability, and quality have made them a cornerstone of modern generative AI. In future posts, we’ll explore accelerated variants like **DDIM** and extensions like **Latent Diffusion Models (LDM)**. Stay tuned!

## References

1. Ho et al., *Denoising Diffusion Probabilistic Models* (2020).
2. Sohl-Dickstein et al., *Deep Unsupervised Learning Using Nonequilibrium Thermodynamics* (2015).

3. [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239)

4. [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)

5. [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)

6. [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)

7. [Latent Diffusion Models](https://arxiv.org/abs/2112.10752)  -

8. [Understanding Diffusion Models: A Unified Perspective](https://www.cvmart.net/community/detail/6827)
