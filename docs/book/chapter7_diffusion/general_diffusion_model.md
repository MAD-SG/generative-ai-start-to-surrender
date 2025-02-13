
# General Diffusion Theory

## 1. General Diffusion Theory

Diffusion models, such as DDPM (Denoising Diffusion Probabilistic Models), reformulate complex data generation as a gradual denoising process. The core idea is to define two processes:

### 1.1 The Forward (Diffusion) Process

- **Process Description**
  The forward process is a fixed Markov chain that gradually adds noise to the original data \( x_0 \) until it eventually resembles pure noise. At each time step \( t \), the sample \( x_t \) is obtained by:
  $$
  q(x_t \mid x_{t-1}) = \mathcal{N}\Big(x_t; \sqrt{1-\beta_t}\, x_{t-1},\, \beta_t \mathbf{I}\Big)
  $$
  where \(\beta_t\) is a pre-defined noise schedule.

- **Overall Process**
  After \( T \) steps, one can show that:
  $$
  q(x_T \mid x_0) = \mathcal{N}\Big(x_T; \mu_T(x_0),\, \sigma_T^2 \mathbf{I}\Big)
  $$
  When \( T \) is sufficiently large, \( x_T \) approximates a standard normal distribution, effectively losing most information about the original data.

### 1.2 The Reverse (Generative) Process

- **Process Description**
  The reverse process is parameterized by a model \( p_\theta(x_{t-1} \mid x_t) \) that aims to reverse the forward diffusion. Starting from a noise sample \( x_T \sim \mathcal{N}(0, \mathbf{I}) \), the model gradually denoises the data to recover \( x_0 \):
  $$
  p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} \mid x_t)
  $$
  Each reverse transition is typically modeled as a Gaussian:
  $$
  p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\Big(x_{t-1};\, \mu_\theta(x_t, t),\, \Sigma_\theta(x_t, t)\Big)
  $$

- **Training Objective**
  The model is trained by maximizing the log-likelihood of the data or, more commonly, by minimizing a variational lower bound (ELBO). In practice, a simplified loss is often used where the model directly predicts the noise added to the data.

### 1.3 Theoretical Insights

- **Stability in Generation:**
  The gradual denoising in the reverse process simplifies the learning task to removing noise incrementally, which contributes to stable training and generation.

- **Inverse Problem Framing:**
  The diffusion process transforms data generation into solving an inverse problem, where the goal is to reconstruct the original data from noisy observations.

---

## 2. Conditional Diffusion Models

In many applications (e.g., image generation, text-to-image synthesis), it is desirable for the generated samples to adhere to certain conditions \( y \) (such as class labels, textual descriptions, or structural information). In the diffusion framework, conditional generation is typically incorporated into the reverse process:

- **Conditional Reverse Process**
  The model is modified to condition on \( y \):
  $$
  p_\theta(x_{t-1} \mid x_t, y)
  $$
  This means that at each denoising step, the model not only uses the noisy input \( x_t \) but also leverages the condition \( y \) to guide the restoration process.

- **Implementation Strategies**
  In practice, the condition \( y \) is often embedded into the network (e.g., through feature concatenation, cross-attention, or other conditioning mechanisms) so that the network learns to generate data that meets the desired criteria.

---

## 3. Classifier-Guided Diffusion

### 3.1 Basic Idea

- **Separate Classifier Training**
  In this approach, an auxiliary classifier \( p(y \mid x_t) \) is trained to predict the condition \( y \) from the noisy sample \( x_t \). Note that the classifier must be robust to varying noise levels since \( x_t \) is noisy.

- **Guiding the Generation Process**
  During sampling, the gradient of the classifier’s log-probability with respect to \( x_t \) is used to adjust the denoising direction. By applying the chain rule, one obtains:
  $$
  \nabla_{x_t} \log p(x_t \mid y) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(y \mid x_t)
  $$
  Thus, in the reverse step, an extra gradient term is incorporated:
  $$
  \tilde{\mu}_\theta(x_t, t, y) = \mu_\theta(x_t, t) + s\, \Sigma_\theta(x_t, t) \nabla_{x_t} \log p(y \mid x_t)
  $$
  Here, \( s \) is a guidance scale that balances sample diversity and conditional fidelity.

### 3.2 Advantages and Challenges

- **Advantages:**
  - Significantly improves the alignment of the generated sample with the condition \( y \).
  - Can yield high-quality, condition-consistent results in certain tasks.

- **Challenges:**
  - Requires an additional, robust classifier that can handle noisy inputs across different diffusion time steps.
  - Excessive guidance strength (i.e., a large \( s \)) may lead to instability or a reduction in sample diversity.

---

## 4. Classifier-Free Diffusion

### 4.1 Core Idea

- **Joint Training Strategy**
  Instead of relying on an external classifier, the model is trained to perform both conditional and unconditional denoising. During training, the condition \( y \) is randomly dropped (or masked) with a certain probability. This forces the network to learn two mappings:
  - The conditional noise prediction: \( \epsilon_\theta(x_t, y) \)
  - The unconditional noise prediction: \( \epsilon_\theta(x_t) \)

### 4.2 Conditional Guidance During Sampling

- **Sampling Strategy**
  At generation time, the two predictions are combined to guide the sampling:
  $$
  \hat{\epsilon}_\theta(x_t, y) = \epsilon_\theta(x_t) + s\, \bigl( \epsilon_\theta(x_t, y) - \epsilon_\theta(x_t) \bigr)
  $$
  Here, \( s \) again controls the influence of the condition. When \( s > 1 \), the effect of the condition is amplified, encouraging the generation of samples that more strongly adhere to \( y \).

### 4.3 Advantages

- **Simplified Architecture:**
  No need to train or maintain an extra classifier, resulting in a cleaner overall model design.

- **Stability:**
  Since the guidance is integrated into the same network, the generated gradients tend to be more stable, often leading to a balanced quality in the generated samples.

---

## 5. Summary

1. **General Diffusion Framework (DDPM):**
   - **Forward Process:** Incrementally adds noise to data, eventually producing a sample from a simple distribution (e.g., standard normal).
   - **Reverse Process:** Uses a neural network to gradually remove noise, reconstructing the original data from the noisy input.

2. **Conditional Diffusion Models:**
   - Conditions \( y \) are incorporated into the reverse process so that the generated samples meet specific criteria.

3. **Two Main Conditional Guidance Methods:**
   - **Classifier-Guided Diffusion:**
     - Uses a separately trained classifier to compute \( \nabla_{x_t} \log p(y \mid x_t) \) and guides the reverse process with this gradient.
   - **Classifier-Free Diffusion:**
     - Relies on a training strategy that randomly drops the condition during training, allowing the model to learn both conditional and unconditional noise predictions.
     - At sampling time, these predictions are combined to steer the generation toward the condition \( y \).

Each method has its trade-offs: classifier-guided approaches can more precisely leverage condition information but require an extra robust classifier, while classifier-free methods offer a simpler, more stable alternative that is widely used in practice.

This overview abstracts the diffusion model theory from DDPM and explains how to incorporate conditions using both classifier-guided and classifier-free strategies.
