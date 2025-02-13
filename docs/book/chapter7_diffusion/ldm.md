### **Latent Diffusion Models (LDM): A Summary**
Latent Diffusion Models (LDMs), introduced in *High-Resolution Image Synthesis with Latent Diffusion Models* (Rombach et al., 2022), are a powerful extension of diffusion models that operate in a compressed **latent space** rather than directly in pixel space. This innovation significantly improves computational efficiency while maintaining high-quality generation, particularly for high-resolution data like images.

---

#### **Core Idea**
Instead of applying diffusion processes directly to raw data (e.g., pixels), LDMs use two stages:

1. **Compression**: Train an autoencoder (VAE) to map data \( x \) (e.g., images) to a lower-dimensional latent space \( z \).
2. **Diffusion in Latent Space**: Train a diffusion model (e.g., DDPM) on the compressed latent representations \( z \).

This decoupling reduces computational costs and enables scalable high-resolution generation.

---

#### **Key Components**

1. **Autoencoder**:
   - **Encoder \( \mathcal{E} \)**: Maps input \( x \) to latent \( z = \mathcal{E}(x) \).
   - **Decoder \( \mathcal{D} \)**: Reconstructs \( x \) from \( z \), i.e., \( \tilde{x} = \mathcal{D}(z) \).
   - Trained with a reconstruction loss (e.g., perceptual loss + adversarial loss).

2. **Latent Diffusion Process**:
   - **Forward Process**: Gradually adds noise to \( z \) over \( T \) steps (like DDPM).
   - **Reverse Process**: A U-Net learns to denoise \( z_t \) in latent space, conditioned on inputs (e.g., text prompts).

3. **Conditioning Mechanism**:
   - Enables class- or text-guided generation via cross-attention layers.
   - For text-to-image: Text embeddings (e.g., CLIP) are injected into the U-Net.

---

#### **Mathematical Formulation**

- **Forward Process (Latent Space)**:
  $$
  q(z_t | z_{t-1}) = \mathcal{N}(z_t; \sqrt{1-\beta_t} z_{t-1}, \beta_t \mathbf{I})
  $$
- **Reverse Process**:
  $$
  p_\theta(z_{t-1} | z_t, c) = \mathcal{N}\left(z_{t-1}; \mu_\theta(z_t, t, c), \Sigma_\theta(z_t, t, c)\right)
  $$
  where \( c \) is the conditioning input (e.g., text).

- **Training Objective**:
  $$
  \mathcal{L} = \mathbb{E}_{z, c, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(z_t, t, c) \|^2 \right]
  $$
  The model \( \epsilon_\theta \) predicts noise in the latent space.

---

#### **Advantages**

1. **Efficiency**:
   - Operates on compressed latents (e.g., 64x64 instead of 512x512 pixels), reducing memory and compute.
2. **High-Quality Generation**:
   - Focuses diffusion on perceptually relevant features in the latent space.
3. **Flexibility**:
   - Supports conditional generation (text, class labels, segmentation maps).
4. **Scalability**:
   - Enables training on high-resolution data (e.g., 1024x1024 images).

---

#### **Applications**

- **Text-to-Image Synthesis** (e.g., Stable Diffusion).
- **Image Inpainting**, super-resolution, and style transfer.
- **Medical Imaging** and scientific data generation.

---

#### **Why LDMs Matter**
By decoupling the compression and generative modeling stages, LDMs address the computational bottleneck of pixel-space diffusion models. This makes them practical for real-world applications while retaining the quality and flexibility of diffusion processes.

---

**Reference**:
Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* (CVPR 2022).
