# VQ-VAE

- paper: [VA-VAE:arxiv](https://arxiv.org/pdf/1711.00937v2)
- year: 2018
- Author Deepmind

!!! note "Note"
    Please refer basic VAE theorem for better understanding

## VAE Intrduction

In recent years, Variational Autoencoders (VAEs) and their discrete counterparts, Vector Quantized Variational Autoencoders (VQ-VAEs), have gained significant attention in the deep learning community. While both models share a common goal of learning efficient latent representations, their underlying loss functions and theoretical derivations differ notably. In this post, we summarize the key points discussed regarding these loss functions and explore the theoretical motivations behind VQ-VAE’s design.

The VAE Loss Function

The standard VAE is built upon a probabilistic framework. Its loss function consists of two main terms:

$$
\mathcal{L}_{\text{VAE}} = -\mathbb{E}_{q_\phi(z|x)}\big[\log p_\theta(x|z)\big] + D_{\text{KL}}\big(q_\phi(z|x) \,\|\, p(z)\big)
$$

- **Reconstruction Loss**:
  \(-\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]\) ensures that the decoder can accurately reconstruct the input \(x\) from the latent variable \(z\). Depending on the nature of the data (continuous or discrete), this term is instantiated as a mean squared error (MSE) or a cross-entropy loss.

- **KL Divergence Regularization**:
  \(D_{\text{KL}}(q_\phi(z|x) \| p(z))\) acts as a regularizer, forcing the approximate posterior \(q_\phi(z|x)\) (often modeled as a Gaussian with parameters \(\mu(x)\) and \(\sigma(x)\)) to be close to the prior \(p(z)\) (usually chosen as a standard normal distribution).

This formulation is derived from maximizing the Evidence Lower Bound (ELBO) on the data likelihood, providing a rigorous probabilistic foundation for learning continuous latent representations.

---

VQ-VAE: Architecture and Loss Function

## Architecture

![alt text](../../images/image-41.png)

Unlike the continuous latent spaces of VAEs, VQ-VAE employs **discrete latent representations** using vector quantization. The key difference lies in how the encoder’s output is mapped to a latent code:

- **Vector Quantization**:
  The encoder produces a continuous representation \(z_e(x)\), which is then quantized by mapping it to the closest vector \(e\) in a finite codebook.

1. The Codebook is a $K\times D$ table, corresponds to the purple $e_1,e_2,\ldots, e_K$, each is a $D$ dimension vector.
2. The encoder compresses the input image $H\times W\times 3$ to $h\times w \times D$ dimension feature map (or considered as compressed information).
3. Use the feature map to find the index of the closest feature among $e_1, ...e_K$ in the codebook for each location, which is $q(z|x)$. Here we need to consider the prior distribution $p(z)$ as the discrete distribution. $z$ can be considered as the random variable of the $e_i$ or the index of the codebook $i$. In the following section, we consider it as the index.

4. After we have the index for each location, we can replace it with the corresponding vector in the codebook which will be fed into the decoder to recover the original image.

Let $E$ and $G$ be the encoder and decoder, respectively, we have

$$z_e(x) = E(x), \hat{x} = G(z_q(x))$$

In fact, $z$ should have the same shape as $z_e(x)$ which is $h\times w$, in other words, $z$ should be a feature map of indexes. Then the probability $p(z)$ and $q(z|x)$ will be the discrete probability distribution.

Analytically, $z=(z_{ij})$ and $q(z|x) = q(z_{1,1}, z_{1,2},\ldots, z_{h,w}|x)$.

!!! note "Revision of one-hot posterior distribution"
    In the paper, it is assumed that $p(z|x)$ is the one-hot

    $$
    q(z=k|x) =
    \begin{cases}
    1 &\forall k = \argmin_j ||z_e(x) - e_j||,\\
    0 & otherwise
    \end{cases}
    $$

    This is not analytically correct since $z_e(x)$ is not one vector. More accurately, we have

    $$
    q(z_{ij}=k;i\in[1,2,...,h],j\in[1,2,...,w]|x) =
    \begin{cases}
    1 &\forall k = \argmin_j ||z_e[x]_{i,j} - e_j||,\\
    0 & otherwise
    \end{cases}
    $$

And thus we have

$$
z_q(x)_{ij} = e_k, where \; k  = \argmin_j ||z_e(x)_{i,j} - e_j||\; \forall i\in[1,2,...,h],j\in[1,2,...,w]
$$

## Loss
The loss function in VQ-VAE is composed of three terms:

$$
\mathcal{L}_{\text{VQ-VAE}} = \underbrace{\| x - \hat{x} \|^2}_{\text{Reconstruction Loss}} + \underbrace{\| \operatorname{sg}[z_e(x)] - e \|^2}_{\text{Codebook Loss}} + \beta\, \underbrace{\| z_e(x) - \operatorname{sg}[e] \|^2}_{\text{Commitment Loss}}
$$

- **Reconstruction Loss (\(\| x - \hat{x} \|^2\))**:
  Similar to VAE, this term ensures that the decoder can reconstruct the input \(x\) from the quantized latent representation.

- **Codebook Loss (\(\| \operatorname{sg}[z_e(x)] - e \|^2\))**:
  Here, the **stop-gradient operator** (\(\operatorname{sg}[\cdot]\)) prevents gradients from flowing into the encoder. This term updates the codebook vectors so that they move closer to the encoder outputs, akin to updating cluster centers in k-means clustering.

- **Commitment Loss (\(\beta\, \| z_e(x) - \operatorname{sg}[e] \|^2\))**:
  This term ensures that the encoder commits to the discrete codebook representations by penalizing large deviations between its continuous outputs and the corresponding codebook vectors. The hyperparameter \(\beta\) balances its contribution relative to the other loss components.

Unlike the VAE, the VQ-VAE loss function is not derived from a strict probabilistic model but is rather engineered based on heuristic motivations and practical considerations for training models with discrete latent variables.

## Tricks on Reconstruction Loss

The problem in the reconstruction loss is that $z_e(x)$ to $z_q(x)$ is not differentiable thus the gradient cannot be passed successfully.

To solve this, let

$$L_{reconstruct} = ||x - D(z_e(x) + sg[z_q(x) - z_e(x)])||$$

```python3 title="reconstruction loss"
L = x - decoder(z_e + (z_q - z_e).detach())
```

Theoretical Justification: Commitment Loss as a KL Divergence

A recurring question is whether the commitment loss in VQ-VAE can be theoretically derived or understood as an equivalent to the KL divergence in the VAE loss. Although VQ-VAE’s loss is largely heuristic, several insightful approximations help bridge the conceptual gap:

## Gaussian Approximation

One can imagine “softening” the hard quantization by approximating the encoder output with a Gaussian distribution:

- **Approximate Posterior**:
  $$
  q(z|x) = \mathcal{N}(z_e(x), \sigma^2 I)
  $$

- **Local Prior Centered at the Codebook Vector**:
  $$
  p(z) = \mathcal{N}(e, \sigma^2 I)
  $$

Under these assumptions, the KL divergence between \(q(z|x)\) and \(p(z)\) becomes

$$
D_{\mathrm{KL}}\Big(q(z|x) \,\|\, p(z)\Big) = \frac{1}{2\sigma^2}\|z_e(x)-e\|^2,
$$

which is proportional to the squared Euclidean distance between \(z_e(x)\) and \(e\). With proper scaling (multiplying by \(2\sigma^2\)), this shows that minimizing the commitment loss is analogous to minimizing a KL divergence term.

## Limit Arguments

Another perspective involves considering extreme limits:

- **Encoder Output as a Dirac Delta**:
  Let the posterior \(q(z|x)\) tend to a delta function \(\delta(z-z_e(x))\), reflecting the hard quantization.

- **Flat Prior Approximation**:
  Although a truly flat prior can be seen as a Gaussian with variance tending to infinity, in practice, one assumes both the posterior and prior share the same finite variance \(\sigma^2\). This shared scale ensures that the derived KL divergence maintains sensitivity to differences in the means.

In this limit, after appropriate re-scaling, the commitment loss

$$
\|z_e(x)-e\|^2
$$

effectively serves the same purpose as the KL divergence term in a VAE, enforcing that the encoder output remains close to the codebook vector.

---

Is the VQ-VAE Loss Theoretically Derived?

In contrast to the VAE’s loss function, which is rigorously derived from maximizing the ELBO, the VQ-VAE loss is primarily an engineered objective. Its components are motivated by:

- **Reconstruction Fidelity**: Ensuring that the autoencoder captures the key information of the input.
- **Discrete Representation Learning**: Employing vector quantization to yield interpretable, discrete latent codes.
- **Stabilizing Training**: Using the codebook and commitment losses to regulate the encoder’s outputs and update the codebook effectively.

While heuristic derivations using approximations and limit arguments (as described above) provide useful insights—particularly regarding the similarity between commitment loss and a properly scaled KL divergence—the overall loss function of VQ-VAE is not strictly deduced from a complete probabilistic model.

## Sampling with VQ-VAE

In VQ‑VAE, the **sampling** (or **generation**) process is typically handled by two main components:

1. A **discrete prior** (e.g., an autoregressive model) over the latent codes.
2. The **decoder**, which maps the sampled discrete codes back to the data space.

Below is an outline of how it usually works:

---

### Training a Discrete Prior

- **Why we need a prior**:
  In a standard VAE, the prior \( p(z) \) is often a continuous Gaussian, and we can easily sample \( z \) from it. However, in VQ‑VAE, the latent representation is **discrete** (a grid of codebook indices). We therefore need a model that can learn a distribution over these discrete indices.

- **How it’s done**:
  After training the VQ‑VAE (encoder, decoder, and codebook), we collect the **discrete codes** (the nearest codebook index at each spatial location) for all training samples. These codes can be viewed as sequences (or 2D arrays) of discrete tokens.
  We then train a **prior model** on these discrete sequences. Common choices include:
  - **Autoregressive models** like PixelCNN, PixelSNAIL, or Transformers (GPT-like).
  - **Masked language modeling** style approaches.

  The prior model learns \( p(\text{codes}) \), effectively capturing how different code indices co-occur in space.

---

### Sampling from the Discrete Prior

- **Sampling the latent codes**:
  Once the prior is trained, we can generate new sequences of code indices by **sampling** from it. For an autoregressive model, for example, we sample each code index one step at a time, conditioning on previously sampled indices.

- **Spatial layout**:
  Often, the latent representation in VQ‑VAE is a 2D grid (e.g., \(h \times w\) codes). If we use a PixelCNN-style prior, we sample code indices in raster-scan order (row by row, column by column). If we use a Transformer, we might flatten the 2D grid into a 1D sequence and sample tokens left to right.

---

### Decoding the Sampled Codes

- **Feed the discrete codes into the decoder**:
  After sampling a full grid of code indices (e.g., \(\{c_{ij}\}\) for \(i=1,\dots,h\) and \(j=1,\dots,w\)), we look up the **codebook embeddings** for each index and form a 2D grid of embedded vectors. This is the quantized latent representation.

- **Generate data**:
  We pass this grid of embedded vectors into the VQ‑VAE’s **decoder**. The decoder is a CNN (or other architecture) that upsamples (or transforms) the latent grid back to the original data space. The result is a **synthetic sample** in the style of the training data.

## code example

```py3

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image


class AnimeFacesDataset(Dataset):
    """
    A simple dataset wrapper for the 'jlbaker361/anime_faces_dim_128_40k' dataset.
    Adjust if the dataset structure differs from these assumptions.
    """
    def __init__(self, split="train"):
        super().__init__()
        # Load the dataset split
        self.data = load_dataset("jlbaker361/anime_faces_dim_128_40k", split=split)
        # Basic transform: convert PIL image to tensor. Add other transforms as needed.
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # The dataset typically returns a PIL image in item["image"].
        # If it’s already a PIL Image, we can directly apply self.transform.
        image = item["image"]
        image = self.transform(image)
        return image


class AnimeFacesDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the AnimeFacesDataset.
    """
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        # We only define a train dataset here, but you can also split off validation/test sets if needed.
        self.train_dataset = AnimeFacesDataset(split="train")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)


class VQVAE(pl.LightningModule):
    """
    A simple VQ-VAE implementation with:
      - A small encoder/decoder
      - A codebook (nn.Embedding)
      - Basic VQ losses: reconstruction, codebook, and commitment
    """
    def __init__(
        self,
        in_channels=3,
        hidden_channels=64,
        embedding_dim=32,
        n_embeddings=128,
        commitment_cost=0.25,
        lr=1e-3
    ):
        super().__init__()
        self.save_hyperparameters()

        # Encoder: downsamples the image and outputs embedding_dim channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 4, 2, 1),  # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 4, 2, 1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(hidden_channels, self.hparams.embedding_dim, 1)
        )

        # Decoder: upsamples back to the original image size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.hparams.embedding_dim, hidden_channels, 4, 2, 1),  # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, 2, 1),  # 64 -> 128
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, in_channels, 1)
        )

        # Codebook: nn.Embedding for vector quantization
        self.codebook = nn.Embedding(self.hparams.n_embeddings, self.hparams.embedding_dim)
        nn.init.uniform_(self.codebook.weight, -1, 1)  # Initialize codebook

    def forward(self, x):
        """
        Forward pass returns:
          z_e: The continuous encoder output.
          z_q: The quantized output (nearest codebook embeddings).
          indices: The codebook indices selected for each latent position.
        """
        # Encode
        z_e = self.encoder(x)  # (B, embedding_dim, H, W)

        # Reshape/flatten for nearest-neighbor search in codebook
        B, C, H, W = z_e.shape
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, embedding_dim)

        # Compute distances to each embedding in the codebook
        codebook_weight = self.codebook.weight  # (n_embeddings, embedding_dim)
        z_e_sq = (z_e_flat ** 2).sum(dim=1, keepdim=True)  # (B*H*W, 1)
        e_sq = (codebook_weight ** 2).sum(dim=1)           # (n_embeddings)
        # distances: (B*H*W, n_embeddings)
        distances = z_e_sq + e_sq.unsqueeze(0) - 2 * z_e_flat @ codebook_weight.T

        # Nearest embedding index for each latent vector
        indices = distances.argmin(dim=1)  # (B*H*W)

        # Quantize: get codebook vectors and reshape back
        z_q = self.codebook(indices).view(B, H, W, C).permute(0, 3, 1, 2)

        return z_e, z_q, indices

    def training_step(self, batch, batch_idx):
        """
        Computes the VQ-VAE loss, which includes:
          1) Reconstruction loss
          2) Codebook loss (MSE between z_q and stop_grad(z_e))
          3) Commitment loss (MSE between stop_grad(z_q) and z_e)
        """
        x = batch
        z_e, z_q, _ = self(x)

        # Reconstruct
        x_recon = self.decoder(z_q)
        recon_loss = F.mse_loss(x_recon, x)

        # Codebook loss: codebook vectors should match encoder output (stop-grad on encoder)
        codebook_loss = F.mse_loss(z_q.detach(), z_e)

        # Commitment loss: encoder output should commit to codebook vector (stop-grad on codebook)
        commitment_loss = F.mse_loss(z_q, z_e.detach())

        # Weighted sum of losses
        loss = recon_loss + codebook_loss + self.hparams.commitment_cost * commitment_loss

        # Logging
        self.log("train_recon_loss", recon_loss)
        self.log("train_codebook_loss", codebook_loss)
        self.log("train_commitment_loss", commitment_loss)
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    # Instantiate the DataModule
    dm = AnimeFacesDataModule(batch_size=32)

    # Create the VQ-VAE model
    model = VQVAE(
        in_channels=3,
        hidden_channels=64,
        embedding_dim=32,
        n_embeddings=128,
        commitment_cost=0.25,
        lr=1e-3
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",  # auto-detect GPU if available
        devices="auto"       # use all available GPUs if accelerator="gpu"
    )

    # Train
    trainer.fit(model, dm)
```

## References

1. [轻松理解 VQ-VAE：首个提出 codebook 机制的生成模型](https://www.zhihu.com/search?type=content&q=VQVAE)
