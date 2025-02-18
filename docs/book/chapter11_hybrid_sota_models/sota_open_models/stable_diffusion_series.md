# Stable Diffusion Series
## VQ-GAN

- Year: 2020 Dec - 2022 Jan
- Paper:
  - [Taming Transformers for High-Resolution Image Synthesis](https://compvis.github.io/taming-transformers/)
- Repo: [taming-transformers](https://github.com/CompVis/taming-transformers)
- Organization: CompVis

Please refer [VQ-GAN](../../chapter5_GAN/vq_gan.md) for more details.

## Stable Diffusion v0

- Year: Dec 2021 -Nov 2022
- Paper: [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- Repo: [<https://github.com/Stability-AI/stablediffusion?tab=readme-ov-file>](https://github.com/CompVis/latent-diffusion)
- Organization: CompVis

Please refer [LDM](../../chapter7_diffusion/ldm.md) for more details

## Stable DIffusion v1

- Year
- Ideas
  - [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
  - [classifier free guidance sampling](https://arxiv.org/pdf/2207.12598)
- Repo: [Stable_Diffusion_v1_Model_Card.md](https://github.com/CompVis/stable-diffusion/blob/main/Stable_Diffusion_v1_Model_Card.md)
- Organization: CompVis
### Summary:
- **Architecture:**
  - A *latent diffusion model* that combines an autoencoder with a diffusion model operating in the autoencoder’s latent space.
  - **Image Encoding:** Images are downsampled by a factor of 8, converting an image from shape H x W x 3 to a latent representation of shape H/8 x W/8 x 4.
  - **Text Conditioning:** Uses a ViT-L/14 text encoder; its non-pooled output is integrated into the UNet backbone via cross-attention.

- **Training Objective:**
  - The model is trained to reconstruct the noise added to the latent representations, essentially predicting the noise in the latent space.

- **Training Data:**
  - Primarily trained on LAION-5B and various curated subsets, including:
    - *laion2B-en*
    - *laion-high-resolution* (for high-resolution images)
    - *laion-aesthetics v2 5+* (filtered for aesthetics and watermark probability)

- **Checkpoints Overview:**
  - **sd-v1-1.ckpt:**
    - 237k steps at 256x256 resolution (laion2B-en)
    - 194k steps at 512x512 resolution (laion-high-resolution)
  - **sd-v1-2.ckpt:**
    - Continued from v1-1; 515k steps at 512x512 using laion-aesthetics v2 5+ data.
  - **sd-v1-3.ckpt & sd-v1-4.ckpt:**
    - Both resumed from v1-2 with additional 10% text-conditioning drop to improve classifier-free guidance sampling.

- **Training Setup:**
  - **Hardware:** 32 x 8 x A100 GPUs
  - **Optimizer:** AdamW
  - **Batch Details:** Gradient accumulations and batch size set to a total of 2048 images per update
  - **Learning Rate:** Warmup to 0.0001 over 10,000 steps, then kept constant
### Dataset
#### LAION-Aesthetics Dataset Summary

- **Overview:**
  - A curated subset of the larger LAION image-text dataset that emphasizes high-quality, visually appealing images.
  - Utilizes a deep learning–based aesthetic predictor to assign scores reflecting the perceived visual quality of each image.

- **Filtering Process:**
  - **Aesthetic Scoring:** Images are evaluated with the LAION-Aesthetics Predictor, and only those exceeding a certain score threshold (e.g., >5.0) are selected.
  - **Additional Filters:**
    - Ensures images have a minimum resolution (original size ≥ 512×512).
    - Applies a watermark probability filter to exclude images with a high likelihood of watermarks.

- **Purpose and Applications:**
  - Designed to serve as high-quality training data for generative models, such as Stable Diffusion.
  - Aims to improve the aesthetic quality of generated images by providing models with visually appealing training examples.

#### LAION-5B Dataset
- Massive Scale: Contains around 5 billion image-text pairs scraped from the internet.
- Diversity: Offers a broad spectrum of visual content and associated textual descriptions.
- Purpose: Designed to power large-scale machine learning and generative models, ensuring rich semantic variety.
- Open Access: Available for research and development, promoting transparency and innovation in
- Total size: 12 TB

## Stable Diffusion v2

- Year: Dec 2021 -Nov 2022
- Ideas:
  - /<https://arxiv.org/pdf/2204.06125>
  - <https://arxiv.org/pdf/2202.00512>
- repo: <https://github.com/Stability-AI/stablediffusion?tab=readme-ov-file>
- oganization: Stability-AI

## Stable Diffusion SDXL

- repo: <https://github.com/Stability-AI/generative-models>

## Stable Diffusion v3

- Paper: /<https://arxiv.org/pdf/2403.03206>
- Ideas

## Stable Diffusion v3.5

- repo: <https://github.com/Stability-AI/sd3.5>
