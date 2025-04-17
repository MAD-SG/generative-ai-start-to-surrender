# Generative AI: From Start to Surrender

A comprehensive guide to understanding and implementing generative AI technologies, from fundamental concepts to state-of-the-art models.

## 📚 Documentation

Access our comprehensive documentation online:

[![Read Online](https://img.shields.io/badge/Read-Online-blue?style=for-the-badge)](https://mad-sg.github.io/generative-ai-start-to-surrender/)

## 📖 Content Overview

- **Introduction**
  - Terminology
  - Fourier Transform
  - Signal Processing
  - Statistics
  - Tutorials

- **Generation Theory**
  - Maximal Likelihood
  - Manifold Hypothesis
  - Generative Model Categories

- **Energy Based Models**
  - Score Functions
  - Sampling Methods
  - Contrastive Divergence
  - Score Matching

- **VAE & GANs**
  - VAE Introduction
  - VQ-VAE
  - From GAN to StyleGAN
  - StyleGAN Series (StyleGAN, StyleGAN2, StyleGAN3)

- **Diffusion Models**
  - Discrete Diffusion (DDPM, LDM)
  - SDE Diffusion
  - Advanced Topics

- **State-of-the-Art Models**
  - DALL·E Series
  - Stable Diffusion Series
    - Stable Diffusion XL
    - Stable Diffusion 3
    - Stable Diffusion 3.5
  - Flux.1

- **Applications**
  - Anomaly Detection
  - Deepfake Detection

## 🛠️ Development Setup

### VSCode Configuration

1. **Image Path Settings**
```json
{
    "markdown.copyFiles.destination": {
        "**/*": "${documentWorkspaceFolder}/docs/images/"
    }
}
```

2. **Markdownlint Settings**
```json
{
   "editor.codeActionsOnSave": {
      "source.fixAll.markdownlint": true
   },
   "markdownlint.config": {
        "default": true,
        "MD033": false,
        "MD049": false,
        "MD022": false
    }
}
```

### Cross-referencing

To reference specific sections:

```markdown
<a id="section-id"></a>
#### Section Title

[Reference Link](<path/to/file.md#section-id>)
```

## 🤝 Contributors

We appreciate all contributions to this project:

- [![GitHub](https://img.shields.io/badge/GitHub-Qian%20Lilong-lightgrey?logo=github&style=social)](https://github.com/tsiendragon)

## 📄 License

This repository contains both book content and source code, each licensed separately:

1. **Book Content**: Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
2. **Code**: Licensed under [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

## 🐛 Known Issues

### Markdown Rendering
- bmatrix cannot be rendered
- Use "MathJax 3 Plugin for Github" for equation rendering
- Cannot render \mathbf

- Math blocks (`$$ $$`) require line breaks before and after

