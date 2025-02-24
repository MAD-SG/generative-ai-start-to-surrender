# Scaling Rectified Flow Transformers for High-Resolution Image Synthesis

> Paper: <https://arxiv.org/pdf/2403.03206/>

Before reading this paper, we suggest reader to undertand the flow matching chapter first.

## Abstract

- Establish the practic for the rectified flow

## Introduction

- The forward path from data to noise is important in efficient training

  - Fails to remove all noise from the data (? now understand what this meaning)
  - affect sampling efficiency
  - curved paths required more integration steps
  - straight path has less error accumulations
- No large size experiments for class conditional rectified flow models
  - Introduce **re-weighting of the noise scales** in rectified flow models

- Text representation fussion
  - claim that text representation fed into model directly is not **ideal**
  - Introduce **new architecture** for information flow between text and image

## Preliminary

> This section revisted the flow matching scheme
