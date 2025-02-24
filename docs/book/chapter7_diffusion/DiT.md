# Scalable Diffusion Models with Transformers (DiT)

- Year 2023 Mar
- Author:
  - 威廉·皮布尔斯（William Peebles）
    - DiT model (UC Berkely)
    - SORA project (open ai)
  - 谢赛宁（Saining Xie）
    - DiT model (New York University)
    - ResNeXt
    - Cambrian-1 with Yan Lecun
- Main contribution
  - replace the U-Net backbone with transformers
  - Analyze the scalability properties

![alt text](../../images/image-80.png)

## Diffusion Transformer Design

!!! note ""
    The transformer diffusion is also trained with the latent diffusion model. Thus the transformer diffusion is desinged on the latent space.

### Patchify

- convert spatial input into a sequence of $T$ tokens
- suppose the input size if [64, 64, 4], $p=16$, then for each $16\times16$ sqaure, we flatten into the single token with a linear embedding layer. totally, we obtained $64\times 64 / (p^2) = 16$ tokens. The output shape should be [16,d]
- In the design, the patchsize  $p$ is regarded as a parameter

We can use the conv2d to do the patchify, then we can use the flatten and transpose to reshape the output.

```py3
import torch
import torch.nn as nn
# Define the patch size
patch_size = 16
# Create a convolutional layer to extract patches
# The number of output channels (embedding_dim) is a hyperparameter
embedding_dim = 768  # Example embedding dimension
patchify = nn.Conv2d(
    in_channels=4,       # Number of input channels
    out_channels=embedding_dim,  # Embedding dimension
    kernel_size=patch_size,      # Size of each patch
    stride=patch_size            # Stride to ensure non-overlapping patches
)
# Example input tensor with shape (batch_size, channels, height, width)
batch_size = 8  # Example batch size
input_tensor = torch.randn(batch_size, 4, 64, 64)  # Random input tensor
# Apply the patchify operation
patches = patchify(input_tensor)  # Shape: (batch_size, embedding_dim, 4, 4)
# Reshape patches to (batch_size, num_patches, embedding_dim)
# Here, num_patches = (64 / 16) * (64 / 16) = 4 * 4 = 16
patches = patches.flatten(2)  # Shape: (batch_size, embedding_dim, num_patches)
patches = patches.transpose(1, 2)  # Shape: (batch_size, num_patches, embedding_dim)
print(patches.shape)  # Output: torch.Size([batch_size, 16, 768])
```

### positional encoding
Similar to the normal sine-cosine positional encoding, we can also use the conv2d to do the positional encoding.

```py3
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a matrix of shape (max_len, d_model) to hold the positional encodings
        pe = torch.zeros(max_len, d_model)
        # Position indices (0, 1, 2, ..., max_len-1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Compute the div_term for the exponential function
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply sine to even indices in the array; 2i
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices in the array; 2i+1
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add a batch dimension
        pe = pe.unsqueeze(0)
        # Register pe as a buffer to prevent it from being considered a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input tensor
        x = x + self.pe[:, :x.size(1), :]
        return x
```

In transformer models, positional encoding is used to inject information about the position of tokens in a sequence, since the model itself doesn't inherently capture positional information. A common method is to use sinusoidal functions to generate these encodings. The formulas for sine and cosine positional encodings are as follows:

For a given position \( \text{pos} \) and embedding dimension \( i \):

- When \( i \) is even:

$$ \text{PE}(\text{pos}, i) = \sin\left(\frac{\text{pos}}{10000^{\frac{i}{d_{\text{model}}}}}\right) $$


- When \( i \) is odd:

$$\text{PE}(\text{pos}, i) = \cos\left(\frac{\text{pos}}{10000^{\frac{i}{d_{\text{model}}}}}\right) $$


Here, \( d_{\text{model}} \) represents the dimensionality of the model's embeddings. These functions use different frequencies to encode each dimension, allowing the model to distinguish between different positions in the sequence. This approach was introduced in the "Attention Is All You Need" paper.

<iframe width="560" height="315" src="https://www.youtube.com/embed/8DDDAT9JCu0" frameborder="0" allowfullscreen></iframe>

## Dit Block Design

The core is to design how to accept condition information like

- timestep $t$
- class label $c$
- guided text
- guided spatial information such as depth, etc.

To implement this, the paper design four different blocks to accept the condition
### In-context conditioning

Similar to the `cls` token in ViT, treat the embeddings of $t$ and $c$ as two additional tokens in the input sequence, which is

$$
[e_t, e_c, patch_1, patch_2, ...,patch_N]
$$


Then the input sequence is passed through a transformer encoder to extract context information.

### Cross Attenstion Block

Similar in the Latent Diffusion [latent diffusion handson](./ldm_handson.md)

<img src="../../images/image-81.png" width="200" />

![](https://media.licdn.com/dms/image/v2/D5612AQG0az4xyd5ODQ/article-cover_image-shrink_600_2000/article-cover_image-shrink_600_2000/0/1687101703871?e=2147483647&v=beta&t=5dL9K_PU8JZedBPh67hVx71LDc_K-GND1Y5vB8i0sOA)

The $t$ and $c$ are concated as lenght-two sequence, and act as $K$ and $V$ in the cross attention block as show in the above picture.
