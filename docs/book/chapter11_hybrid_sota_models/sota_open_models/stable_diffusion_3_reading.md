# [SD3] Scaling Rectified Flow Transformers for High-Resolution Image Synthesis

> [Paper: Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
](https://arxiv.org/pdf/2403.03206)

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

## Network Structure
### VAE
In the previous stable diffusion models, the VAE transform the original image of shape

$$ [H,W,3] \longrightarrow [\frac{H}{8},\frac{W}{8},d], \quad d=4$$
![alt text](../../../images/image-80.png)
Increase $d$ can improve the performance as described in the above table.

Let consider the details of the SDVAE carefully

=== "SDVAE"

    ```py3 title='SDVAE'
    class SDVAE(torch.nn.Module):
        def **init**(self, dtype=torch.float32, device=None):
            super().**init**()
            self.encoder = VAEEncoder(dtype=dtype, device=device)
            self.decoder = VAEDecoder(dtype=dtype, device=device)

        @torch.autocast("cuda", dtype=torch.float16)
        def decode(self, latent):
            return self.decoder(latent)

        @torch.autocast("cuda", dtype=torch.float16)
        def encode(self, image):
            hidden = self.encoder(image)
            mean, logvar = torch.chunk(hidden, 2, dim=1)
            logvar = torch.clamp(logvar, -30.0, 20.0)
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
    ```

    encoder predicted the mean and log variance.   And then saples by reparametrization.
    The encoder outputs a hidden representation, which is then split into a **mean** (\(\mu\)) and **log-variance** (\(\log \sigma^2\)):

    $$
    h = \text{Encoder}(x)
    $$

    $$
    \mu, \log\sigma^2 = \text{split}(h)
    $$

    The **log-variance** is clamped to prevent extreme values:

    $$
    \log \sigma^2 = \text{clamp}(\log \sigma^2, -30, 20)
    $$

    The standard deviation is computed as:

    $$
    \sigma = \exp\left(\frac{1}{2} \log \sigma^2\right)
    $$

    Using the **reparameterization trick**, the latent variable \( z \) is sampled as:

    $$
    z = \mu + \sigma \cdot \epsilon, \quad \text{where } \epsilon \sim \mathcal{N}(0, I)
    $$

    #### **Decoding Step:**
    The decoder takes the latent variable \( z \) and reconstructs the image:

    $$
    \hat{x} = \text{Decoder}(z)
    $$

    where \(\hat{x}\) is the reconstructed image.

    These equations describe how the VAE transforms input images into a latent space representation and reconstructs them using the decoder.

=== "VAEEncoder"
    ```py3
    class VAEEncoder(torch.nn.Module):
        def __init__(self, ch=128, ch_mult=(1,2,4,4), num_res_blocks=2, in_channels=3, z_channels=16, dtype=torch.float32, device=None):
            super().__init__()
            self.num_resolutions = len(ch_mult)
            self.num_res_blocks = num_res_blocks
            # downsampling
            self.conv_in = torch.nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
            in_ch_mult = (1,) + tuple(ch_mult)
            self.in_ch_mult = in_ch_mult
            self.down = torch.nn.ModuleList()
            for i_level in range(self.num_resolutions):
                block = torch.nn.ModuleList()
                attn = torch.nn.ModuleList()
                block_in = ch*in_ch_mult[i_level]
                block_out = ch*ch_mult[i_level]
                for i_block in range(num_res_blocks):
                    block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dtype=dtype, device=device))
                    block_in = block_out
                down = torch.nn.Module()
                down.block = block
                down.attn = attn
                if i_level != self.num_resolutions - 1:
                    down.downsample = Downsample(block_in, dtype=dtype, device=device)
                self.down.append(down)
            # middle
            self.mid = torch.nn.Module()
            self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dtype=dtype, device=device)
            self.mid.attn_1 = AttnBlock(block_in, dtype=dtype, device=device)
            self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dtype=dtype, device=device)
            # end
            self.norm_out = Normalize(block_in, dtype=dtype, device=device)
            self.conv_out = torch.nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
            self.swish = torch.nn.SiLU(inplace=True)
        def forward(self, x):
            # downsampling
            hs = [self.conv_in(x)]
            for i_level in range(self.num_resolutions):
                for i_block in range(self.num_res_blocks):
                    h = self.down[i_level].block[i_block](hs[-1])
                    hs.append(h)
                if i_level != self.num_resolutions-1:
                    hs.append(self.down[i_level].downsample(hs[-1]))
            # middle
            h = hs[-1]
            h = self.mid.block_1(h)
            h = self.mid.attn_1(h)
            h = self.mid.block_2(h)
            # end
            h = self.norm_out(h)
            h = self.swish(h)
            h = self.conv_out(h)
            return h
    ```

    The code is same as that in the LDM, see more details in ![Latent Diffusion Model](../../chapter7_diffusion/ldm_handson.md)

    Here is an updated table summarizing the **VAEEncoder** feature map shapes, including the **network components** (e.g., ResNet blocks, self-attention layers, convolutions, etc.) at each stage.

    | **Stage**               | **Resolution (H × W)**                          | **Channels**       | **Downsampling Applied?** | **Network Components** |
    |-------------------------|-------------------------------------------------|--------------------|----------------------|-------------------------|
    | **Input Image**         | \( H \times W \)                                | 3                  | No                   | Raw image input |
    | **Conv In**             | \( H \times W \)                                | 128                | No                   | \(3 \times 3\) Conv layer |
    | **Downsampling Level 1** | \( H \times W \) → \( \frac{H}{2} \times \frac{W}{2} \) | 128 → 256 | ✅ Yes | 2 × ResNet Blocks + \(3 \times 3\) Downsampling |
    | **Downsampling Level 2** | \( \frac{H}{2} \times \frac{W}{2} \) → \( \frac{H}{4} \times \frac{W}{4} \) | 256 → 512 | ✅ Yes | 2 × ResNet Blocks + \(3 \times 3\) Downsampling |
    | **Downsampling Level 3** | \( \frac{H}{4} \times \frac{W}{4} \) → \( \frac{H}{8} \times \frac{W}{8} \) | 512 → 512 | ✅ Yes | 2 × ResNet Blocks + \(3 \times 3\) Downsampling |
    | **Middle (Bottleneck)** | \( \frac{H}{8} \times \frac{W}{8} \)            | 512                | ❌ No  | 1 × ResNet Block + 1 × Self-Attention Block + 1 × ResNet Block |
    | **Final Output**        | \( \frac{H}{8} \times \frac{W}{8} \)            | 32 (2 × z_channels) | ❌ No  | Normalization + Swish + \(3 \times 3\) Conv layer (outputs mean & log-variance) |

    ### **Key Observations**
    - **ResNet Blocks** are used at every resolution level to refine features.
    - **Self-Attention is applied only in the middle (bottleneck)** to capture global dependencies.
    - **Downsampling occurs at levels 1, 2, and 3**, reducing spatial dimensions by half each time.
    - The **final layer outputs mean & log-variance** of the latent distribution for sampling.

=== "VAEDecoder"

    ```py3 title="VAEDecoder"
    class VAEDecoder(torch.nn.Module):
        def __init__(self, ch=128, out_ch=3, ch_mult=(1, 2, 4, 4), num_res_blocks=2, resolution=256, z_channels=16, dtype=torch.float32, device=None):
            super().__init__()
            self.num_resolutions = len(ch_mult)
            self.num_res_blocks = num_res_blocks
            block_in = ch * ch_mult[self.num_resolutions - 1]
            curr_res = resolution // 2 ** (self.num_resolutions - 1)
            # z to block_in
            self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
            # middle
            self.mid = torch.nn.Module()
            self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dtype=dtype, device=device)
            self.mid.attn_1 = AttnBlock(block_in, dtype=dtype, device=device)
            self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dtype=dtype, device=device)
            # upsampling
            self.up = torch.nn.ModuleList()
            for i_level in reversed(range(self.num_resolutions)):
                block = torch.nn.ModuleList()
                block_out = ch * ch_mult[i_level]
                for i_block in range(self.num_res_blocks + 1):
                    block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dtype=dtype, device=device))
                    block_in = block_out
                up = torch.nn.Module()
                up.block = block
                if i_level != 0:
                    up.upsample = Upsample(block_in, dtype=dtype, device=device)
                    curr_res = curr_res * 2
                self.up.insert(0, up) # prepend to get consistent order
            # end
            self.norm_out = Normalize(block_in, dtype=dtype, device=device)
            self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
            self.swish = torch.nn.SiLU(inplace=True)
        def forward(self, z):
            # z to block_in
            hidden = self.conv_in(z)
            # middle
            hidden = self.mid.block_1(hidden)
            hidden = self.mid.attn_1(hidden)
            hidden = self.mid.block_2(hidden)
            # upsampling
            for i_level in reversed(range(self.num_resolutions)):
                for i_block in range(self.num_res_blocks + 1):
                    hidden = self.up[i_level].block[i_block](hidden)
                if i_level != 0:
                    hidden = self.up[i_level].upsample(hidden)
            # end
            hidden = self.norm_out(hidden)
            hidden = self.swish(hidden)
            hidden = self.conv_out(hidden)
            return hidden
    ```

    | **Stage**               | **Resolution (H × W)**                          | **Channels**       | **Upsampling Applied?** | **Network Components** |
    |-------------------------|-------------------------------------------------|--------------------|----------------------|-------------------------|
    | **Latent Input**        | \( \frac{H}{8} \times \frac{W}{8} \)            | 16                 | No                   | Raw latent space representation \( z \) |
    | **Conv In**             | \( \frac{H}{8} \times \frac{W}{8} \)            | 512                | No                   | \(3 \times 3\) Conv layer |
    | **Middle (Bottleneck)** | \( \frac{H}{8} \times \frac{W}{8} \)            | 512                | No                   | 1 × ResNet Block + 1 × Self-Attention Block + 1 × ResNet Block |
    | **Upsampling Level 1**  | \( \frac{H}{8} \times \frac{W}{8} \) → \( \frac{H}{4} \times \frac{W}{4} \) | 512 → 512 | ✅ Yes | 3 × ResNet Blocks + Upsample (\(2 \times\)) |
    | **Upsampling Level 2**  | \( \frac{H}{4} \times \frac{W}{4} \) → \( \frac{H}{2} \times \frac{W}{2} \) | 512 → 256 | ✅ Yes | 3 × ResNet Blocks + Upsample (\(2 \times\)) |
    | **Upsampling Level 3**  | \( \frac{H}{2} \times \frac{W}{2} \) → \( H \times W \) | 256 → 128 | ✅ Yes | 3 × ResNet Blocks + Upsample (\(2 \times\)) |
    | **Upsampling Level 4**  | \( H \times W \)                                | 128 → 128 | ❌ No | 3 × ResNet Blocks (Final stage, no upsampling) |
    | **Final Processing**    | \( H \times W \)                                | 3                  | ❌ No                | Normalize → Swish Activation → \(3 \times 3\) Conv layer |

    The VAEDecoder reconstructs an image from a latent space representation by progressively upsampling and refining features through ResNet blocks, self-attention, and convolutional layers.

=== "AttnBlock"

    ```py3 title="AttnBlock"
    class AttnBlock(torch.nn.Module):
    def __init__(self, in_channels, dtype=torch.float32, device=None):
        super().__init__()
        self.norm = Normalize(in_channels, dtype=dtype, device=device)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)
    def forward(self, x):
        hidden = self.norm(x)
        q = self.q(hidden)
        k = self.k(hidden)
        v = self.v(hidden)
        b, c, h, w = q.shape
        q, k, v = map(lambda x: einops.rearrange(x, "b c h w -> b 1 (h w) c").contiguous(), (q, k, v))
        hidden = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # scale is dim ** -0.5 per default
        hidden = einops.rearrange(hidden, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)
        hidden = self.proj_out(hidden)
        return x + hidden
    ```
    which is a standard self attention block.

=== "resnet block"
    ```py3 title="ResnetBlock"
    def Normalize(in_channels, num_groups=32, dtype=torch.float32, device=None):
        return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True, dtype=dtype, device=device)
    class ResnetBlock(torch.nn.Module):
        def **init**(self, *, in_channels, out_channels=None, dtype=torch.float32, device=None):
            super().**init**()
            self.in_channels = in_channels
            out_channels = in_channels if out_channels is None else out_channels
            self.out_channels = out_channels
            self.norm1 = Normalize(in_channels, dtype=dtype, device=device)
            self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
            self.norm2 = Normalize(out_channels, dtype=dtype, device=device)
            self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
            if self.in_channels != self.out_channels:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)
            else:
                self.nin_shortcut = None
            self.swish = torch.nn.SiLU(inplace=True)
        def forward(self, x):
            hidden = x
            hidden = self.norm1(hidden)
            hidden = self.swish(hidden)
            hidden = self.conv1(hidden)
            hidden = self.norm2(hidden)
            hidden = self.swish(hidden)
            hidden = self.conv2(hidden)
            if self.in_channels != self.out_channels:
                x = self.nin_shortcut(x)
            return x + hidden
    ```

    $$
    F(X) = \text{Conv2} \left( \text{Swish} \left( \text{Norm2} \left( \text{Conv1} \left( \text{Swish} \left( \text{Norm1}(X) \right) \right) \right) \right) \right)
    $$

    The **final output**  is:

    $$   Y = X + F(X) $$

### Patchify

=== "PatchEmbed"
    ```py3 title="Patchify"
    class PatchEmbed(nn.Module):
        """ 2D Image to Patch Embedding"""
        def __init__(
                self,
                img_size: Optional[int] = 224,
                patch_size: int = 16,
                in_chans: int = 3,
                embed_dim: int = 768,
                flatten: bool = True,
                bias: bool = True,
                strict_img_size: bool = True,
                dynamic_img_pad: bool = False,
                dtype=None,
                device=None,
        ):
            super().__init__()
            self.patch_size = (patch_size, patch_size)
            if img_size is not None:
                self.img_size = (img_size, img_size)
                self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
                self.num_patches = self.grid_size[0] * self.grid_size[1]
            else:
                self.img_size = None
                self.grid_size = None
                self.num_patches = None
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.strict_img_size = strict_img_size
            self.dynamic_img_pad = dynamic_img_pad
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias, dtype=dtype, device=device)
        def forward(self, x):
            B, C, H, W = x.shape
            x = self.proj(x)
            if self.flatten:
                x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
            return x
    ```

Use the convolution to do the patchify that convert the original image from [B,C,H,W] to $[B,N,C]$, where $N=\frac{H}{\text{patch size}}\times\frac{W}{\text{patch size}}=\frac{H\times W}{4}$. The convolutional kernel size is $patch_size\times patch_size$.

### TimeEmbedding

=== "TimestepEmbedder"
    ```py3
    class TimestepEmbedder(nn.Module):
        """Embeds scalar timesteps into vector representations."""

        def __init__(self, hidden_size, frequency_embedding_size=256, dtype=None, device=None):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(frequency_embedding_size, hidden_size, bias=True, dtype=dtype, device=device),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device),
            )
            self.frequency_embedding_size = frequency_embedding_size

        @staticmethod
        def timestep_embedding(t, dim, max_period=10000):
            """
            Create sinusoidal timestep embeddings.
            :param t: a 1-D Tensor of N indices, one per batch element.
                            These may be fractional.
            :param dim: the dimension of the output.
            :param max_period: controls the minimum frequency of the embeddings.
            :return: an (N, D) Tensor of positional embeddings.
            """
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period)
                * torch.arange(start=0, end=half, dtype=torch.float32)
                / half
            ).to(device=t.device)
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            if torch.is_floating_point(t):
                embedding = embedding.to(dtype=t.dtype)
            return embedding

        def forward(self, t, dtype, **kwargs):
            t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(dtype)
            t_emb = self.mlp(t_freq)
            return t_emb
    ```

The **timestep embedding** function generates a vector representation of time \( t \) using **sinusoidal embeddings**. The formula can be rewritten as a single vector equation.

**Embedding as a Vector**
For a given timestep \( t \), the embedding vector **\( E_t \)** is computed as:

$$
E_t = \left[ \cos\left(t \cdot f_1\right), \sin\left(t \cdot f_1\right), \cos\left(t \cdot f_2\right), \sin\left(t \cdot f_2\right), \dots, \cos\left(t \cdot f_{\frac{D}{2}}\right), \sin\left(t \cdot f_{\frac{D}{2}}\right) \right]
$$

where:

- \( f_i = e^{-\frac{\log(\text{max\_period})}{D/2} \cdot i} \) are the frequency components.
- \( D \) is the embedding dimension.
- The final embedding vector **\( E_t \)** has shape **\( (D,) \)**.

If \( D \) is **odd**, an extra zero is appended:

$$
E_t = \left[ E_t, 0 \right]
$$

This vector representation ensures **smooth temporal encoding** and is widely used in **diffusion models, transformers, and time-aware architectures**. 🚀

### MM-DiT
![alt text](../../../images/image-81.png)

The overall structure of the MM-DiT network

![](https://pica.zhimg.com/v2-5852fb562c033510fef96f7772e41278_r.jpg)

In each MM-DiT Block, the text context and image context are fed into the attention by concatenation

#### DiT Block
=== "DismantledBlock"

    ```python3
    # code
    class DismantledBlock(nn.Module):
        """A DiT block with gated adaptive layer norm (adaLN) conditioning."""
        ATTENTION_MODES = ("xformers", "torch", "torch-hb", "math", "debug")
        def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            attn_mode: str = "xformers",
            qkv_bias: bool = False,
            pre_only: bool = False,
            rmsnorm: bool = False,
            scale_mod_only: bool = False,
            swiglu: bool = False,
            qk_norm: Optional[str] = None,
            dtype=None,
            device=None,
            **block_kwargs,
        ):
            super().__init__()
            assert attn_mode in self.ATTENTION_MODES
            if not rmsnorm:
                self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
            else:
                self.norm1 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, attn_mode=attn_mode, pre_only=pre_only, qk_norm=qk_norm, rmsnorm=rmsnorm, dtype=dtype, device=device)
            if not pre_only:
                if not rmsnorm:
                    self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
                else:
                    self.norm2 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            mlp_hidden_dim = int(hidden_size * mlp_ratio)
            if not pre_only:
                if not swiglu:
                    self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU(approximate="tanh"), dtype=dtype, device=device)
                else:
                    self.mlp = SwiGLUFeedForward(dim=hidden_size, hidden_dim=mlp_hidden_dim, multiple_of=256)
            self.scale_mod_only = scale_mod_only
            if not scale_mod_only:
                n_mods = 6 if not pre_only else 2
            else:
                n_mods = 4 if not pre_only else 1
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, n_mods * hidden_size, bias=True, dtype=dtype, device=device))
            self.pre_only = pre_only
        def pre_attention(self, x: torch.Tensor, c: torch.Tensor):
            assert x is not None, "pre_attention called with None input"
            if not self.pre_only:
                if not self.scale_mod_only:
                    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
                else:
                    shift_msa = None
                    shift_mlp = None
                    scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
                qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
                return qkv, (x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
            else:
                if not self.scale_mod_only:
                    shift_msa, scale_msa = self.adaLN_modulation(c).chunk(2, dim=1)
                else:
                    shift_msa = None
                    scale_msa = self.adaLN_modulation(c)
                qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
                return qkv, None
        def post_attention(self, attn, x, gate_msa, shift_mlp, scale_mlp, gate_mlp):
            assert not self.pre_only
            x = x + gate_msa.unsqueeze(1) * self.attn.post_attention(attn)
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
            return x
        def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
            assert not self.pre_only
            (q, k, v), intermediates = self.pre_attention(x, c)
            attn = attention(q, k, v, self.attn.num_heads)
            return self.post_attention(attn, *intermediates)
    ```

It got three main blocks

- pre_attention
- attention
- post_attention

=== "SelfAttention"
    ```py3
    class SelfAttention(nn.Module):
        ATTENTION_MODES = ("xformers", "torch", "torch-hb", "math", "debug")
        def **init**(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_scale: Optional[float] = None,
            attn_mode: str = "xformers",
            pre_only: bool = False,
            qk_norm: Optional[str] = None,
            rmsnorm: bool = False,
            dtype=None,
            device=None,
        ):
            super().**init**()
            self.num_heads = num_heads
            self.head_dim = dim // num_heads

            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
            if not pre_only:
                self.proj = nn.Linear(dim, dim, dtype=dtype, device=device)
            assert attn_mode in self.ATTENTION_MODES
            self.attn_mode = attn_mode
            self.pre_only = pre_only
            if qk_norm == "rms":
                self.ln_q = RMSNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6, dtype=dtype, device=device)
                self.ln_k = RMSNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6, dtype=dtype, device=device)
            elif qk_norm == "ln":
                self.ln_q = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6, dtype=dtype, device=device)
                self.ln_k = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6, dtype=dtype, device=device)
            elif qk_norm is None:
                self.ln_q = nn.Identity()
                self.ln_k = nn.Identity()
            else:
                raise ValueError(qk_norm)
        def pre_attention(self, x: torch.Tensor):
            B, L, C = x.shape
            qkv = self.qkv(x)
            q, k, v = split_qkv(qkv, self.head_dim)
            q = self.ln_q(q).reshape(q.shape[0], q.shape[1], -1)
            k = self.ln_k(k).reshape(q.shape[0], q.shape[1], -1)
            return (q, k, v)
        def post_attention(self, x: torch.Tensor) -> torch.Tensor:
            assert not self.pre_only
            x = self.proj(x)
            return x
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            (q, k, v) = self.pre_attention(x)
            x = attention(q, k, v, self.num_heads)
            x = self.post_attention(x)
            return x
    ```

It is a normal self attention network but with different layernorm layers. Also, it cobines the linear projections of Q,K, and V together into a single linear projection.

##### RMSNorm

Given an input feature vector $x$ with **dimension**  $d$, RMSNorm is defined as:

$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma
$$

where:

- **RMS (Root Mean Square) is calculated as:**

$$
\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}
$$

- $\epsilon$ is a small constant to prevent division by zero.

- $\gamma$ is a **learnable scaling parameter**  (similar to LayerNorm).
If a **bias term**  $\beta$ is added, the formula becomes:

$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma + \beta
$$

## References

- stable diffusion 3 reading: <https://zhuanlan.zhihu.com/p/684068402?utm_source=chatgpt.com>
- [sd 3 inference code](https://github.com/Stability-AI/sd3-ref)
- [Instruct Training script based on SD3](https://github.com/huggingface/diffusers/blob/main/examples/instruct_pix2pix/train_instruct_pix2pix_sdxl.py)
- [Flexible PyTorch implementation of StableDiffusion-3 based on  diffusers](https://github.com/haoningwu3639/SimpleSDM-3)
- [Stable Diffusion 3 Fintune Guide](https://stabilityai.notion.site/Stable-Diffusion-3-Medium-Fine-tuning-Tutorial-17f90df74bce4c62a295849f0dc8fb7e)
