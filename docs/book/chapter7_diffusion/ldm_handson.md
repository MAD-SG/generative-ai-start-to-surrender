# Latent Diffusion Handson
## Build A Web Demo

Web Demo

![alt text](../../images/image-46.png)

Refer the guide in `experiment/latent_diffusion/readme.md`

## Generating Results

## Code
The overall framework concists of two main blocks, `ddpm` and `vae`

The `vae` devides intro two types, either the `VQ-GAN` or `KL-VAE`. Refer VQ-GAN for more details. Here we only look as the structure details about the `KL-VAE`. And in the future version, stable diffusion only used the `KL-VAE`.

### Autoencoder
#### AutoencoderKL

```py3 title='AutoEncoderKL'

class AutoencoderKL(pl.LightningModule):
    """
    该类实现了一个基于 KL 散度约束的自编码器，其编码器生成潜变量分布（高斯分布），解码器重构输入图像。
    通过两个卷积层实现了从编码器特征到潜变量分布参数的转换以及从潜变量到解码器输入的转换。
    支持对抗性训练：通过两个独立的优化器分别训练自动编码器和判别器。
    同时还内置了数据预处理、图像记录和颜色映射等辅助功能，方便后续训练和可视化。
    """
    def __init__(self,
                 ddconfig, # 传递给编码器和解码器的配置字典，包含网络结构参数。
                 lossconfig, # 损失函数的配置，通过 instantiate_from_config 实例化具体的损失函数。
                 embed_dim, # 潜变量空间的维度。
                 ckpt_path=None, # 若不为空，则从指定的检查点加载预训练模型参数
                 ignore_keys=[], # 加载预训练参数时需要忽略的键列表。
                 image_key="image", # 从输入 batch 中提取图像数据时所使用的键，默认是 "image"。
                 colorize_nlabels=None, # 如果提供，则说明模型支持将多通道（例如分割图）映射到 RGB，数值表示标签数量。
                 monitor=None, # 用于监控训练过程中的指标（例如用于早停等）。
                 ):
        ```
        根据 ddconfig 实例化编码器和解码器。
        根据 lossconfig 实例化损失函数。
        断言 ddconfig["double_z"] 为 True，确保潜变量的通道数是成对的（常见于均值和对数方差的组合）。
        定义了两个卷积层：
        quant_conv：将编码器输出的特征（通道数为 2*z_channels）映射到 2*embed_dim，用于生成高斯分布的参数（均值和对数方差）。
        post_quant_conv：在解码之前，将潜变量从 embed_dim 映射回到解码器所需的 z_channels。
        如果 colorize_nlabels 不为空，则注册一个名为 colorize 的 buffer，用于之后将多通道的分割图转换为 RGB 图像。
        如果提供了检查点路径，则调用 init_from_ckpt 方法加载预训练模型参数。
        ```
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        """ 参数加载方法 init_from_ckpt
        功能：从指定路径加载预训练的模型状态字典（state_dict）。
        细节：
        加载后会遍历所有键，如果键名以 ignore_keys 中的某个前缀开头，则会将该键从状态字典中删除（适用于忽略部分预训练参数）。
        最后使用 load_state_dict 加载参数，strict=False 意味着可以允许部分参数缺失或多余。
        """
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        """
        encode 方法：
        输入图像 x 经过编码器得到中间特征 h。
        使用 quant_conv 将特征映射为“时刻”（moments），通常包括均值和对数方差。
        利用这些参数构造一个 DiagonalGaussianDistribution（对角高斯分布），作为后验分布返回。
        """
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        """
        将潜变量 z 通过 post_quant_conv 转换到适合解码器的维度，然后经过解码器生成重构图像。
        """
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        """
        调用 encode 得到后验分布 posterior。
        根据 sample_posterior 决定是采样（随机采样）还是取模式（均值）作为潜变量 z。
        将 z 输入到 decode 得到最终输出。
        返回解码后的图像和后验分布。
        """
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        ```
        功能：从 batch 中提取图像数据。
        细节：
        根据 image_key 取出对应的数据。
        如果输入数据的维度为 3（例如没有明确通道维度），则在最后加一个维度。
        将数据从 (batch, height, width, channels) 转换为 (batch, channels, height, width)（符合 PyTorch 的标准格式）。
        转换为浮点型，并确保内存格式连续。
        ```
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        """
        training_step 方法：
        根据 optimizer_idx 的值，分别计算自动编码器（encoder+decoder）和判别器的损失：
        当 optimizer_idx == 0：计算自动编码器的损失（重构损失、KL 散度等），并记录相应的日志。
        当 optimizer_idx == 1：计算判别器的损失（通常用于对抗训练），并记录日志。
        使用 self.loss 对象来计算不同部分的损失，传入当前的全局步数和最后一层参数（用于例如梯度惩罚或权重调整）。
        validation_step 方法：

        与训练步骤类似，但计算的是验证集上的损失，并记录验证日志（例如 "val/rec_loss"）
        """
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        """
        功能：
            定义了两个独立的优化器：
            一个用于训练自动编码器（包含编码器、解码器以及两个卷积层）的 Adam 优化器。
            一个用于训练判别器（通过 self.loss.discriminator 访问）的 Adam 优化器。
            两个优化器都使用相同的学习率（self.learning_rate）以及相同的 Adam 参数（betas=(0.5, 0.9)）。
        """
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        """返回解码器中最后一层卷积的权重。这个权重有时用于辅助损失计算或者梯度分析。"""
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        """
        用于在训练或验证过程中记录输入图像、重构图像以及随机生成的样本图像。
        通道数大于 3（例如分割图），则调用 to_rgb 方法将其转换为 RGB 图像。
        返回一个包含 "inputs"、"reconstructions"、"samples" 等键的字典，便于日志记录和可视化。
        """
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        """
        专门用于将分割图（或通道数大于3的图像）通过一个卷积操作映射到 RGB 图像。
        使用预先注册的 colorize buffer（若不存在则新注册）作为卷积核，完成通道映射后，对结果进行归一化，确保像素值在 [-1, 1] 范围内。
        """
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
```

Compared with a "normal" VAE, we need to check if the loss and encoding and decoding process is same to the normal VAE.

At first look, the loss combines two parts

1. Standard VAE Loss, that is, the reconstruction loss + KL divergence (regularization) loss
2. GAN loss

Theoretical, the loss should be

$$\tag{1}
 \mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \| x_i - \hat{x}_i \|^2+ \sum_{l} \lambda_l \| \phi_l(x) - \phi_l(G(z)) \|_2^2  + \frac{1}{2} \sum_{j=1}^{d} (1 + \log \sigma_{i,j}^2 - \mu_{i,j}^2 - \sigma_{i,j}^2)+ L_{adv}
$$

where

$$
L_{adv} = -\mathbb{E}_{z \sim p(z)}\left[ D\bigl(G(z)\bigr) \right]
$$

and $\phi_l(x)$ is the feature map from the pretrained CNN.

In equation (1), the first two can be considered as the perceptual reconstruction loss which is used in `VQ-GAN` to train the encoder and decoder.

Let's check if it is same with the above assumption.

#### Gaussian Sampling

```py3 title="Gaussian Sampling"
class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)
    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])
    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)
    def mode(self):
        return self.mean
```

#### Losses

Details of the loss

```python3 title="LPIPSWithDiscriminator"
import torch
import torch.nn as nn

from taming.modules.losses.vqperceptual import *
class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        # LPIPS（Learned Perceptual Image Patch Similarity）能够捕捉图像之# 间更高层次的相似性。当 perceptual_weight 大于 0 时，会将该损失与 # L1 像素损失相加，从而获得更符合人眼感知的重建效果。
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        # self.logvar 是一个可学习的标量参数，用于对重建损失进行缩放。具体来说，我们将重建误差除以 $\exp(\text{logvar})$，
        # 再加上 logvar，这样可以动态平衡重建损失的尺度，同时在训练中让模型学习一个合适的权重。
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        # 判别器使用 NLayerDiscriminator 构建，支持多层结构，同时可以通过 use_actnorm 选择是否使用激活归一化。

        self.discriminator_iter_start = disc_start
        # discriminator_iter_start 用来设置在训练的哪个步骤开始引入判别器的损失，这样可以先让生成器学到较为稳定的重建，再加入对抗训练。

        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        # disc_conditional 参数用于判断是否为条件判别器，即在输入图像的基础上是否还需要额外的条件信息（如类别、语义信息等）。
        # disc_loss 根据传入的参数选择 hinge 损失或者 vanilla 损失。
        self.disc_factor = disc_factor
        # disc_factor 和 discriminator_weight 用于对判别器相关损失进行加权，使得 GAN 部分的损失不会直接主导整个损失函数。

        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        """
        在生成器部分，我们希望同时优化重建（NLL）和生成器对抗损失（g_loss），而二者的尺度可能相差较大。为此，代码中引入了自适应权重计算
        """

        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log
```

The default config for loss is

```yaml
params:
    disc_start: 50001
    kl_weight: 0.000001
    disc_weight: 0.5
```

Here `adopt_weight` is defined as

```py3
def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight
```

which control in which step, we introduce the GAN loss.

##### Adaptive weights for reconstruction loss and gan loss

We can express `calculate_adaptive_weight`  using the following formula

1. assume the gradient of last layer from the NLL loss：

$$
\nabla_{\text{nll}} = \nabla_{\theta_{\text{last}}} \mathcal{L}_{\text{nll}}
$$

and the gradient of the last layer from the GAN loss：

$$
\nabla_{g} = \nabla_{\theta_{\text{last}}} \mathcal{L}_{g}
$$

2. Adptive weights
\[
w_{\text{adaptive}} = \text{clip}\!\left(\frac{\|\nabla_{\text{nll}}\|}{\|\nabla_{g}\| + \epsilon}, \, 0, \, 10^4\right)
\]

3. Finally adding the weights \(w_{\text{disc}}\)：

$$
d_{\text{weight}} = w_{\text{adaptive}} \times w_{\text{disc}}
$$

It assumes that the norm of the gradient of the GAN should be equal to that of the reconstruction loss, make them in same level. Of course, finnaly, it will multiply another weights to control the importance of the GAN loss compared with other loss

##### Forward

The forward part is splited by the optimizer index, when `optimizer_idx==0`, it optimized the VAE, when `optimizer_idx==1`, it optimized the GAN

###### Optimization of VAE

It also splited into three parts

1. reconstruction loss
    It used the $L_1$ loss for reconstruction loss. If the `perceptual_weight>0`, it also combines with the `perceptual_loss`, which is usually a loss to constrained the feature map distances between the generation and input via a pretrained CNN feature extraction like "VGG"

```py3
rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
# 如果设置了感知损失权重，则计算 LPIPS 感知损失并叠加
if self.perceptual_weight > 0:
    p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
    rec_loss = rec_loss + self.perceptual_weight * p_loss
    # NLL 损失：将重建损失经过动态缩放（通过 logvar）后计算得到
nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
weighted_nll_loss = nll_loss
if weights is not None:
    weighted_nll_loss = weights * nll_loss
weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
```

2. KL Divergence
   posteriors.kl() the KL divence between the posteriors and the Gaussian distribution $U(0,1)$. It is defined in the Gaissian Sampler `DiagonalGaussianDistribution`.  It it is a deterministic sampling, this loss returns `0`.

3. Gan Loss
    Since here it aimed to train the generator (VAE), we dont have the real part for adversary loss. If it is a conditional GAN, then put the conditional information into the input, which is a quite standard adversary loss.

    ```py3
    if cond is None:
        assert not self.disc_conditional
        logits_fake = self.discriminator(reconstructions.contiguous())
    else:
        assert self.disc_conditional
        logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
    g_loss = -torch.mean(logits_fake)
    ```

###### Optimization of GAN

```py3
if optimizer_idx == 1:
    # 对判别器而言，需要分别计算真实图像和生成图像的 logits
    if cond is None:
        logits_real = self.discriminator(inputs.contiguous().detach())
        logits_fake = self.discriminator(reconstructions.contiguous().detach())
    else:
        logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
        logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

    # 同样采用 adopt_weight 控制判别器损失的引入时机
    disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
    # 判别器损失根据选用的 hinge 或 vanilla 损失函数进行计算
    d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

    log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
           "{}/logits_real".format(split): logits_real.detach().mean(),
           "{}/logits_fake".format(split): logits_fake.detach().mean()
           }
    return d_loss, log
```

To optimize the GAN, we only need the fake/real images. Also, add the condition into the `discriminator` when it is a conditional GAN.

```py3
def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss
```

use hinge or vanilla loss by the parameter
Below are the mathematical formulas for the two loss functions as implemented in the code:

---

###### Hinge Loss

Given the discriminator outputs for real samples, \( D(x) \), and for generated samples, \( D(\tilde{x}) \), the hinge loss for the discriminator is defined as:

$$
\mathcal{L}_D^{\text{hinge}} = \frac{1}{2} \left( \mathbb{E}_{x \sim p_{\text{data}}}\left[\max\left(0,\, 1 - D(x)\right)\right] + \mathbb{E}_{\tilde{x} \sim p_G}\left[\max\left(0,\, 1 + D(\tilde{x})\right)\right] \right)
$$

- \(\max(0,\, 1 - D(x))\) computes the loss for real samples.
- \(\max(0,\, 1 + D(\tilde{x}))\) computes the loss for generated (fake) samples.

###### Vanilla Loss

For the vanilla loss, we use the softplus function, defined as:

$$
\text{softplus}(x) = \ln(1 + e^x)
$$

Then the vanilla loss for the discriminator is given by:

$$
\mathcal{L}_D^{\text{vanilla}} = \frac{1}{2} \left( \mathbb{E}_{x \sim p_{\text{data}}}\left[\ln\left(1 + e^{-D(x)}\right)\right] + \mathbb{E}_{\tilde{x} \sim p_G}\left[\ln\left(1 + e^{D(\tilde{x})}\right)\right] \right)
$$

These formulas correspond exactly to the implementation in the code:

- For the hinge loss, the code computes \(\text{loss\_real} = \text{mean}(\text{ReLU}(1 - \text{logits\_real}))\) and \(\text{loss\_fake} = \text{mean}(\text{ReLU}(1 + \text{logits\_fake}))\), then takes their average.
- For the vanilla loss, the code computes \(\text{mean}(\text{softplus}(-\text{logits\_real}))\) and \(\text{mean}(\text{softplus}(\text{logits\_fake}))\) and averages them.
Here we plot the curve of the above two types of losses for intuitive comparison.

![alt text](../../images/image-47.png)

#### Summary
Based on the above analysis, we can see that the loss is basically a normal `VAE loss` +  `GAN loss` + `Perceptural Loss`.

### DDPM
