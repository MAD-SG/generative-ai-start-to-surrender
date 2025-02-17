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

$$
w_{\text{adaptive}} = \text{clip}\!\left(\frac{\|\nabla_{\text{nll}}\|}{\|\nabla_{g}\| + \epsilon}, \, 0, \, 10^4\right)
$$

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

Train latent diffusion

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/latent-diffusion/<config_spec>.yaml -t --gpus 0,
```

Let's study the codes step by step. The code is in `ldm/models/diffusion/ddpm.py` in [github: CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion)

#### DDPM Class

##### `__init__`

```py3 title="init"

class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

```

The DDPM class is herited from the `pl.LightningModule`, includes the forward, reversem training and sampling processes.

In the init part,

- `timesteps=1000` is the diffusion steps, usually be 1000,
- `beta_schedule='linear'` is the noise scheduler that controls $\beta_t$
- `parameterization="eps"` is the training target, 'eps' means network predicted the noise, `x0` means predict the denoied image
- `use_ema=True` means the expotentinoal moving average to stabalize the training. `LitEma(self.model)` defines the ema model
- `logvar_init=0`: the initial value of log var in the traiing. Can be choose to learn or use fixed value.
- `self.model = DiffusionWrapper(unet_config, conditioning_key)` defines the U-Net used in the diffusion proces

##### `register_schedule`

```py3 title="register_schedule"
    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()
```

`register_schedule` is mainly used tp **calculate the noise scheduler**, including the  $\beta_t$ (noise), $\alpha_t$ (original image info)、 accumulated $\alpha_t$, ($\bar{\alpha}_t$), and mean and variance of posterior distribution $q(x_{t-1} | x_t, x_0)$.

**$\beta_t$**

- **Linear**:

  $$
  \beta_t = \text{linear\_start} + t \cdot \frac{\text{linear\_end} - \text{linear\_start}}{\text{timesteps} - 1}
  $$

  Here  `linear_start = 1e-4`，`linear_end = 2e-2`，noise is increasing w.r.t $t$.

- **Cosine**:

  $$
  \beta_t = 1 - \frac{\cos\left(\frac{t / T + s}{1 + s} \cdot \frac{\pi}{2} \right)^2}{\cos\left(\frac{s}{1+s} \cdot \frac{\pi}{2}\right)^2}
  $$

    `s = 8e-3` controls the initial noise

**$\alpha_t$ and  $\bar{\alpha}_t$**

- **$\alpha_t$**:

  $$
  \alpha_t = 1 - \beta_t
  $$

- **$\bar{\alpha}_t$ (product of previous $t$ steps)**:

  $$
  \bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i
  $$

    $\bar{\alpha}_t$ represents the aspect of original info is left from $x_0$ to $x_t$

**$q(x_{t-1} | x_t, x_0)$**

- **mean $\mu_t$**:

  $$
  \mu_t = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t
  $$

- **varience $\sigma_t^2$**:

  $$
  \sigma_t^2 = (1 - v) \cdot \frac{\beta_t (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} + v \cdot \beta_t
  $$

  `v`（`v_posterior`) determines the calculatetion of the variance, default `v=0`。

According to the 'make_beta_schedule' formula

```python
betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
alphas = 1. - betas
alphas_cumprod = np.cumprod(alphas, axis=0)
alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
```

- Here `betas`  is  $\beta_t$，`alphas` is $\alpha_t$，`alphas_cumprod` is $\bar{\alpha}_t$.

Here is uses the `make_beta_schedule` functoin, which is

The function `make_beta_schedule` is designed to generate the noise schedule $\beta_t$ used in the diffusion process. Different scheduling methods impact the convergence and final image quality of the diffusion model.

Input Parameters

- `schedule`: Specifies the method for computing **$\beta_t$**.
- `n_timestep`: Total number of diffusion steps `T`, typically **1000**.
- `linear_start` & `linear_end`:
  - **Linear schedule**: $\beta_t$ gradually increases from `linear_start` to `linear_end`.
  - **Square root schedule**: $\beta_t$ increases based on a **square root transformation**.
- `cosine_s=8e-3`:
  - Used in **cosine schedule**, typically for **Stable Diffusion**, improving stability.

- **1. Linear Schedule**

    ```python
    if schedule == "linear":
        betas = (
            torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )
    ```

    **Mathematical Formula**:

     $$
     \beta_t = \left( \sqrt{\text{linear\_start}} + \frac{t}{T} \left( \sqrt{\text{linear\_end}} - \sqrt{\text{linear\_start}} \right) \right)^2
     $$

    First **take the square root**, then apply **linear interpolation**, and finally **square the values**.
    This ensures a **smooth increase** in $\beta_t$.

- **2. Cosine Schedule**

    ```python
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)
    ```

    **Mathematical Formula**:

    $$
      \alpha_t = \cos^2 \left( \frac{t/T + s}{1 + s} \cdot \frac{\pi}{2} \right)
    $$
  - `s` controls the **initial noise strength**.
  - $\alpha_t$ is normalized so that $\alpha_0 = 1$.
  - $\beta_t = 1 - \frac{\alpha_{t+1}}{\alpha_t}$ ensures $\beta_t$ is computed correctly.

    - **Key Features**:
      - Cosine scheduling **increases slowly at first** and then **rapidly** towards the end.
      - This better **matches human perception of noise** and improves **detail generation**.

- **3. Square Root Linear Schedule**

    ```python
    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    ```

    **Mathematical Formula**:
    $$
     \beta_t = \text{linear\_start} + \frac{t}{T} \left( \text{linear\_end} - \text{linear\_start} \right)
    $$
    $\beta_t$ **increases uniformly** over time.

- **4. Square Root Schedule**

    ```python
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    ```

    **Mathematical Formula**:

    $$
    \beta_t = \left( \text{linear\_start} + \frac{t}{T} \left( \text{linear\_end} - \text{linear\_start} \right) \right)^{0.5}
    $$

     **Takes the square root** to achieve a **smoother increase** in $\beta_t$.

**Comparison of Different $\beta_t$ Schedules**

|  Schedule Type  | Formula | Usage |
|-----------|-------------------------------|----------------|
| **Linear** | \( \beta_t = \left( \sqrt{\text{start}} + \frac{t}{T} \left( \sqrt{\text{end}} - \sqrt{\text{start}} \right) \right)^2 \) | **Standard DDPM, suitable for most tasks** |
| **Cosine** | \( \alpha_t = \cos^2 \left( \frac{t/T + s}{1 + s} \cdot \frac{\pi}{2} \right) \) | **Stable Diffusion, improves image quality** |
| **Sqrt Linear** | \( \beta_t = \text{start} + \frac{t}{T} \left( \text{end} - \text{start} \right) \) | **Used for denoising tasks** |
| **Sqrt** | \( \beta_t = \left( \text{start} + \frac{t}{T} \left( \text{end} - \text{start} \right) \right)^{0.5} \) | **Smooth increase, useful for video tasks** |

![addd](../../images/image-48.png)

The choice of schedule impacts **model convergence, image quality, and noise control**. **Experiments** are often necessary to determine the most suitable $\beta_t$ scheduling method.

And calculated the posterior distribution：

```python
posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + self.v_posterior * betas
```

Here `posterior_variance` is $\sigma_t^2$

Mean parameters

```python
self.register_buffer('posterior_mean_coef1', to_torch(
    betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
self.register_buffer('posterior_mean_coef2', to_torch(
    (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
```

Which is the two coes of the $\mu_t$

##### training step

```py3
def q_sample(self, x_start, t, noise=None):
    noise = default(noise, lambda: torch.randn_like(x_start))
    return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

def forward(self, x, *args, **kwargs):
    # input image shape is: [b, c, h, w]
    t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
    # random pick a time step t, as the argument of the model
    return self.p_losses(x, t, *args, **kwargs)

def p_losses(self, x_start, t, noise=None):
    noise = default(noise, lambda: torch.randn_like(x_start))
    x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
    model_out = self.model(x_noisy, t)

    loss_dict = {}
    if self.parameterization == "eps":
        target = noise
    elif self.parameterization == "x0":
        target = x_start
    else:
        raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

    loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

    log_prefix = 'train' if self.training else 'val'

    loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
    loss_simple = loss.mean() * self.l_simple_weight

    loss_vlb = (self.lvlb_weights[t] * loss).mean()
    loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

    loss = loss_simple + self.original_elbo_weight * loss_vlb

    loss_dict.update({f'{log_prefix}/loss': loss})

    return loss, loss_dict
```

Let's recall the forward process of ddpm

$$
x_t = \sqrt{1-\beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon_t,\quad \epsilon_t \sim \mathcal{N}(0,I)
$$

$x_0$ is the noiseless image. The diffusion process (adding noise) is definedin the `q_sample` method, which is exactly

在 DDPM（Denoising Diffusion Probabilistic Models）中，加噪声的过程（forward diffusion process）可以描述为一个逐步加入噪声的马尔可夫链。具体来说，每一步的噪声添加公式为：

$$
x_t = \sqrt{1-\beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon_t,\quad \epsilon_t \sim \mathcal{N}(0,I)
$$

with

$$
\bar{\alpha}_t = \prod_{s=1}^{t} (1-\beta_s)
$$

Theoretically,

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon
$$

And in the code  `q_sample`,

- `self.sqrt_alphas_cumprod` $\leftrightarrow$ \(\sqrt{\bar{\alpha}_t}\) (product of square root of \(\prod_{s=1}^{t} (1-\beta_s)\)），
- `self.sqrt_one_minus_alphas_cumprod` $\leftrightarrow$ \(\sqrt{1-\bar{\alpha}_t}\)，

There are two types of target, one is `x_0`, one is `eps`, corresponds to the noiseless image and the noise. We usually used the noise prediction. The loss is then used $L_1$ or $L_2$.

The loss typically is quite simple, but it gots another re-weighted copy which is

This code snippet is part of the loss computation in a diffusion model when the model is parameterized to predict the noise (often referred to as "eps" parameterization). Here’s what it does:
weights help balance the contribution of each term.

```py3
...
if self.parameterization == "eps":
    lvlb_weights = self.betas ** 2 / (
        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
···
def p_losses(self, x_start, t, noise=None):
    ...
    loss_vlb = (self.lvlb_weights[t] * loss).mean()
    loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

    loss = loss_simple + self.original_elbo_weight * loss_vlb
    ...
```

**Accounting for the Noise Schedule and Posterior Variance:**

- `self.betas` represents the noise schedule parameters at each timestep.
- `self.posterior_variance` is the variance of the reverse (posterior) process \( q(x_{t-1}\mid x_t, x_0) \).
- `alphas` (converted to a torch tensor) and `self.alphas_cumprod` (the cumulative product of \( 1-\beta_t \)) are used to capture how the signal decays over time.

   The formula

   $$
   \text{lvlb\_weights} = \frac{\beta_t^2}{2 \cdot \text{posterior\_variance} \cdot \alpha_t \cdot (1 - \text{alphas\_cumprod}_t)}
   $$
   adjusts the loss term based on these factors, ensuring that the loss is appropriately scaled at each timestep.

**Usage in the Loss Function:**
   When the model predicts \(\epsilon\) (the noise added to \( x_0 \)), the training loss often reduces to a weighted mean squared error (MSE) between the predicted and actual noise. The `lvlb_weights` are used to weight this error term, so that the model’s learning objective better approximates the true variational lower bound of the data likelihood.
We can visualize the curve of te weights according to $t$, to see whether weights are focused on.

![alt text](../../images/image-49.png)

This seems that the weights are more focused on the $t$ close to zero, which means the model focus more on the denosing steps that is close to the noiseless images.

So far, we have understand the ddpm process. Let's go further on the latent ddpm process
#### Sampling

```py3
    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

```

We give the following explanation for the DDPM sampling steps and corresponds it to the original DDPM reverse process

- **\(x_0\) Estimation:**
  $$
  x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}\, x_t - \sqrt{\frac{1}{\bar{\alpha}_t} - 1}\, \epsilon
  $$

  ↔ `predict_start_from_noise`
  (Coefficients: `sqrt_recip_alphas_cumprod` and `sqrt_recipm1_alphas_cumprod`)

- **Posterior Mean Calculation:**
  $$
  \mu_{\text{posterior}}(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\, \beta_t}{1-\bar{\alpha}_t}\, x_0 + \frac{\sqrt{\alpha_t}\,(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\, x_t
  $$
  ↔ `q_posterior`
  (Coefficients: `posterior_mean_coef1` and `posterior_mean_coef2`)

- **Reverse Process Sampling:**
  $$
  x_{t-1} = \mu_\theta(x_t,t) + \sigma_t\, z,\quad z \sim \mathcal{N}(0,I)
  $$

  ↔ `p_sample`
  (Mean from `p_mean_variance` and variance computed via

  $$e^{0.5 \times \text{model-log-variance}}$$

- **Forward Process Sampling:**
  $$
  x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon
  $$
  ↔ `q_sample`
  (Coefficients: `sqrt_alphas_cumprod` and `sqrt_one_minus_alphas_cumprod`)

Need to mention that, every time, it predicted a $x_0$ and then calculate $x_{t-1}$ according to the forward process.

### Latent DM
In this section, we can just compare the differences between the LDM ad DDPM.

First, in the init stage, it has extra

```py3
        self.instantiate_first_stage(first_stage_config)
```

to initialize the auto encoder which map the original image to the latent space.

In summary, the loss are exactly the same as that of DDPM. The differences comes from the forward step, which

If do not considet the condition, the main difference is

The LDM will first convert image to latent space by the pretrained VAE

```py3
encoder_posterior = self.encode_first_stage(x)
z = self.get_first_stage_encoding(encoder_posterior).detach()
```

All the rest is same.

The codes got a lot of lines dealing with the conditioning including `get_input`, `apply_model`

#### Conditional Latent DM
Next, we focus on how the latent DM put the condition into the network. The training is not guided on the conditioning, it just puts the condition information into the betwork for conditional generation in a implicite architecture.

The diffusion model accepts three different types of conditions

```py3 title='DiffusionWrapper'

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out
```

The conditioning_key have "hybrid", "concat", "crossattn", "adm" types.

- Concat
  In the "Concat" format, the condition information is concated in to the original input, which require the same shape with the input $x$. It can handles the conditions like
  - segmentation map
  - edge map
  - depthmap
  - any spacial condition with the same shape

- CrossAttention
    In this type, the condition should be a sequence of shape [B,L,D]. Which usually handles for conditions
  - texts (ex. from the clip model)
  - class embeddings
  - any conditions in sequence format

    These infomation will be fed into the attention block in the UNet via cross attention

- ADM
    The condition shape if of [B,D] which is a single token embedding. Can handle conditions like
    The information will be assigned to class label $y$.
    In the network, the class label condition is added into the time step condition

    ```py3
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
    ```

  - classfier guided generation
- Hybrid
    Which combines different types of conditions together

#### Sampling with DDPM class
This is exactly the same as that of DDPM sampling except with handling with different conditions
### UNet

=== "SpatialTransformer"

    ```bash

    Input x [B, C, H, W]
        │
        ├────────────────────┐
        │                     │
        ↓                     │
    LayerNorm (self.norm)     │
        │                     │
        ↓                     │
    Projection (self.proj_in) │
    [B, C', H, W]             │
        │                     │
        ↓                     │
    Reshape (rearrange)       │
    [B, (H×W), C']            │
        │                     │
        ↓                     │
    ┌────────────────┐       │
    │Transformer Block│       │
    │  ┌─────────┐   │       │
    │  │Self-Attn│    │       │
    │  └─────────┘   │       │
    │      │          │       │
    │  ┌─────────┐   │       │
    │  │Cross-Att│←──┼────context
    │  └─────────┘   │       │
    │      │          │       │
    │  ┌─────────┐   │       │
    │  │   FF    │    │       │
    │  └─────────┘   │       │
    └────────────────┘       │
        │                     │
        ↓                     │
    Reshape (rearrange)       │
    [B, C', H, W]             │
        │                     │
        ↓                     │
    Projection (self.proj_out)│
    [B, C, H, W]              │
        │                     │
        ↓                     │
        +←──────────────────┘
        │
    Output
    [B, C, H, W]
    ```
=== "Basic Transfomer Block"

    ```
    Input x
    │
    ├─────────────┐
    ↓             │
    LayerNorm1    │
    │             │
    ↓             │
    Self-Attention│
    (attn1)       │
    │             │
    └─────→ + ────┘
            │
            ├─────────────┐
            ↓              │
        LayerNorm2         │
            │              │
            ↓              │
        Cross-Attention    │
        (attn2)            │
           │               │
           └─────→ + ──────┘
                    │
                    ├─────────────┐
                    ↓              │
                    LayerNorm3     │
                    │              │
                    ↓              │
                Feed-Forward       │
                    (ff)           │
                    │              │
                    └────→ + ────┘
                            │
                            ↓
                        Output
    ```

=== "Cross Attention"

    ```
    Input x             Context
    [b, n, dim]       [b, m, dim]
        │                  │
        ↓                  ├─────────────┐
    to_q Linear          ↓             ↓
        │             to_k Linear    to_v Linear
        ↓                  │             │
    [b, n, h*d]    [b, m, h*d]    [b, m, h*d]
        │                  │             │
        ↓                  ↓             ↓
    Reshape to heads  Reshape to heads   Reshape
    [b*h, n, d]      [b*h, m, d]    [b*h, m, d]
        │                  │             │
        │                  │             │
        └──→ Q·K^T * scale ←─┘           │
            [b*h, n, m]                  │
                │                        │
                ↓                        │
            softmax(dim=-1)              │
            [b*h, n, m]                  │
                │                        │
                └──────→ · V ←─────────┘
                        │
                        ↓
                    [b*h, n, d]
                        │
                        ↓
                Reshape to batch
                [b, n, h*d]
                        │
                        ↓
                Linear Output
                        │
                        ↓
                [b, n, query_dim]
    ```

Based on the above basic block, the overall structure is
![alt text](../../images/unet.png)

### DDIM sampling

```py3
"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
```

In DDIM (Denoising Diffusion Implicit Models), the reverse (denoising) step from \(x_t\) to \(x_{t-1}\) is given by:

1. **Predicting \(x_0\):**

   $$
   \hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar{\alpha}_t}},
   $$

   where:
   - \(x_t\) is the current noisy image.
   - \(\epsilon_\theta(x_t,t)\) is the model’s predicted noise.
   - \(\bar{\alpha}_t\) is the cumulative product of the \(\alpha\) values up to time \(t\).

2. **Computing the DDIM Update:**

   $$
   x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\,\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2}\,\epsilon_\theta(x_t,t) + \sigma_t\,z,\quad z\sim\mathcal{N}(0,I),
   $$

   where:
   - \(\sqrt{\bar{\alpha}_{t-1}}\) scales the predicted \(x_0\).
   - \(\sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2}\) scales the noise prediction.
   - \(\sigma_t\) (controlled via the parameter \(\eta\)) governs the stochasticity; if \(\sigma_t=0\) (i.e. \(\eta=0\)), the process is deterministic.
   - \(z\) is standard Gaussian noise.

The code implements this DDIM denoising process primarily in the function `p_sample_ddim`. Let’s break down the correspondence:

1. **Prediction of \(x_0\):**

   - **Theoretical Formula:**
     $$
     \hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar{\alpha}_t}}
     $$

   - **Code:**

     ```python
     pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
     ```

     - Here, `x` represents \(x_t\).
     - `e_t` is the model’s noise prediction \(\epsilon_\theta(x,t)\).
     - `a_t` holds the value of \(\bar{\alpha}_t\) (so that `a_t.sqrt()` is \(\sqrt{\bar{\alpha}_t}\)).
     - `sqrt_one_minus_at` represents \(\sqrt{1-\bar{\alpha}_t}\).

2. **Computing the Direction Term:**

   - **Theoretical Term:**
     \(\sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2}\,\epsilon_\theta(x_t,t)\)

   - **Code:**

     ```python
     dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
     ```

     - `a_prev` corresponds to \(\bar{\alpha}_{t-1}\) (so that `a_prev.sqrt()` is \(\sqrt{\bar{\alpha}_{t-1}}\)).
     - `sigma_t` is the standard deviation \(\sigma_t\) at time \(t\).
     - This computes the directional component by scaling the predicted noise \(e_t\).

3. **Adding the Noise Term and Forming \(x_{t-1}\):**

   - **Theoretical Formula:**
     $$
     x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\,\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2}\,\epsilon_\theta(x_t,t) + \sigma_t\,z
     $$

   - **Code:**

     ```python
     noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
     x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
     ```

     - `a_prev.sqrt() * pred_x0` implements the \(\sqrt{\bar{\alpha}_{t-1}}\,\hat{x}_0\) term.
     - `dir_xt` is the directional term.
     - The variable `noise` represents \(\sigma_t\,z\), where `noise_like(...)` samples from a standard Gaussian distribution.
     - The `temperature` parameter can further modulate the noise level.

4. **Handling Conditioning and Guidance (Optional):**

   - The code also includes conditional sampling and guidance:

     ```python
     if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
         e_t = self.model.apply_model(x, t, c)
     else:
         x_in = torch.cat([x] * 2)
         t_in = torch.cat([t] * 2)
         c_in = torch.cat([unconditional_conditioning, c])
         e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
         e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
     ```

     - This branch adjusts the noise prediction \(e_t\) to steer the generation process in conditional settings. Although not part of the base DDIM equations, it is a common extension for improving sample quality.

5. **DDIM Timesteps and Schedule:**

   - The function `make_schedule` precomputes the necessary parameters (such as `ddim_alphas`, `ddim_alphas_prev`, `ddim_sigmas`, etc.) based on the chosen DDIM discretization schedule.
   - These parameters represent the cumulative products \(\bar{\alpha}_t\) and their variants, ensuring that the updates in `p_sample_ddim` use the correct coefficients at each timestep.

In summary

- **Prediction of \(x_0\):**
  The code computes
  $$
  \hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar{\alpha}_t}}
  $$
  via

  ```python
  pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
  ```

- **Directional Component and Noise Addition:**
  The update for \(x_{t-1}\) is given by
  $$
  x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\,\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2}\,\epsilon_\theta(x_t,t) + \sigma_t\,z,
  $$
  which is implemented as:

  ```python
  dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
  noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
  x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
  ```

- **Control of Stochasticity:**
  The parameter \(\eta\) (used when creating `ddim_sigmas`) controls \(\sigma_t\). When \(\eta=0\), \(\sigma_t=0\) and the process is deterministic.

- **Additional Conditioning:**
  The code includes mechanisms for conditional generation and classifier-free guidance, which modify the noise prediction before performing the update.

This detailed comparison shows how the DDIM denoising process in the code mirrors the theoretical formulas, translating the mathematical update steps into practical sampling operations.
