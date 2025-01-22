# From GAN to PGGAN: Some Base GAN model Introduction

# [2014]GAN : [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)

GANs (Generative Adversarial Networks) are based on an adversarial process framework. In this framework, there are two networks that work in opposition: a generator and a discriminator. This can be compared to a scenario where one team (generator) tries to produce counterfeit items without being detected, while the other team (discriminator) acts like law enforcement trying to detect the fakes.
Overall Introduction

- Base model structure
  ![image](https://hackmd.io/_uploads/BkDsNVoDyx.png)
*Image Source: [Semi Engineering - GAN Knowledge Center](https://semiengineering.com/knowledge_centers/artificial-intelligence/neural-networks/generative-adversarial-network-gan/)*



  
  - The model works through two independent learning models: a Generative Model and a Discriminative Model, which learn through adversarial training to produce high-quality outputs.
    - Generative Model,short as $G$：
      - Takes random noise $z$ as input -> Generates images $G(z)$
      - Aims to create images that look real enough to fool $D$
    - Discriminative Model, short as $D$,is a binary classifier:
      - Takes an image x as input -> Output $D(x)$, representing the probability that $x$ is a real image
        - $D(x)=1$ means 100% confidence it's real
        - $D(x)=0$ means it's definitely fake
- Adversarial training: The optimization process is a minimax game, with the goal of reaching Nash equilibrium
  
  - The generator tries to minimize the probability of the discriminator detecting fake samples
    The discriminator tries to maximize its ability to distinguish between real and fake samples
## **Loss function**:
  
  $$V(D,G) = \underset{D}{\max} {\underbrace{\mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1-D(G(z)))]}_{\text{Discriminator Loss}:L_D}} +\underset{G}{\min} {\underbrace{\mathbb{E}_{z \sim p_z(z)}[\log(1-D(G(z)))]}_{\text{Generator Loss: }L_G}}$$
  
  Where:
  - $p_{data}(x)$ is the real data distribution
  - $p_z(z)$ is the noise distribution
  - $G(z)$ is generator mapping from noise to synthetic data
  - $D(x)$ is the discriminator's estimate of the probability that $x$ is real

overall training process:
  ![image](https://hackmd.io/_uploads/S1ROHEiwke.png)
*Image Source: Goodfellow et al., "Generative Adversarial Networks" (2014) arXiv:1406.2661*


  
- In GAN training, we iterate the Discriminator ($D$) $K$ times before updating the Generator ($G$) once：
  
  - because we need $D$ to be powerful enough to provide accurate feedback for $G's$ improvement
  - This iterative strategy helps maintain training stability and prevent mode collapse, although K needs to be carefully balanced 
      - too large and G can't learn effectively, too small and D's feedback becomes unreliable.

## Base model structure

Code reference: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py

```Python
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
            
        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img
```

```Python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
```
**Training process and images generated:**

![image](https://hackmd.io/_uploads/HJX7I4jDyl.png)
*Image Source: Goodfellow et al., "Generative Adversarial Networks" (2014) arXiv:1406.2661*

- Real data distribution $p_{data}$(Black dotted line)
- Generator distribution $p_g$(Green solid line )
- Discriminator output $D$(Blue dashed line)
- Noise space where $z$ is sampled (Lower horizontal line)
- Data space $x$(Upper horizontal line)
- Generator G's mapping from  $z$ to  $x$ (Arrows connecting lines)

From the picture, we can see the training Evolution (from a to d):

- Initial Stage:

  - The generated distribution  $p_g$ (green) differs significantly from the real distribution $p_{data}$ (black)

  - Discriminator $D$  (blue) attempts to distinguish samples, but performs unstably

- Discriminator Training: $D$ is trained to reach optimal solution: $D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$

- Generator Update:  $G$ updates based on gradients from $D$

- Final Convergence: When $G$ and $D$ have sufficient capacity（ $p_{data} =p_g$), they reach Nash equilibrium


![image](https://hackmd.io/_uploads/HJn4UVswye.png)
*Image Source: Goodfellow et al., "Generative Adversarial Networks" (2014) arXiv:1406.2661*


# [2014][cGAN: Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)
## Overall Introduction: 

Conditional generation
Traditional GANs produce samples from random noise but **can't control the output features**, as they are **unsupervised learning**.

While conditional GANs (cGANs) **incorporate conditional information into both the generator and discriminator, enabling control over the output properties**. This is achieved through a semi-supervised approach.

The cGAN paper only shows its generated results on the MNIST dataset, where simply concatenating label embeddings might have limited impact. However, the core idea of "guiding the generation process with conditional information" proposed by cGAN has significantly influenced subsequent generative models. 

- For example, models like DALL-E and Stable Diffusion, although utilizing different architectures like Diffusion, have adopted the principle of conditional generation: they use text embeddings as conditional information to control image generation.

## Base model structure 
![image](https://hackmd.io/_uploads/Hy6cONjwkx.png)
*Source: Mirza et al., "Conditional Generative Adversarial Nets" (2014) arXiv:1411.1784*

How to combine the condition into input:

- First convert categorical labels into continuous vector representations using nn.Embedding

  - `self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)`

- Then concatenates torch.cat label embeddings(y) with noise vectors(z) along dimension -1: 

  - `gen_input = torch.cat((self.label_emb(labels), noise), -1)`

- Uses this concatenated vector as input to generate images through multiple network layers

```python
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img
```

IN Discriminator:

- Flattens input images: 

    - `img.view(img.size(0), -1)`

- Similarly, processes labels through embedding: `self.label_embedding(labels)`

- Concatenates flattened images and label embeddings:

    - `d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)`

- Passes concatenated vector through discriminator network for real/fake classification

```python 
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            nn.Linear(opt.n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity
```
Actually, it's still a Binary Classification Task. The output of cGAN discriminator still maintains GAN's binary output (real/fake). 

- Doesn't explicitly verify condition-image matching
- Output is a single scalar through Sigmoid/BCELoss or MSELoss

For example, an input condition: number "7". If the generator generates an image that looks like "3". Although the discriminator will not directly point out "this is not 7", because the label of "7" in the training data has never been paired with the image of "3". So this wrong match will be identified as "generated" by the discriminator.

## Loss function 

$$\min_{G} \max_{D} V(D,G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x|y)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z|y)))]$$

Where: $\mathbb{E}$: Expected value (expectation) $\mathbb{E}_{x \sim p_{data}(x)}[\log D(x|y)]$ : 

- $x \sim p_{data(x)}$:  $x$sampled from real data distribution

- $D(x|y)$: Discriminator's output for real data  $x$ given condition  $y$

- $E[\log D(x|y)]$ - Discriminator's ability to identify real samples

$\mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z|y)))]$ : 

- $z \sim p_z(z)$ z sampled from noise distribution

- $G(z|y)$: Generator's output from noise z given condition y

- $E[log(1 - D(G(z|y)))]$ - Discriminator's ability to identify fake samples

```python 
for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        batch_size = imgs.shape[0]
        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))
        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))
        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)
        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)
        g_loss.backward()
        optimizer_G.step()
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)
        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)
        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

```
![image](https://hackmd.io/_uploads/SJkkKEjvkl.png)
*Source: Mirza et al., "Conditional Generative Adversarial Nets" (2014) arXiv:1411.1784*


---
# [2015]DCGAN(Deep Convolutional GAN）：[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)

## Overall Introduction:  Deep Convolutional GAN

DCGAN integrates the strengths of Convolutional Neural Networks into GANs with key innovations:

1. Convolutional Layers: Transposed convolutions in the generator and strided convolutions in the discriminator enhance spatial information retention.

2. Batch Normalization: Used extensively in both parts to improve stability and prevent mode collapse.

3. Activation Functions: The generator uses ReLU with a Tanh final layer, and the discriminator employs LeakyReLU.

- Starts from 100-dimensional noise z, gradually generating $64×64$ images through multiple convolution layers

- Feature map progression(C*H*W): 

  - $100\times1\times1$->$1024\times4\times4$ -> $512\times8\times8$->$256\times16\times16$->$128\times32\times32$->$3\times64\times64$
  
## Base model structure
![image](https://hackmd.io/_uploads/SytbYEsPkl.png)
*Source: Radford et al., "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" (2016) arXiv:1511.06434*

About Transposed convolutions:

- Input Matrix Expansion: Initially, the input feature map undergoes an expansion by inserting zeros between each element. The number of zeros inserted depends on the stride parameter. 

- Application of the Convolution Kernel: Next, the convolution kernel is applied to the expanded feature map. This process is similar to traditional convolution operations, where the kernel slides over the expanded feature map, computing the dot product with local regions. Unlike regular convolution, this operation results in a larger output feature map because the input has been expanded.

- Adjustment of Output Size: Finally, the output feature map might be cropped or padded to adjust its dimensions to the desired size. This adjustment depends on the padding parameter, which can either reduce or increase the spatial dimensions of the output

## Loss function 

Same as GAN objective : $\min_G \max_D V(D,G) = \mathbb{E}_{x\sim p_{data}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]$


---

# [2017] WGAN:[ Wasserstein GAN](https://arxiv.org/abs/1701.07875)

## Overall Introduction:  

WGAN introduces Wasserstein distance and Lipschitz constraint  in loss function to "improve the stability of learning, get rid of problems like mode collapse, and provide meaningful learning curves useful for debugging and hyperparameter searches". WGAN replaces the JSD with the Wasserstein distance to measure the distribution distance in the original GAN.

- If the discriminator is trained too well, the generator's gradients vanish, and the generator's loss cannot decrease 

- If the discriminator is not trained well enough, the generator's gradients become inaccurate, causing it to move erratically. 

The discriminator needs to be trained to just the right degree - neither too well nor too poorly - but this balance is very difficult to achieve. Moreover, this optimal balance might even vary at different stages within the same training epoch, which is why GANs are so difficult to train.

## Drawbacks as JSD：

### Gradient Vanishing Problem & training instability

We have introduced above, under an (approximately) optimal discriminator, minimizing the generator's loss is equivalent to minimizing the JS divergence between $P_r$ and $P_g$. Since $P_r$ and $P_g$ almost inevitably have negligible overlap, their JS divergence will always be the constant $\log 2$, regardless of how far apart they are. This ultimately leads to the generator's gradient (approximately) becoming 0, resulting in gradient vanishing.
![image](https://hackmd.io/_uploads/SyN95VovJl.png)
*Source: Arjovsky et al., "Wasserstein GAN" (2017) arXiv:1701.07875*

### Model collapse

Secondly, even the previously mentioned standard KL divergence term has flaws. Because KL divergence is not a symmetric measure, $KL(P_g\|P_r)$ and $KL(P_r\|P_g)$ are different. 
Taking the former as an example: $KL(P_g||P_r) = \int_x P_g(x)\log(\frac{P_g(x)}{P_r(x)})dx$

- When $P_g(x) \to 0$ and $P_{r}(x) \to 1$, $P_g(x)\log\frac{P_g(x)}{P_{r}(x)} \to 0$, contributing nearly 0 to $KL(P_g||P_r)$

- When $P_g(x) \to 1$ and $P_{r}(x) \to 0$, $P_g(x)\log\frac{P_g(x)}{P_{r}(x)} \to +\infty$, contributing positively infinite to $KL(P_g||P_r)$

- In other words, $KL(P_g||P_r)$ penalizes these two types of errors differently. 

- The first type of error corresponds to **"generator failing to generate real samples"** with small penalty.

- The second type corresponds to **"generator generating unrealistic samples"** with large penalty. 

The first type of error represents a lack of diversity, while the second type represents a lack of accuracy. As a result, the generator would rather generate some repetitive but "safe" samples, and is reluctant to generate diverse samples, because one small mistake could lead to the second type of error, resulting in an unacceptable loss. This phenomenon is commonly referred to as mode collapse.

![image](https://hackmd.io/_uploads/SJAaYViwJg.png)
*Source: Arjovsky et al., "Wasserstein GAN" (2017) arXiv:1701.07875*



Why Wasserstein distance?

The superiority of the Wasserstein distance compared to KL divergence and JS divergence lies in its ability to reflect the proximity between two distributions even when they don't overlap. While KL divergence and JS divergence are discontinuous , being either maximum or minimum. The Wasserstein distance is smooth and offers a more natural way to measure distances between distributions.

1. Training Stability: Provides meaningful gradients even when distributions do not overlap, significantly improving the stability of GAN training.

2. Reduced Mode Collapse: Encourages diversity in generated samples by considering the overall differences between distributions, reducing mode collapse.

3. Intuitive Loss Function: Serves as a loss metric, where a smaller Wasserstein distance indicates closer alignment with the target distribution's statistical properties.

4. Effective GAN Training: WGANs use Wasserstein distance to offer a more stable and effective training process, enhancing the quality and diversity of generated samples.


## Loss function 
1. About Wasserstein Distance (measure distance by estimating the difference between expectation):
  - Mathematical Definition ：$W(P,Q) = \inf_{\gamma \in \Pi(P,Q)} \mathbb{E}_{(x,y)\sim \gamma}[||x-y||]$
    - $\Pi(P_r, P_g)$ is the set of all possible joint distributions whose marginal distributions are $P_r$ and $P_g$. 
    - In other words, for each distribution in $\Pi(P_r, P_g)$, its marginal distributions are $P_r$ and $P_g$. 
    - For each possible joint distribution $\gamma$,sample $(x,y) \sim \gamma$ to get a real sample $x$ and a generated sample $y$, and calculate the distance between these samples $\|x-y\|$. 
    - $\mathbb{E}_{(x,y)\sim\gamma}[\|x-y\|]$calculates the expected value of the sample distance under this joint distribution. The infimum of this expected value among all possible joint distributions $\inf_{\gamma\sim\Pi(P_r,P_g)} \mathbb{E}_{(x,y)\sim\gamma}[\|x-y\|]$ is defined as the Wasserstein distance.
    - Intuitively, $\mathbb{E}_{(x,y)\sim\gamma}[\|x-y\|]$ can be understood as the "cost" of moving "$P_r$ pile of earth" to " $P_g$ location" under this "transport plan", and $W(P_r, P_g)$ is the "minimum cost" under the "optimal transport plan", which is why it's called the Earth-Mover distance.
2. Lipschitz Constraint
  - A function f is called Lipschitz continuous if it satisfies:$$|f(x) - f(y)| \leq C|x - y|$$
    - where $C$ is the Lipschitz constant.
      - When inputs x and y are close to each other, their corresponding outputs $f(x)$ and $f(y)$ must also be close. This property ensures smoothness and continuity in the function
  - As for a discriminator in WGAN, D is constrained to be 1-Lipschitz functions: $|D(x) - D(y)| \leq |x - y|$
3. Wasserstein Distance in WGAN:
  - GAN objective function :$$\min_{G} \max_{D} V(D,G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$
  - In WGAN, the objective function can be written as:$$W(P_r, P_g) = \sup_{||f||_L \leq 1} \mathbb{E}_{x \sim \mathbb{P}_r}[D(x)] - \mathbb{E}_{x \sim \mathbb{P}_g}[D(x)]$$
  where ：
    - D is constrained to be 1-Lipschitz functions.
    - $\mathbb{P}_r$ represents the distribution of real data (real probability distribution)
      - $x\sim\mathbb{P}_r$ means x is sampled from the real data distribution
    - $\mathbb{P}_g$ represents the distribution of generated data (generated probability distribution)
      - ${x}\sim\mathbb{P}_g$ means x is sampled from the generator's distribution
    - WGAN tries to minimize the Wasserstein distance between $\mathbb{P}_r$ and $\mathbb{P}_g$ 
    - $||f||_L \leq 1$ means that the discriminator D must satisfy the Lipschitz condition.
    


---
how do we get the loss function:

* Primal Form (Original Wasserstein Distance):

  $$W(P,Q) = \inf_{\gamma \in \Pi(P,Q)} \mathbb{E}_{(x,y)\sim \gamma}[||x-y||]$$

  where $\Pi(P,Q)$ is the set of all joint distributions (couplings) whose margins are P and Q.

* Kantorovich Duality Theorem:According to the duality Theorem, this problem is equivalent to:

$$W(P,Q) = \sup_{f\in Lip_1} \left(\int f\,dP - \int f\,dQ\right)$$
where $Lip_1$ is the set of 1-Lipschitz functions.

*  Expectation Form:Converting the integrals to expectations:

$$W(P,Q) = \sup_{||f||_L \leq 1} [\mathbb{E}_{x\sim P}[f(x)] - \mathbb{E}_{x\sim Q}[f(x)]]$$

*  Application to GAN: When P = Pr (real distribution) and Q = Pg (generated distribution):

$$W(P_r,P_g) = \sup_{||f||_L \leq 1} [\mathbb{E}_{x\sim P_r}[f(x)] - \mathbb{E}_{x\sim P_g}[f(x)]]$$

---

4. Weight Clipping : -> implementation of Lipschitz Constraint in WGAN

- WGAN forces the discriminator to satisfy the Lipschitz constraint through weight clipping or gradient penalty.

- After each gradient update, the weights of the critic (discriminator) are clipped to a fixed range [-c, c] 

  - In the paper, c = 0.01
  
```python
# 在每次参数更新后执行
for param in discriminator.parameters():
    param.data.clamp_(-c, c)  # c通常设为0.01
Weight Clipping -> all weight values are forced to be limited to the range of [-0.01, 0.01].

```

- Any value outside this range will be "clipped" to the boundary value. 

- This ensures the Lipschitz constraint of the network, but may also lead to limitations in expressiveness.

**"Weight clipping is a clearly terrible way to enforce a Lipschitz constraint."** -- M. Arjovsky, S. Chintala and L. Bottou, "Wasserstein Generative Adversarial Networks," in International Conference on Machine Learning, 2017, pp. 214-223.

## Training process -> Wasserstein & Lipschitz  Gradient

![image](https://hackmd.io/_uploads/SkSZ5NovJl.png)
*Source: Arjovsky et al., "Wasserstein GAN" (2017) arXiv:1701.07875*



1. Discriminator  Gradient:

The Wasserstein loss for the Discriminator is:$$L(w) = \mathbb{E}_{x \sim \mathbb{P}_r}[f_w(x)] - \mathbb{E}_{z \sim p(z)}[f_w(g_\theta(z))]$$
where:

* $f_w(x)$ Discriminator evaluates real data samples

* $x\sim\mathbb{P}_r$ means x is sampled from the real data distribution

* $f_w(g_θ(z))$ Discriminator evaluates generated data samples

    * $z$ is random noise transformed by generator $g_θ$ 

    * $g_θ(z)$ represents the Generator generated samples
    
For a batch of size m, the empirical version becomes:
$$L(w) = \frac{1}{m}\sum_{i=1}^m f_w(x^{(i)}) - \frac{1}{m}\sum_{i=1}^m f_w(g_\theta(z^{(i)}))$$

Therefore, the gradient with respect to Discriminator parameters w is:

$$\nabla_w L = \frac{1}{m}\sum_{i=1}^m \nabla_w f_w(x^{(i)}) - \frac{1}{m}\sum_{i=1}^m \nabla_w f_w(g_\theta(z^{(i)}))$$

2. Generator Gradient:

The generator's objective is to minimize:

$$L(θ) = -\mathbb{E}_{z \sim p(z)}[f_w(g_\theta(z))]$$

For a batch of size m, this becomes:

$$L(θ) = -\frac{1}{m}\sum_{i=1}^m f_w(g_\theta(z^{(i)}))$$

The gradient with respect to generator parameters θ is:

$$\nabla_\theta L = -\frac{1}{m}\sum_{i=1}^m \nabla_\theta f_w(g_\theta(z^{(i)}))$$

- Line 5: Discriminator gradient computation
$g_w ← \nabla_w [\frac{1}{m}\sum_{i=1}^m f_w(x^{(i)}) - \frac{1}{m}\sum_{i=1}^m f_w(g_\theta(z^{(i)}))]$

- Line 10: Generator gradient computation $g_\theta ← -\nabla_\theta \frac{1}{m}\sum_{i=1}^m f_w(g_\theta(z^{(i)}))$

```python
for i, (imgs, _) in enumerate(dataloader):
        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        # Generate a batch of images
        fake_imgs = generator(z).detach()
        # Adversarial loss
        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))
        loss_D.backward()
        optimizer_D.step()
        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)
        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            # Generate a batch of images
            gen_imgs = generator(z)
            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs))
            loss_G.backward()
            optimizer_G.step()
```

![image](https://hackmd.io/_uploads/r1Q694jv1e.png)
*Source: Arjovsky et al., "Wasserstein GAN" (2017) arXiv:1701.07875*


---

# [2017] WGAN-GP: [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
## Overall Introduction:  gradient penalty 

WGAN-GP replaces the weight clipping in the original WGAN by adding a gradient penalty term at the random interpolation points between the real and generated data, thereby achieving a more stable training process and better generation effects.

Loss function in WGAN: 
$$L = \sup_{||f||_L \leq 1} \mathbb{E}_{x \sim \mathbb{P}_r}[D(x)] - \mathbb{E}_{x \sim \mathbb{P}_g}[D(x)]$$

Loss function in WGAN-GP: 
$$L = \mathbb{E}_{x\sim P_r}[D(x)] - \mathbb{E}_{x\sim P_g}[D(x)] + \lambda \mathbb{E}_{\hat{x}\sim P_{\hat{x}}}[(||\nabla_{\hat{x}}D(\hat{x})||_2 - 1)^2]$$

where:
- Wasserstein Distance Term :$\mathbb{E}_{x\sim P_r}[D(x)] - \mathbb{E}_{x\sim P_g}[D(x)]$

  - measure distance between real and generated distributions

- Gradient Penalty Term: $\lambda \mathbb{E}_{\hat{x}\sim P_{\hat{x}}}[(||\nabla_{\hat{x}}D(\hat{x})||_2 - 1)^2]$

  - λ is penalty coefficient (typically 10)

  - Ensures gradient norm is close to 1

  - $\hat{x}$ is a random interpolation between real samples and generated samples:

### Drawbacks as weight-clipping


![image](https://hackmd.io/_uploads/SJUHiEiD1g.png)
*Source: Gulrajani et al., "Improved Training of Wasserstein GANs" (2017) arXiv:1704.00028*


#### Capacity underuse

- Main Issues:

  - Theoretically, this critic should maintain unit gradient magnitudes everywhere, but when using weight clipping constraints, the critic in WGAN tends to learn overly simplistic functions. 

- Experimental Validation:

  - To verify this, we conducted experiments using the real distribution plus random noise as the generator's output. 

  - The results showed that critics with weight clipping indeed overlook the complex features of the data, learning only simple approximations. 

#### Exploding and vanishing gradients

- Main Issues:

  - WGAN faces optimization challenges during training, caused by the interaction between weight constraints and the loss function. If the clipping threshold $c$ is not carefully adjusted, it may lead to either vanishing or exploding gradients.

- Experimental Validation:

  - Researchers conducted experiments on the Swiss Roll toy dataset using three different clipping thresholds: 0.1, 0.01, and 0.001. With weight clipping:

    - At $c=0.1$, gradients exhibited exponential growth (red line going up).

    - At $c=0.01$ and $c=0.001$, gradients exhibited exponential decay (purple and green lines going down).

The two smaller graphs on the right show differences in weight distribution:

- The upper graph: Weight clipping pushes weights toward two extreme values.

- The lower graph: Gradient penalty results in a more normal distribution of weights.

## Loss function 

Training process:
![image](https://hackmd.io/_uploads/H1KOo4sw1l.png)
*Source: Gulrajani et al., "Improved Training of Wasserstein GANs" (2017) arXiv:1704.00028*

**How does the Gradient Penalty term work in WGAN-GP？**

1. The loss function of WGAN-GP:  $\min_G \max_D \mathbb{E}_{x\sim P_r}[D(x)] - \mathbb{E}_{z\sim P_z}[D(G(z))] + \lambda \mathbb{E}_{\hat{x}\sim P_{\hat{x}}}[(||\nabla_{\hat{x}}D(\hat{x})||_2 - 1)^2]$

2. The core idea of the Gradient Penalty term is to enforce the 1-Lipschitz constraint on discriminator D across the sample space： $|D(x_1) - D(x_2)| \leq |x_1 - x_2|$

3. The 1-Lipschitz constraint above is equivalent to having the gradient norm of the discriminator not exceeding 1 at any point:  $||\nabla_x D(x)||_2 \leq 1$

4. WGAN-GP enforces the gradient norm to be equal to 1, rather than less than or equal to 1, through the penalty term:$\mathcal{L}_{GP} = \lambda \mathbb{E}_{\hat{x}\sim P_{\hat{x}}}[(||\nabla_{\hat{x}}D(\hat{x})||_2 - 1)^2]$

  - $\hat{x} = \epsilon x + (1-\epsilon)G(z), \epsilon \sim U[0,1]$

    - So $\hat{x}$ is a linear interpolation between data points of the real data distribution $P_r$ and the generated data distribution $P_g$.

    - Why sampling?

      - According to $||\nabla_x D(x)||_2 \leq 1$, the optimal critic forms a line between the paired points of the real and generated distributions with a gradient norm of 1. Therefore, as a compromise, the constraint is only enforced along these sampled lines.

      - Easy to implement and worked out in experiments.

  $\hat{x}$:
```python
alpha = torch.rand(real_samples.size(0), 1, 1, 1)
interpolates = alpha * real_samples + (1 - alpha) * fake_samples
```
  $\nabla_{\hat{x}}D(\hat{x})$:
```python
#d(x)
d_interpolates = D(interpolates)  
# \nabla_{\hat{x}}D(\hat{x}):
gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True
    )[0]
```

   $(||\nabla_{\hat{x}}D(\hat{x})||_2 - 1)^2$:
   
```python 
gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
```


5. The regulatory effect of this penalty term is manifested in:

  - When $||\nabla_{\hat{x}}D(\hat{x})||_2 > 1$, showing:

    - Too Steep Gradients: Discriminators tend to be "aggressive" in judging real/fake samples and change too rapidly 

    - May lead to training instability:

      - Likely to cause discriminator overfitting

      - Provides too strong gradient signals to the generator

  - When $||\nabla_{\hat{x}}D(\hat{x})||_2 < 1$, showing:

    - Too Flat Gradients: The Discriminator tends to be  insensitive to input changes 

    - Insufficient Discrimination :  The Discriminator cannot effectively distinguish real/fake samples

    - Vanishing Gradients: Generator might not receive effective training signals
    
  - Only when gradient norm  $||\nabla_{\hat{x}}D(\hat{x})||_2 = 1$, the penalty term becomes zero
  
```python 
def compute_gradient_penalty(D, real_samples, fake_samples):
    # Random interpolation coefficient
    alpha = torch.rand(real_samples.size(0), 1, 1, 1)
    # Generate interpolated samples
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)
    # Compute discriminator output
    d_interpolates = D(interpolates)  
    # Compute gradients
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True
    )[0]
    # Compute gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Training loop
for epoch in epochs:
    for real_data in dataloader:
        # Compute gradient penalty
        gradient_penalty = compute_gradient_penalty(D, real_data, fake_data)
        # Discriminator loss
        d_loss = -torch.mean(D(real_data)) + torch.mean(D(fake_data)) + lambda_gp * gradient_penalty
```

![image](https://hackmd.io/_uploads/S1MnAVowJl.png)
*Source: Gulrajani et al., "Improved Training of Wasserstein GANs" (2017) arXiv:1704.00028*

![image](https://hackmd.io/_uploads/rygjCViDyx.png)
*Source: Gulrajani et al., "Improved Training of Wasserstein GANs" (2017) arXiv:1704.00028*


---

# [2018] PGGAN: [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)

## Overall Introduction:

1. Progressive Growing

  * Core Idea: Start at low resolution and progressively increase to higher resolutions.

  * Advantages:  More stable training & Higher computational efficiency & Better memory utilization

  * Implementation: Smoothly fade in new layers and synchronous growth of the generator and discriminator

2. Minibatch Standard Deviation 

  * Purpose: Increase the diversity of generated images & prevents mode collapse.

  * Implementation: 

    * Introduce a statistical layer late in the discriminator

    * Calculate the standard deviation within a minibatch of samples

    * Concatenate statistical features with the original features

3. Normalization Strategies

  * Purpose: strategies ensure underlying training stability.

  * Implementation: 

    * Generator: Uses PixelNorm 

The structure of PGGAN laid an important foundation for subsequent work (such as StyleGAN).

## PROGRESSIVE GROWING OF GANS
![image](https://hackmd.io/_uploads/H1g1JBjvJx.png)
*Source: Karras et al., "Progressive Growing of GANs for Improved Quality, Stability, and Variation" (2018) arXiv:1710.10196*

![image](https://hackmd.io/_uploads/SJvW1HsDye.png)
*Source: Karras et al., "Progressive Growing of GANs for Improved Quality, Stability, and Variation" (2018) arXiv:1710.10196*

* Each resolution stage has two phases:

  1. Fade-in Phase:

    * The new layer is gradually blended in using alpha parameter

    * Alpha increases linearly from 0 to 1

      * In PGGAN, the growth of the α parameter is linear and is controlled by the number of training iterations. This is achieved as follows:

```python
# 假设fade_in_iters是fade-in阶段的总迭代次数
fade_in_iters = 600000  # 600k images
# 当前迭代次数current_iter
alpha = min(current_iter / fade_in_iters, 1.0)

def fade_in(self, alpha, upscaled, generated):
    return torch.tanh(alpha * generated + (1-alpha) * upscaled)
......
final_upscaled = self.rgb_layers[steps-1](upscaled)
final_out = self.rgb_layers[steps](out)
return self.fade_in(alpha, final_upscaled, final_out)
```

  2. Stabilization Phase: 

    * Train network with new layers fully active

    * Old paths are removed

    * Network stabilizes at new resolution

Time Allocation:

  * Fade-in Phase: 600k images

  * Stabilization Phase: 600k images

  * Total per resolution: 1.2M images (600k + 600k)

```
Complete Training Process Example (from 4×4 to 1024×1024):
4×4:   Only Stabilization     600k images
8×8:   Fade-in + Stabilization 1.2M images
16×16: Fade-in + Stabilization 1.2M images
32×32: Fade-in + Stabilization 1.2M images
64×64: Fade-in + Stabilization 1.2M images
128×128: Fade-in + Stabilization 1.2M images
256×256: Fade-in + Stabilization 1.2M images
512×512: Fade-in + Stabilization 1.2M images
1024×1024: Fade-in + Stabilization 1.2M images
```
Smooth Layer Transitions:
```python
def forward(self, x, alpha):
    # Old path (lower resolution)
    old_rgb = self.from_rgb_old(x)
    old_rgb = self.upsample(old_rgb)
    
    # New path (higher resolution)
    new_x = self.upsample(x)
    new_x = self.conv(new_x)
    new_rgb = self.to_rgb_new(new_x)
    
    # Smooth blending
    return (1 - alpha) * old_rgb + alpha * new_rgb
```
  - toRGB: 1×1 convolution to convert features to RGB

  - fromRGB: 1×1 convolution to convert RGB to features

  - 2×: Upsampling (nearest neighbor)

  - 0.5×: Downsampling (average pooling)



## INCREASING VARIATION USING MINIBATCH STANDARD DEVIATION

![image](https://hackmd.io/_uploads/S1bEyBjwkx.png)
*Source: Wang et al., "Citrus Disease Image Generation and Classification Based on Improved FastGAN and EfficientNet-B5" (2023) Electronics, 12(5), 1232*

1. For each feature and spatial location i, compute standard deviation across the batch:
  $\sigma_i(x) = \sqrt{\frac{1}{N}\sum_{k=1}^{N}(x_{ik} - \mu_i)^2}$
where:
  - $x_{ik}$ is the feature value for sample k at position i

  - $\mu_i = \frac{1}{N}\sum_{k=1}^{N}x_{ik}$ is the mean across the batch

  - N is the batch size

2. Average the standard deviations across features and spatial dimensions:
  $\sigma = \frac{1}{C \times H \times W}\sum_{i}\sigma_i(x)$

  where:

  - $C$ : channels. $H$: height. $W$ :width

These statistics are then:

1. Replicated into a $[1×1×H×W]$ tensor

2. Further replicated N times to match batch size: $[N×1×H×W]$

3. Concatenated with original input along channel dimension to get final output of shape $[N×(C+1)×H×W]$

```python
def minibatch_stddev_layer(x, group_size=4):
    N, C, H, W = x.shape
    G = min(group_size, N)  # 分组大小
    
    # [NCHW] -> [GMCHW] 分成G组
    y = x.reshape(G, -1, C, H, W)
    
    # 计算标准差
    y = torch.sqrt(y.var(0) + 1e-8)  # [-MCHW]
    
    # 取平均得到单个值
    y = y.mean(dim=[0,1,2,3])  # []
    
    # 广播回原始形状
    y = y.reshape(1, 1, 1, 1)
    y = y.repeat(N, 1, H, W)  # [N1HW]
    
    # 连接到输入特征图
    return torch.cat([x, y], 1)  # [N(C+1)HW]
```
The main advantages of this technique are:

1. Helps the discriminator identify the statistical characteristics of the generated images
2. Encourages the generator to produce more diverse outputs
3. Helps avoid mode collapse

## NORMALIZATION IN GENERATOR AND DISCRIMINATOR
### Normalization in passed GAN-related-model


| GAN model | Normalization applied | Implementation detail                                                                                                                    |
| --------- | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| GAN       | batch normalization   | Typically employs basic batch normalization in both the generator and discriminator.                                                     |
| cGAN      | batch normalization   | Typically employs basic batch normalization in both the generator and discriminator.                                                     |
| DCGAN     | batch normalization   | Generator: BN is used in all layers except the output layer. Discriminator: BN is used in all layers except the input and output layers. |
| WGAN     | Remove BN in the discriminator   |Advises against using batch normalization due to its impact on the Lipschitz constraint. Completely removes BN in the discriminator. The generator may use BN, but it is often omitted in practice.|
| WGAN-GP     | Remove BN in the discriminator   | Discriminator: Recommends using layer normalization or instance normalization instead of batch normalization. This is because BN introduces correlations between samples, affecting the calculation of the gradient penalty.|
| PGGAN     | Completely Remove BN and Pixel-wise normalization in the generator and new weight initialization   | Generator: BN is used in all layers except the output layer. Discriminator: BN is used in all layers except the input and output layers. |


### PIXELWISE FEATURE VECTOR NORMALIZATION IN GENERATOR

Applied after each convolutional layer in the generator at each pixel position independently:

$$b_{x,y} = \frac{a_{x,y}}{\sqrt{\frac{1}{N}\sum_{j=0}^{N-1}(a_{x,y}^j)^2 + \epsilon}}$$

Where:
- $\epsilon = 10^{-8}$

- $a_{x,y} \text{ is the original feature vector at pixel position } (x,y)$

- $b_{x,y} \text{ is the normalized feature vector at pixel position } (x,y)$

- $N \text{ is the number of feature maps (channels)}$

- $\epsilon = 10^{-8} \text{ is a small constant to prevent division by zero}$

- $\text{The sum } \sum_{j=0}^{N-1} \text{ is taken over all } N \text{ feature maps for that pixel position}$

```python
class PixelNorm(nn.Module):
    def forward(self, x):
        norm = torch.mean(x ** 2, dim=1, keepdim=True)
        norm = torch.sqrt(norm + self.epsilon)
        return x / norm    
```
- $x ** 2$:  Calculate the squared values of all features at each pixel $(a_{x,y}^j)^2$

- `torch.mean(..., dim=1)`:  Average these squares across all feature maps ${\frac{1}{N}\sum_{j=0}^{N-1}(a_{x,y}^j)^2 }$

- torch.sqrt(... + epsilon): Take the square root of the average (plus ε) ${\sqrt{\frac{1}{N}\sum_{j=0}^{N-1}(a_{x,y}^j)^2 + \epsilon}}$

- $x / norm$:  normalization $\frac{a_{x,y}}{\sqrt{\frac{1}{N}\sum_{j=0}^{N-1}(a_{x,y}^j)^2 + \epsilon}}$$


### QUALIZED LEARNING RATE

**Problem**: In traditional neural network training, parameters of different layers may have different dynamic ranges. When using adaptive optimizers like RMSProp or Adam, they normalize gradient updates based on the standard deviation of parameters. This results in parameters with larger dynamic ranges requiring more time to adjust properly.

Specific Implementation:

1. Traditional Weight Initialization 
  - In standard neural networks, weights are typically initialized using methods 

    - like He initialization N(0, sqrt(2/n)), Better suited for ReLU

```python
# Standard He initialization (for comparison)
weight_shape = (out_channels, in_channels, kernel, kernel)
std = np.sqrt(2.0 / (in_channels * kernel * kernel))
weights = np.random.normal(0, std, weight_shape)
```
  - This can lead to different layers learning at different rates, causing training instability

  - The variance of the gradients can differ significantly between layers

2. The Equalized Learning Rate Solution: 
  - initialization (happens only once when the model is created):

    - Instead of using usual standard initialization, weights are initialized from N(0,1)

## Equalized learning rate approach

```python
weights = np.random.normal(0, 1.0, weight_shape)
runtime_coef = std  # Applied during forward pass
```

- During training (happens every forward pass): scaling process

- Each layer's weights are explicitly scaled during runtime by a layer-specific constant 

- The scaling factor is the per-layer normalization constant from He initialization

### In EqualizedConv2d

```python
self.scale = np.sqrt(2) / np.sqrt(fan_in)  # Calculate He scaling factor
scaled_weight = self.weight * self.scale    # Apply scaling at runtime
```
     
- This ensures that the dynamic range and learning speed are similar for all weights
- Standard layers: Apply scaling during initialization
    - Equalized layers: Apply scaling during each forward pass
    - This ensures the gradient updates remain properly scaled throughout training
    
Why It's Useful:

1. Ensures that all weights have a similar dynamic range.

2. Makes the learning process more balanced, avoiding slow learning for some parameters due to large ranges.

3. Better adapts to c

- In standard neural networks, weights are typically initialized using methods 

  - like He initialization $N(0, sqrt(2/n))$, Better suited for ReLU

---
RMSProp or Adam, they normalize gradient updates based on the standard deviation of parameters：

RMSprop:
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t + \epsilon}} g_t$$ Adam:$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

1. Standard Deviation Estimation:
  - $v_t$ actually estimates the exponential moving average of squared gradients:$v_t = \beta v_{t-1} + (1-\beta)g_t^2$

  - This accumulated squared gradient $v_t$ is essentially estimating the second moment of the gradient, and its square root $\sqrt v_t$ approximates the standard deviation of the gradient

2. Normalization Effect:
  - In the update formula, the gradient term ($g_t$ or $ĥ_t$) is divided by $\sqrt v_t$

  - This is equivalent to normalizing the gradient update by the gradient's standard deviation

  - Mathematically equivalent to:$$\text{normalized update} = \frac{g_t}{\sqrt{v_t + \epsilon}}$$

3. Why This Is Standard Deviation Normalization:

  - If a parameter has large gradient variations (high standard deviation), $\sqrt v_t$ will become larger

  - This will make the actual update step smaller

  - Conversely, if gradient variations are small (low standard deviation), the update step will become larger accordingly

  - This achieves adaptive standard deviation normalization

      - This is also why EQUALIZED LEARNING RATE solves this problem by explicitly controlling the dynamic range of parameters ($ŵᵢ = wᵢ/c$).
      
    
Main Advantages:

1. Keeps the learning speed consistent for all weights.

2. Avoids the issue of having both too high and too low learning rates at the same time.

3. By dynamically scaling during runtime rather than statically at initialization, it makes the training process more stable.

---
