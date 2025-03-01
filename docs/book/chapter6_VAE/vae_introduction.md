# VAE 理论

## 1. $p_\theta$ 为高斯分布

### 1.1 Variational Autoencoder (VAE) 中关于 $P_\theta(x|z)$ 是高斯分布的假设

在 VAE 的框架中，解码器 \( P_\theta(x|z) \) 通常被假设为 **高斯分布**。这种假设是 VAE 的基础之一，对模型的重建误差定义和优化目标至关重要。以下是关于这个假设的相关知识。

---

### 1.2 为什么假设 $P_\theta(x|z)$ 为高斯分布？

#### 1.2.1 简化问题

- 高斯分布是一种连续型概率分布，数学性质良好，易于计算。
- 对于连续型数据（如图像像素值、音频信号等），高斯分布能很好地拟合大多数数据点的波动特性。

#### 1.2.2 符合重建误差的定义

- 高斯分布的对数似然具有以下形式：

  $$
  \log P_\theta(x|z) = -\frac{\|x - \mu_\theta(z)\|^2}{2\sigma^2} - \frac{1}{2}\log(2\pi\sigma^2)
  $$

  - 这里，$\mu_\theta(z)$ 是解码器预测的高斯分布均值，$\sigma^2$ 是方差（通常可以假设为常量或由解码器预测）。
  - 最大化 $\log P_\theta(x|z)$ 等价于最小化均方误差（MSE）：

    $$
    \|x - \mu_\theta(z)\|^2
    $$

#### 1.2.3 允许建模数据的不确定性

- 假设 $P_\theta(x|z)$ 为高斯分布，解码器不仅预测重构的均值 $\mu_\theta(z)$，还可以通过方差 $\sigma^2_\theta(z)$ 捕捉数据的不确定性。
- 方差的引入有助于避免过度拟合，尤其是在训练数据存在噪声的情况下。

---

### 1.3 数学形式

#### 1.3.1 解码器分布
在 VAE 中，解码器定义为条件概率分布 $P_\theta(x|z)$，假设为高斯分布：

$$
P_\theta(x|z) = \mathcal{N}(x; \mu_\theta(z), \sigma_\theta^2(z))
$$

- $\mu_\theta(z)$：解码器预测的重构均值，通常由神经网络建模。
- $\sigma_\theta^2(z)$：解码器预测的重构方差，可以是固定常量，也可以由网络建模。

---

## 2. 其他分布

### 2.1 拉普拉斯分布与 $L_1$ 范数

- **假设分布**:
  $P_\theta(x|z) = \text{Laplace}(x; \mu_\theta(z), b)$
- **对数似然**:

  $$
  \log P_\theta(x|z) = -\frac{\|x - \mu_\theta(z)\|_1}{b} - \log(2b)
  $$

- **损失函数**:

  $$
  L = \frac{1}{b}\|x - \mu_\theta(z)\|_1
  $$

- **对应形式**: 绝对误差（MAE）。

### 2.2 伯努利分布与交叉熵

- **假设分布**:
  $P_\theta(x|z) = \text{Bernoulli}(x; p_\theta(z))$
- **对数似然**:

  $$
  \log P_\theta(x|z) = x \log p_\theta(z) + (1-x) \log (1-p_\theta(z))
  $$

- **损失函数**:

  $$
  L = -[x \log p_\theta(z) + (1-x) \log (1-p_\theta(z))]
  $$

- **对应形式**: 二元交叉熵损失。

### 2.3 多项分布与多分类交叉熵

- **假设分布**:
  $P_\theta(x|z) = \text{Categorical}(x; \mathbf{p}_\theta(z))$
- **对数似然**:

  $$
  \log P_\theta(x|z) = \sum_i x_i \log p_{\theta,i}(z)
  $$

- **损失函数**:

  $$
  L = -\sum_i x_i \log p_{\theta,i}(z)
  $$

- **对应形式**: 多分类交叉熵。

### 2.4 混合高斯分布

- **假设分布**:
  $P_\theta(x|z) = \sum_k \pi_k \mathcal{N}(x; \mu_k(z), \sigma_k^2(z))$
- **对数似然**:

  $$
  \log P_\theta(x|z) = \log \sum_k \pi_k \mathcal{N}(x; \mu_k(z), \sigma_k^2(z))
  $$

- **特点**: 用于多模态数据建模，计算损失需要数值近似。

---

### 2.5 总结

- **核心思想**:
  损失函数与概率分布的关系为我们提供了统一的视角，用于设计和优化机器学习模型。常见损失函数（如 MSE、MAE、交叉熵）均可以从对应的分布假设中推导而来。

- **实际应用**:
  根据数据特性选择合适的分布假设和损失函数可以提高模型的性能。例如：
  - 连续值数据适用高斯分布（MSE）。
  - 二值数据适用伯努利分布（交叉熵）。
  - 稀疏数据适用拉普拉斯分布（MAE）。

---

### 2.6 相关论文

1. **《Auto-Encoding Variational Bayes》**
   - **作者**: Kingma, D.P., Welling, M.
   - **链接**: [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)
   - **内容**: 提出变分自编码器（VAE），推导出重建误差项与分布假设的关系。

2. **《Hybridised Loss Functions for Improved Neural Network Generalisation》**
   - **作者**: Matthew C. Malan 等
   - **链接**: [https://arxiv.org/abs/2204.12241](https://arxiv.org/abs/2204.12241)
   - **内容**: 探讨交叉熵和均方误差的混合损失及其影响。

3. **《p-Huber损失函数及其鲁棒性研究》**
   - **作者**: 余博天
   - **链接**: [https://pdf.hanspub.org/AAM20201200000_75579140.pdf](https://pdf.hanspub.org/AAM20201200000_75579140.pdf)
   - **内容**: 研究 p-Huber 损失在有噪声数据中的表现。
