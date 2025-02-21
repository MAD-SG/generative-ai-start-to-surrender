# Generative Category

In this article, we will list the main categories of generative models from different perspectives.

## Main Categories

- Variational Autoencoder (VAE)
- Generative Adversarial Network (GAN)
- Discrete Diffusion Models
- Continuous Diffusion Models
- Energy/Score Based Generative Models
- Normalizing Flow (NF)
- Continuous Normalizing Flow (CNF)
- Flow Matching (FM)
- RL-Augmented Generative Models
- Optical Physics

##

**连续归一化流（CNF）：** CNF 的训练目标是最大化模型生成数据的对数似然，即通过最大似然估计（MLE）来优化模型参数。具体而言，CNF 通过定义一个时间相关的向量场，使数据从初始分布（如高斯分布）平滑地变换到目标分布（如真实数据分布），然后通过数值求解常微分方程（ODE）以评估模型生成数据的对数似然。然后，通过最小化负对数似然损失函数，优化模型参数，使生成的数据尽可能接近真实数据的分布。 [YMSHICI.COM](https://www.ymshici.com/tech/2146.html?utm_source=chatgpt.com)

**流匹配（Flow Matching，FM）：** FM 的训练目标是学习一个时间相关的向量场，使其能够匹配从初始分布到目标分布的概率流的动态特性。具体而言，FM 通过定义一个参考概率路径，并优化一个匹配损失函数，该损失函数衡量模型预测的向量场与真实向量场之间的差异。通过最小化该损失，模型学习到一个向量场，使得从初始分布到目标分布的变换过程中的速度场最小化与参考路径的差异。 [BILIBILI.COM](https://www.bilibili.com/read/cv38989899?utm_source=chatgpt.com) **主要区别：**

- **损失函数：** CNF 采用最大似然估计，直接对数据的对数似然进行优化；而 FM 通过最小化模型向量场与参考路径向量场之间的差异来训练模型。

- **训练过程：** CNF 需要数值求解 ODE，并计算雅可比行列式的对数行列式，这可能带来较高的计算成本；而 FM 通过匹配速度场，避免了对 ODE 的数值求解，从而提高了训练效率。

综上，虽然 CNF 和 FM 都旨在将初始分布转换为目标分布，但它们在训练目标和方法上有所不同。CNF 侧重于直接优化数据的对数似然，而 FM 则侧重于学习一个向量场，使其与参考的概率流路径相匹配。

源![Favicon](https://www.google.com/s2/favicons?domain=https://www.bilibili.com&sz=32)
![Favicon](https://www.google.com/s2/favicons?domain=https://www.ymshici.com&sz=32)
