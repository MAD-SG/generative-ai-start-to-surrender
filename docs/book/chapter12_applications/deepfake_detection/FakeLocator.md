# FakeLocator: Robust Localization of GAN-Based Face Manipulations (FakeLocator)

- **作者**：Yihao Huang
- **年份**：2021
- **机构**：East China Normal University
- **论文链接**：[arXiv](https://arxiv.org/abs/2111.00000)


#### 🌟 核心机制（SCCM）

虽然文中没有显式提出"SCCM"这一缩写，但整体方法可以被理解为一种 **Structure-aware Cross-domain Consistency Mapping（结构感知的跨域一致性映射）**，主要体现在以下几个方面：

##### 伪纹理定位（Fake Texture Detection）
- 利用 GAN 中 decoder 的 **上采样模块（upsampling）** 所引入的"伪纹理"（如 checkerboard pattern），作为 GAN 合成图像中篡改区域的线索。
- 这些伪纹理难以通过传统方法去除，成为了定位伪造区域的重要依据。

##### 灰度 fakeness map（gray-scale fakeness map）
- 相比其他方法使用的二值 map（threshold-based），FakeLocator 使用 **像素级灰度图** 作为标签，能够更精确反映篡改区域的程度与边界。
- 训练标签通过真实图像与伪造图像的像素差异计算获得，避免了人为阈值选择带来的不确定性。

##### 注意力机制（Attention via Face Parsing）
- 使用 **face parsing 网络**（FPM）得到人脸不同部位（如头发、嘴巴等）的分割图。
- 将该注意力图作为引导信号乘入 encoder 中间特征图，迫使网络关注潜在篡改区域，提升 **跨属性泛化能力**（cross-attribute universality）。

##### 部分数据增强（Partial Data Augmentation）
- 只对真实图像做数据增强，不改变伪图像，用以扩大真实图像的分布边界，让模型将"非真实"的图像更加自信地识别为伪造。
- 避免对伪图像增强，以免扰乱 GAN 伪纹理特征。

##### 单样本聚类（Single Sample Clustering）
- 为提升 **跨 GAN 泛化能力（cross-method universality）**，利用 t-SNE + k-means 对编码特征做无监督聚类，再用一个带标签的样本判断两个聚类中哪个是"真"、哪个是"假"。

单样本聚类（Single Sample Clustering）是 FakeLocator 提出的用于提升 **跨-GAN 泛化能力**（cross-method universality）的一种轻量但有效的技巧。它的核心思想是：**在没有已知标签的大量目标域样本的情况下，仅凭一个带标签的样本就可以进行伪造图像检测**。

即使训练了一个在源域 GAN（如 FaceForensics++）上表现良好的检测器，在迁移到目标域 GAN（如 StyleGAN）时，**分类器可能无法分辨真实图与伪图**，尽管 encoder 的输出特征在 t-SNE 可视化中可以分出两个明显的簇。

原因是：
- 分类器在源域上训练的决策边界未必适用于目标域；
- 但 encoder 抽取的特征本身对真假图仍有良好区分性。



###### 🧩 具体做法：Single Sample Clustering

###### Step ：提取特征
- 用训练好的 encoder $ F_{enc} $，对目标 GAN 域的样本提取特征：
  $$
  Z = \{F_{enc}(x_i)\}_{i=1}^N
  $$

###### Step ：降维
- 用 t-SNE 把这些特征降到二维或三维，得到 $ Z_{tsne} $

###### Step ：无监督聚类
- 对 $ Z_{tsne} $ 使用 **k-means**（$k=2$）聚成两个簇：
  $$
  \text{Cluster}_0, \text{Cluster}_1
  $$

###### Step ：使用**单个有标签样本**识别真假簇
- 随机取一个 **带标签的样本**（real 或 fake），比如：
  $$
  x^*, \quad y^* \in \{0, 1\}
  $$
  把 $ x^* $ 做降维和聚类，查它属于哪个簇。
- 然后将该簇作为"真"或"假"，并由此判断所有其他样本的标签。
###### Step ：评估与重复
- 为减少随机误差，重复十次，每次选一个随机样本，求平均准确率。



###### 🎯 优点

| 优点 | 说明 |
|------|------|
| 无需目标域分类器 | 规避了跨域泛化性差的问题 |
| 不需要大量标注数据 | 只需一个有标签样本 |
| 充分利用了 encoder 的特征区分能力 | 解耦 encoder 与 classifier，迁移更灵活 |

##### 🧪 效果示例（表格见 Table I）

| 训练域 | 测试域 | 原始准确率 | +Partial Aug | +Clustering |
|--------|--------|--------------|---------------|-------------|
| FaceForensics++ | StarGAN | 0.490 | 0.988 | **0.999** |
| FaceForensics++ | PGGAN | 0.369 | 0.500 | **0.975** |

#### 🧮 Loss 设计

使用的是 **L1 / L2 回归损失** 而不是 Dice/Focal segmentation loss，这是因为灰度 fakeness map 本质是一个连续的强度预测任务而非分割任务。

损失函数：
- L1 loss: $ \text{L1} = \frac{1}{n} \sum_{i,j} |M^{pred}_{i,j} - M^{gt}_{i,j}| $
- 另外加入了分类分支的交叉熵 loss，以计算 detection 准确率。

---

#### 📊 结果与结论

- 在 FaceForensics++, Celeb-DF, DFFD 等多个数据集上超越了现有方法（如 [27]、[30]）。
- 同时具备：
  - 高分辨率 fakeness map 输出
  - 强鲁棒性（抗 JPEG 压缩、模糊、低分辨率、噪声等）
  - 跨属性与跨 GAN 泛化能力

---

#### ✅ 值得借鉴的点

| 借鉴点 | 应用价值 |
|--------|----------|
| 利用上采样缺陷作为伪造纹理特征 | 可以推广到所有基于 GAN 的生成图像检测中 |
| 灰度 fakeness map 表达更丰富 | 更适用于精细定位任务 |
| attention + face parsing | 利用语义信息指导特征提取，提升泛化性 |
| partial augmentation + clustering | 实用的跨域增强与迁移技巧 |
| 不依赖原始 GAN 结构 | 易于迁移到其它类型图像伪造检测 |

