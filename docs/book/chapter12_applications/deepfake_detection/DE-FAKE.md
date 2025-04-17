# DE-FAKE: Detecting Text-to-Image Diffusion Fakes using Prompt-based Learning (DE-FAKE)

- **作者**：Zeyang Sha
- **年份**：2023
- **机构**：Salesforce Research
- **论文链接**：[arXiv](https://arxiv.org/pdf/2301.07833)

---

## 论文背景与动机

随着文本到图像生成模型（如Stable Diffusion）的兴起，伪造图像的多样性和复杂性大幅提升，传统检测方法面临新挑战。DE-FAKE关注于如何利用文本prompt信息辅助伪造检测，提升检测的泛化能力和归因能力。

## 方法原理与实现细节

- **核心思想**：
  - 提出三大关键问题：
    1. 伪造图像能否被区分？
    2. 能否归因伪造图像的生成源？
    3. 哪类prompt更易生成真实感图像？
  - 设计两类检测器：
    1. **Image only detector**：仅输入图像
    2. **Hybrid detector**：输入图像及其prompt
- **检测方法与实验设置**：
  - 训练：在一个text2image模型生成的图像上训练二分类器，在其他模型生成的图像上测试泛化。
  - **Image only detector**：
    - 数据集：随机采样2万张MSCOCO真实图像，Stable Diffusion生成2万张伪造图像（带prompt）
    - 网络结构：ResNet18，二分类输出
  - **Hybrid detector**：
    - 数据集同上
    - 网络结构：预训练CLIP提取图像和文本embedding，拼接后送入2层MLP
    - Caption：优先用数据集自带caption，否则用BLIP生成
- **归因实验（Fake Image Attribute）**：
  - 目标：预测伪造图像的生成器来源（SD/LD/GLIDE/真实）
  - 网络结构与检测类似，输出四分类
  - 结果：Hybrid检测器在归因任务上优于Image only
- **Prompt分析**：
  - **语义分析**：
    - 主题分组：用MSCOCO 80类分组，发现"skis" "snowboard"及动物类最易生成高真实感伪造
    - 嵌入聚类：BERT+DBSCAN聚类，发现"person"相关prompt最易生成高真实感伪造
    - 典型prompt分析：详细物体描述优于环境描述
  - **结构分析**：
    - Prompt长度：25-75词效果最佳，过短或过长均较差
    - 名词比例：与图像真实感无显著相关

## 实验设置与结果分析

- Hybrid检测器显著优于Image only，尤其在跨生成器泛化和归因任务上。
- 归因实验表明多模态特征有助于识别生成源。
- Prompt分析揭示了哪些语义和结构特征更易生成高真实感伪造。
- ![DE-FAKE检测流程与实验结果](../../images/image-154.png)

## 主要贡献与不足

- **贡献**：
  - 提出多模态检测与归因分析框架，系统分析prompt对检测的影响。
- **不足**：
  - Image only分支未用更强主干，公平性有待提升。

## 个人点评/启示

多模态特征融合和prompt语义为伪造检测提供了新思路。未来可探索更复杂的多模态融合和跨模态对抗鲁棒性。