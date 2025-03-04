# Diffusion Anomaly Detection
In this article, we will introduce the application of diffusion model sin the field of anomaly detection.

## Diffusion Models for Medical Anomaly Detection

- author: Julia Wolleb, University of Basel
- Year 2022 Oct
Main idea
![alt text](../../images/image-116.png)
It do the $L$ forward noising steps, and then do $L$ denoising steps.

- Don't need the paired data
- Trained a classification network first
- In the denosing step, use guidance diffusion mode, with a gradient scale $s$ to control the classifier effects
![alt text](../../images/image-117.png)
The gradient scale and the nosing step $L$ need to chosen properly, otherwise, could leads to in-accurate reconstruction.
![alt text](../../images/image-118.png)
![alt text](../../images/image-119.png)

## AnoDDPM:Anomaly Detection with Denoising Diffusion Probabilistic Models using Simplex Noise

- author: Julian Wyatt， Durham University
- url: <https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Wyatt_AnoDDPM_Anomaly_Detection_With_Denoising_Diffusion_Probabilistic_Models_Using_Simplex_CVPRW_2022_paper.pdf>
- year 2022
- code: <https://github.com/JulianWyatt/AnoDDPM>

Contribution

- trained on pure positive data: no anomaly data is used in the training
- partial diffusion: nosing the image to a fixed ratio, not to noising to pure noise
- AnoDDPM with simplex noise

Algorithm

![alt text](../../images/image-120.png)

Results

![alt text](../../images/image-121.png)
## Unsupervised Surface Anomaly Detection with Diffusion Probabilistic Model

- author: zhang xinyi, Xia Shutao
- institute: 清华大学深圳研究生院
- ICCV 2023: /<https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Unsupervised_Surface_Anomaly_Detection_with_Diffusion_Probabilistic_Model_ICCV_2023_paper.pdf>

Current Problem

- Reconstruction is ill-conditioned, which means small variations in the input will lead to large variations
- given test sample might resemble several different norma (non-anomalous) patterns, rather than just one specific pattern. Example: In medical imaging, a normal brain scan may have multiple small variations due to different people’s anatomy, lighting, or imaging conditions. If the model only learns to reconstruct a single dominant pattern, it might misinterpret small variations as anomalies.

Framework

![alt text](../../images/image-122.png)

![alt text](../../images/image-123.png)

![alt text](../../images/image-124.png)

Training

- MVTec-AD datasets with mask annotations
- Need the paired normal and anormal data
  - dataset overview ![alt text](../../images/image-125.png)
  - <https://www.mvtec.com/company/research/datasets/mvtec-ad>

## Fast Unsupervised Brain Anomaly Detection and Segmentation with Diffusion Models

- author: Walter H. L. Pinaya,
- institute: University College London
- MICCAI: 2022

灌水论文

![alt text](../../images/image-126.png)
还是使用Latent Diffusion 的模式，对latent 加噪再去噪，从而重建原图。

## ON DIFFUSION MODELING FOR ANOMALY DETECTION

- ICLR 2024: <https://openreview.net/pdf?id=lR3rk7ysXz>
- author: Victor Livernoche
- institute: McGill University

DTE: Diffusion Time Estimation
估计了给定输入所需要的diffusion time的分布，（预测的时间不是一个值，是时间的分布),然后用众数或均值作为异常分数。

之前的diffusion 用来做异常检测效果都比较好，唯一的问题是计算量比较大。一般的DDPM anormal detection 是利用denoised reconstruction 和原始图片的距离，距离越大，越有可能是异常。

- 预测distance 和预测diffusion time 是否是一致的？
- 预测了扩散时间（或噪声方差）的后验分布？ 扩散时间指的是什么？

Contribution

- 逆扩散过程中，选择初始时间步（timestep）是任意的，但它会显著影响异常检测性能。经验结果表明，选择最大时间步的 25% 作为起始点可以获得较好的检测效果（详细消融实验见附录 A）。
??

## Unsupervised industrial anomaly detection with diffusion models✩
![alt text](../../images/image-127.png)

- paper: <https://drive.google.com/file/d/1cuVGoQo_K6JasrLoEVlfjj-sEXFQgxE0/view>
- author: XU haohao
- J. Vis. Commun. Image R. 97 (2023) 103983
- Contribution
  - use the feature embedding of the original image as a guidance in the reverse diffusion process
Results
![alt text](../../images/image-128.png)

## Patched Diffusion Models for Unsupervised Anomaly Detection in Brain MRI

- code: <https://github.com/> FinnBehrendt/patched-Diffusion-Models-UAD
- Finn Behrendt: Hamburg University of Technology
- Proceedings of Machine Learning Research: 2023
Results
![alt text](../../images/image-129.png)
framework
![alt text](../../images/image-130.png)

we apply the forward diffusion process only on a small part of the input image and use the whole, partly noised image in the backward process to recover the noised patch. At test time, we use the trained pDDPM to sequentially noise and denoise a sliding patch within the input image and then stitch the individual denoised patches to reconstruct the entire image

## Adversarially Robust Industrial Anomaly Detection Through Diffusion Model
![alt text](../../images/image-131.png)
- author: Yuanpu Cao, Lu Lin, Jinghui Chen
- 2024 Aug
- 贡献:
  - 免疫对抗噪声并且同时检测异常

基于深度学习的工业异常检测模型在常用的基准数据集上已取得了极高的准确率。然而，由于对抗样本（adversarial examples）的存在，这些模型的鲁棒性（robustness） 可能并不理想，而对抗样本对深度异常检测器的实际部署构成了重大威胁。

近年来，研究表明扩散模型（diffusion models） 可以用于去除（purify）对抗噪声，从而构建对抗攻击鲁棒的分类器。然而，我们发现，在异常检测中直接应用这一策略（即在异常检测器前放置一个去噪器） 会导致较高的异常漏检率（high anomaly miss rate）。这是因为去噪过程不仅会消除对抗扰动，同时也会抹去异常信号，最终导致后续的异常检测器无法有效检测出异常样本。

为了解决这一问题，我们探索了同时执行异常检测和对抗去噪的可能性。我们提出了一种简单但有效的对抗鲁棒异常检测方法（Adversarially Robust Anomaly Detection, AdvRAD），该方法使扩散模型同时充当异常检测器和对抗噪声去除器。
