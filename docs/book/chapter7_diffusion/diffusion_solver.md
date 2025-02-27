# Diffusion Solver

Before going to the deep details, let's check the preliminary in [Diffusion Unified Representation](../chapter7_diffusion/sde_diffusion_unified_representation.md)
## DPM solver

- paper: <https://arxiv.org/pdf/2206.00927>
- repo: <https://github.com/LuChengTHU/dpm-solver>
- author: 路橙（Cheng Lu）

  - 路橙现任 OpenAI 的技术成员，主要研究方向为大规模深度生成模型和强化学习算法。他于 2023 年 12 月在清华大学 TSAIL 团队获得计算机科学与技术博士学位，导师为朱军教授。在博士期间，他还与陈建飞和李崇轩密切合作。本科阶段，他于 2019 年 7 月获得清华大学计算机科学与技术学士学位。他的研究兴趣包括一致性模型、扩散模型、归一化流和能量基模型，以及它们在图像生成、3D 生成和强化学习中的应用。此外，他曾是清华大学合唱团的男低音歌手，并在 2019 年通过演唱著名爵士歌曲《Autumn Leaves》获得清华大学校园十大歌手比赛的冠军
- Key:
  - 提出 DPM-Solver，一种专门针对扩散常微分方程（ODEs）的高阶求解器，它通过解析计算线性部分并将解简化为神经网络的指数加权积分，大幅提升了Diffusion Probabilistic Models (DPMs) 的采样效率，在无需额外训练的情况下，将采样步骤减少至 10-20 次，并在多个数据集上实现 4× 至 16× 的加速。

Recall the Probability ODE

$$\tag{1}
\begin{aligned}
\frac{d x_t}{d t} &= f(t)x_t + \frac{g(t)^2}{2\sigma_t}\epsilon_\theta  \\
& = \frac{d \log \alpha_t}{dt} x_t + \left( \frac{\sigma_t d \log \sigma_t/\alpha_t}{dt} \right) \epsilon_\theta\\
& = \frac{d \log \alpha_t}{dt} x_t - \left( \frac{\sigma_t d \lambda_t}{2dt} \right) \epsilon_\theta\\
\end{aligned}
$$

Here $\lambda_t =\log \frac{\alpha_t^2}{\sigma_t^2}$ usually means the signal-to-noise ratio, SNR., is a monotanous decreasing function of t.

Here is our base ODE, we will use it to build our DPM Solver

值得注意的是  (1) is a semi-linear ODE, we can have a general solution formula

!!! thm "general solution of semi-linear ODE"

    The solution of (1) is of form

    $$\tag{2}
    \boxed{x_t=
    \exp\!\Bigl[\!\int_{s}^{t}f(\tau)\,d\tau\Bigr]\;x_s
    \;+\;
    \int_{s}^{t}
    \exp\!\Bigl[\!\int_{\tau}^{t}f(r)\,dr\Bigr]\,
    \frac{g^2(\tau)}{2\,\sigma_\tau}\,\epsilon_{\theta}\bigl(x_\tau,\tau\bigr)
    \,d\tau.}
    $$

!!! proof "proof of equation (2)"

    下面演示如何从

    $$
    \frac{dx}{dt}
    \;=\;
    f(t)\,x(t)
    \;+\;
    \frac{g^2(t)}{2\,\sigma_t}\,\epsilon_\theta\bigl(x(t),\,t\bigr),
    $$
    推导出所给的积分形式

    $$
    x_t
    \;=\;
    \exp\!\Bigl[\!\int_{s}^{\,t}f(\tau)\,d\tau\Bigr]\,
    x_s
    \;+\;
    \int_{s}^{t}
    \exp\!\Bigl[\!\int_{\tau}^{\,t}f(r)\,dr\Bigr]\,
    \frac{g^2(\tau)}{2\,\sigma_\tau}\,\epsilon_{\theta}\bigl(x_\tau,\tau\bigr)\,d\tau.
    $$
    为方便阅读，下文把 \(x(t)\) 在 \(t = s\) 时刻的值记为 \(x_s\)，在 \(t = t\) 时刻的值记为 \(x_t\)。

    #### 1. 写成一阶**线性**常微分方程

    原方程可视为
    $$
    x'(t) - f(t)\,x(t)
    \;=\;
    \frac{g^2(t)}{2\,\sigma_t}\,\epsilon_\theta\bigl(x(t),t\bigr).
    $$
    这是一个**非齐次一阶线性 ODE**，其中“非齐次项”为
    \(\tfrac{g^2(t)}{2\,\sigma_t}\,\epsilon_{\theta}\bigl(x(t),t\bigr)\)。


    #### 2. 乘以积分因子并取全导数

    **积分因子**(Integrating Factor) 取
    $$
    \mu(t)
    \;=\;
    \exp\!\Bigl[\,-\!\int_{s}^{\,t}f(u)\,du\Bigr].
    $$
    将上式两边同乘 \(\mu(t)\):

    $$
    \exp\!\Bigl(-\!\int_{s}^{\,t}f(u)\,du\Bigr)
    \,x'(t)
    \;-\;
    f(t)\,
    \exp\!\Bigl(-\!\int_{s}^{\,t}f(u)\,du\Bigr)\,
    x(t)
    \;=\;
    \exp\!\Bigl(-\!\int_{s}^{\,t}f(u)\,du\Bigr)\,
    \frac{g^2(t)}{2\,\sigma_t}\,\epsilon_\theta(x(t),t).
    $$
    左端恰好是对
    \(\displaystyle x(t)\,\mu(t)\)
    做时间导数的结果：
    $$
    \frac{d}{dt}
    \Bigl[
    x(t)\,
    \exp\!\Bigl(-\!\int_{s}^{\,t}f(u)\,du\Bigr)
    \Bigr].
    $$
    因此方程化为

    $$
    \frac{d}{dt}
    \Bigl[
    x(t)\,\mu(t)
    \Bigr]
    \;=\;
    \mu(t)\,\frac{g^2(t)}{2\,\sigma_t}\,\epsilon_\theta\bigl(x(t),t\bigr).
    $$



    #### 3. 在区间 \([s,t]\) 上积分

    对 \(t\) 从 \(s\) 到 \(t\) 积分：

    $$
    \bigl[\,
    x(\tau)\,\mu(\tau)
    \bigr]_{\,\tau=s}^{\,\tau=t}
    \;=\;
    \int_{s}^{t}
    \mu(\tau)\,\frac{g^2(\tau)}{2\,\sigma_\tau}\,\epsilon_\theta\bigl(x(\tau),\tau\bigr)
    \,d\tau.
    $$
    也就是
    $$
    x_t\,\mu(t)
    \;-\;
    x_s\,\mu(s)
    \;=\;
    \int_{s}^{t}
    \mu(\tau)\,
    \frac{g^2(\tau)}{2\,\sigma_\tau}\,
    \epsilon_\theta\bigl(x_\tau,\tau\bigr)\,d\tau.
    $$

    ##### 3.1. 代入

    $$\mu(s)=\exp\!\Bigl(-\!\int_{s}^{\,s} f(u)\,du\Bigr)=1$$

    显然 \(\int_{s}^{\,s}(\cdots)\,du=0\)，故 \(\mu(s)=e^0=1\)。因此

    $$
    x_t\,\mu(t)
    \;-\;
    x_s
    \;=\;
    \int_{s}^{t}
    \mu(\tau)\,
    \frac{g^2(\tau)}{2\,\sigma_\tau}\,
    \epsilon_\theta\bigl(x_\tau,\tau\bigr)\,d\tau.
    $$
    从而

    $$
    x_t\mu(t)=x_s+\int_{s}^{t}\mu(\tau)\frac{g^2(\tau)}{2\,\sigma_\tau}\epsilon_\theta\bigl(x_\tau,\tau\bigr)d\tau
    $$

    ##### 3.2. 还原 \(x_t\)

    回忆 \(\mu(t)=\exp\!\bigl[-\!\int_s^t f(u)\,du\bigr]\)，所以

    $$
    x_t= \exp\!\Bigl[\!\int_{s}^{\,t}f(u)\,du\Bigr]x_s+\int_{s}^{t}
    \exp\!\Bigl(\!\int_{s}^{\,t}f(u)\,du\Bigr)\,
    \mu(\tau)\,
    \frac{g^2(\tau)}{2\,\sigma_\tau}\,\epsilon_\theta(x_\tau,\tau)\,d\tau.
    $$

    但要注意，

    $$
    \exp\!\Bigl(\!\int_{s}^{\,t}f(u)\,du\Bigr)\,\mu(\tau)
    \;=\;
    \exp\!\Bigl(\!\int_{s}^{\,t}f(u)\,du\Bigr)
    \;\exp\!\Bigl(-\!\int_{s}^{\,\tau}f(u)\,du\Bigr),
    $$

    实际上我们更直接的做法是：分拆

    $$
    \int_{s}^{\,t}f(u)\,du
    \;=\;
    \int_{s}^{\,\tau}f(u)\,du
    \;+\;
    \int_{\tau}^{\,t}f(u)\,du,
    $$

    因而

    $$
    \exp\!\Bigl[\!\int_{s}^{\,t}f(u)\,du\Bigr]
    \;\exp\!\Bigl[-\!\int_{s}^{\,\tau}f(u)\,du\Bigr]
    \;=\;
    \exp\!\Bigl[\!\int_{\tau}^{\,t}f(u)\,du\Bigr].
    $$

    所以第二项在被乘以 \(\exp[\int_s^t f(u)\,du]\) 后，可以写成

    $$
    \int_{s}^{t}
    \exp\!\Bigl[\!\int_{\tau}^{\,t}f(r)\,dr\Bigr]\,
    \frac{g^2(\tau)}{2\,\sigma_\tau}\,\epsilon_\theta(x_\tau,\tau)
    \,d\tau.
    $$

    整理得到最后公式

    综上便得到了所需的积分形式解（在标准文献里也叫“Duhamel 原理”形式

    $$
    \boxed{x_t=
    \exp\!\Bigl[\!\int_{s}^{t}f(\tau)\,d\tau\Bigr]\;x_s
    \;+\;
    \int_{s}^{t}
    \exp\!\Bigl[\!\int_{\tau}^{t}f(r)\,dr\Bigr]\,
    \frac{g^2(\tau)}{2\,\sigma_\tau}\,\epsilon_{\theta}\bigl(x_\tau,\tau\bigr)
    \,d\tau.}
    $$

同时，我们带入方程(1), 得到

$$
x_t = \frac{\alpha_t}{\alpha_s} x_s -\frac{ \alpha_t }{2} \int_s^t \frac{d \lambda_\tau}{d\tau} \frac{\sigma_\tau}{\alpha_\tau} \epsilon_\theta\bigl(x_\tau,\tau\bigr)\,d\tau
$$

因为$\lambda_t$ 是单调递减的，它具有逆函数，然后我们进行变量替换

$$ t \rightarrow \lambda$$

我们有

$$d\lambda = \frac{d\lambda_t}{d t} dt$$

因此

!!! note "Exact Solution of Diffusion ODE"

    $$\tag{3}
    x_t =\frac{\alpha_t}{\alpha_s} x_s -\frac{ \alpha_t }{2} \int_{\lambda_s}^{\lambda_t} e^{-\frac{\lambda }{2}} \hat{\epsilon_\theta}\bigl(x_\lambda,\lambda\bigr)\,d\lambda
    $$

因此根据这个公式，我们可以得到线性部分的准确解，当然随机部分还是需要进行积分。但是它至少减少了一部分的误差项。同时我们也可以从另外一个角度理解，可以理解成 $\epsilon_\theta$ 的一个加权平均,而且是指数衰减的，$\lambda$ 越大，贡献越小。也就是$t$ 越大， $\lambda$ 越小，贡献越大.

## DPM Solver ++
