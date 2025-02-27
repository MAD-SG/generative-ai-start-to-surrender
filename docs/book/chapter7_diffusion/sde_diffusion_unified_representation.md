# Unified Representation of SDE Diffusion

## Unified Representation

Diffusion 模型的过程里，我们希望能够建里一个前向的加噪过程，随着时间的增加，噪音的强度会逐渐减弱。具体的用公式来表达就是

$$\tag{1}x_t = \alpha_t x_0 + \sigma_t \epsilon_t,\quad \epsilon_t \sim \mathcal{N}(0, \mathbf{I})$$

其中$\alpha_t$ 和 $\sigma_t$ 分别表示时间$t$ 时刻的前向进度和噪音强度, 我们分别表示noted as "(Signal) scaling"和"(Noise) scheduling"。他们之间的关系可以独立也可以相互影响。

另外的从SDE的角度考虑diffusion，我们构建的是一个分布转移的过程，用SDE表示为

$$\tag{2} d x_t = f(t) x_t d t + g(t) d w_t,\quad x_0 \sim p_{data}(x)$$

初始分布是给定的数据分布，但是目标分布根据扩散过程而不同，对于VP-SDE,目标分布是一个unit normal distribution，对于VE-SDE,目标分布是一个标准差越来越大的高斯分布。

如果我们只关心 ”scheduling" 和 "sclaing", 那么公式(2)也可以用 $\sigma_t$ 和 $\alpha_t$ 表达,具体为

$$\tag{3}
\begin{aligned}
f(t) &= \frac{d \log \alpha_t}{dt},\\
g(t) &= \sqrt{ \frac{d  \sigma_t^2}{dt} - 2 \frac{d \log \alpha_t}{dt}\sigma_t^2}\\
\end{aligned}
$$

相应的逆过程SDE为

$$\tag{4}
\begin{aligned}
d x_t &= \left[ f(t) x_t - g^2(t) \nabla_x \log q_t(x_t)\right] d t + g(t) d \bar{w_t},\\
&=\left[ \frac{d \log \alpha_t}{dt} x_t - \left( \frac{d  \sigma_t^2}{dt} - 2 \frac{d \log \alpha_t}{dt}\sigma_t^2 \right) \nabla_x \log q_t(x_t)\right] d t + \sqrt{ \frac{d  \sigma_t^2}{dt} - 2 \frac{d \log \alpha_t}{dt}\sigma_t^2} d \bar{w_t}\\
\end{aligned}
$$

因此我们只需要设计$\alpha_t$ 和$\sigma_t$,就能决定一个扩撒过程，就能确定前向和逆向的采样。

利用denoising score matching 的方法，我们可以计算出条件score function 在给定$x_0$ 下,.

$$\tag{5}
\begin{aligned}
\nabla_x \log q_t(x_t|x_0) & = \nabla_x \log \left[ C e^{ - \frac{(x_t - \alpha_t x_0)^2}{2 \sigma_t^2}} \right]\\
& = \frac{1}{\sigma_t^2}(x_t - \alpha_t x_0)\\
& = - \frac{\epsilon_t}{\sigma_t}
\end{aligned}
$$

然后根据Probability FLow ODE, 我们可以得到 逆向采用的ODE、

$$\tag{6}
\begin{aligned}
\frac{d x_t}{d t} &= f(t)x_t - \frac{1}{2} g(t)^2\nabla_x \log q_t(x_t)\\
&= \frac{d \log \alpha_t}{dt} x_t -\frac{1}{2} \left( \frac{d  \sigma_t^2}{dt} - 2 \frac{d \log \alpha_t}{dt}\sigma_t^2 \right) \nabla_x \log q_t(x_t)\\
& = \frac{d \log \alpha_t}{dt} x_t  +\left(  \frac{d  \sigma_t}{dt} -  \frac{d \log \alpha_t}{dt}\sigma_t \right) \epsilon_t\\
& = \frac{d \log \alpha_t}{dt} x_t + \left( \frac{\sigma_t d \log \sigma_t/\alpha_t}{dt} \right) \epsilon_t\\
& = \frac{d \log \alpha_t}{dt} x_t - \left( \frac{\sigma_t d e^{\lambda_t/2}}{dt} \right) \epsilon_t\\
\end{aligned}
$$

here $\lambda_t = \frac{\alpha_t^2}{\sigma_t^2}$ usually means the signal-to-noise ratio, SNR.

??? proof "proof of equation (3)"

    1. 显式描述与 SDE 描述

          - **显式描述：**
          我们有过程
          $$
          x_t = \alpha_t x_0 + \sigma_t\,\epsilon_t,\quad \epsilon_t \sim \mathcal{N}(0,\mathbf{I}).
          $$
          因此，
          - 均值为 \(\mathbb{E}[x_t] = \alpha_t x_0\)；
          - 方差为 \(\operatorname{Var}(x_t) = \sigma_t^2\).

          - **SDE 描述：**
          该过程也可以表示为 SDE：
          $$
          dx_t = f(t)x_t\,dt + g(t)\,dw_t,\quad x_0\sim p_{data}(x).
          $$


    2. 均值的推导

          - 对 SDE 取数学期望（注意 \( \mathbb{E}[dw_t]=0 \)），得：
          $$
          \frac{d}{dt}\mathbb{E}[x_t] = f(t)\,\mathbb{E}[x_t].
          $$
          - 令 \( m(t)=\mathbb{E}[x_t] \) 且 \( m(0)=x_0 \)，解得
          $$
          m(t) = x_0 \exp\Big(\int_0^t f(s)\,ds\Big).
          $$
          - 由显式描述得 \( m(t)=\alpha_t x_0 \)，因此
          $$
          \alpha_t = \exp\Big(\int_0^t f(s)\,ds\Big).
          $$
          - 对数后求导，得
          $$
          f(t) = \frac{d\log\alpha_t}{dt}.
          $$

    3. 方差的推导

        1. 利用 Itô 引理推导 \(x_t^2\) 的演化

              - 对 \(x_t^2\) 应用 Itô 引理：

               $$
               d(x_t^2) = 2x_t\,dx_t + (dx_t)^2.
               $$

              - 代入 \(dx_t = f(t)x_t\,dt + g(t)\,dw_t\) 得：

                $$
                \begin{equation}
                d(x_t^2) = 2x_t\Big(f(t)x_t\,dt+g(t)\,dw_t\Big) + g(t)^2\,dt = 2f(t)x_t^2\,dt+2g(t)x_t\,dw_t+g(t)^2\,dt.
                \end{equation}
                $$

              - 取数学期望（利用 \(\mathbb{E}[x_t\,dw_t]=0\)）得到：

                $$
                \frac{d}{dt}\mathbb{E}[x_t^2] = 2f(t)\,\mathbb{E}[x_t^2] + g(t)^2.
                $$

                定义二阶矩 \(M_2(t)=\mathbb{E}[x_t^2]\),即

                $$
                \frac{dM_2(t)}{dt} = 2f(t)M_2(t) + g(t)^2.
                $$

        2. 引入均值与方差的关系

            - 均值：\( m(t)=\alpha_t x_0 \)
            - 方差定义：
                $$
                \sigma_t^2 = \operatorname{Var}(x_t) = M_2(t)-m(t)^2,
                $$

                即

                $$
                \boxed{M_2(t)=\sigma_t^2+m(t)^2.}
                $$

            - 对 \(m(t)^2\) 求导：
            $$
            \frac{d}{dt}\big(m(t)^2\big)=2m(t)\,\frac{dm(t)}{dt} = 2f(t)m(t)^2.
            $$

        1. 推导方差演化方程

            - 将 \(M_2(t)=\sigma_t^2+m(t)^2\) 对时间求导得：

                $$
                \frac{dM_2(t)}{dt} = \frac{d}{dt}\sigma_t^2 + 2f(t)m(t)^2.
                $$

            - 同时由二阶矩演化有：

                $$
                \frac{dM_2(t)}{dt} = 2f(t)\big(\sigma_t^2+m(t)^2\big)+g(t)^2.
                $$

            - 对比两式，消去 \(2f(t)m(t)^2\) 得：

                $$
                \frac{d}{dt}\sigma_t^2 = 2f(t)\sigma_t^2+g(t)^2.
                $$

            - 整理后可解出 \(g(t)\) 的关系：

                $$
                g(t)^2 = \frac{d\sigma_t^2}{dt} - 2f(t)\sigma_t^2.
                $$

            - 结合 \(f(t)=\frac{d\log\alpha_t}{dt}\) 得最终表达：

                $$
                \boxed{g(t) = \sqrt{\frac{d\sigma_t^2}{dt} - 2\,\frac{d\log\alpha_t}{dt}\,\sigma_t^2}.}
                $$

    4. 总结关系

        - **均值与前向进度的关系：**
                $$
                f(t) = \frac{d\log\alpha_t}{dt}.
                $$
        - **二阶矩与方差的关系：**
                $$
                M_2(t) = \sigma_t^2 + m(t)^2.
                $$
        - **方差演化：**
                $$
                \frac{d\sigma_t^2}{dt} = 2f(t)\sigma_t^2 + g(t)^2.
                $$
        - **噪音调度与方差变化的关系：**
                $$
                g(t) = \sqrt{\frac{d\sigma_t^2}{dt} - 2\,\frac{d\log\alpha_t}{dt}\,\sigma_t^2}.
                $$
