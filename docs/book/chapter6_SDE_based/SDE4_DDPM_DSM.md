#  Stochastic Differential Equation for DDPM and DSM

In this section, we show how connect SDE with the denoising defussion probabilistic model and the denoising score matching


In the context of the excerpt, the term “autocorrelation function” refers to the covariance

$$
 \mathbb{E}[\xi(t)\,\xi(t')]\quad
$$

of the noise process $\xi(\cdot)$ evaluated at two different times $t$ and $t'$. For an idealized
**white‐noise**  process (as in standard Brownian motion), this autocorrelation is taken to be a **Dirac delta function** :

$$
 \mathbb{E}[\xi(t)\,\xi(t')] \;=\;\delta(t - t').
$$

Intuitively, this means that $\xi(t)$ is “uncorrelated” with itself at any times $t \neq t'$, and all of its correlation is concentrated at the single point $t = t'$. In more precise terms, the Dirac delta is not a function in the usual sense but a distribution, capturing the idea that white noise has no memory or persistence in time—any two distinct instants are independent, but at the same instant the variance (or “intensity”) is infinite in such a way that it integrates to a finite value over an infinitesimal interval.




## 1. Forward Diffusion
Suppose we have the *forward* SDE(1)$$
 \tag{1}
dx \;=\; f(x,t)\,dt \;+\; g(t)\,dW_t,
$$
where $$W_t$$ is a standard Wiener process, and $$g(t)$$ is (for simplicity) taken to be spatially constant but possibly time‐dependent. Let $$p_t(x)$$ be the probability density of $$x_t$$ at time $$t$$.
### 1.1. Fokker–Planck (Forward Kolmogorov) Equation
The density $$p_t(x)$$ satisfies the Fokker–Planck (or forward‐Kolmogorov) PDE$$
 \frac{\partial p_t(x)}{\partial t}
\;=\;
-\nabla\!\cdot\!\bigl[f(x,t)\,p_t(x)\bigr]
\;+\;
\frac{1}{2}\,g(t)^2\,\Delta p_t(x),
$$
where $$\Delta$$ is the Laplacian in $$x$$.

---


## 2. Time‐Reversal Strategy
We want an SDE whose sample paths “run backward” in time with the *same* distributions as $$(x_t)$$ running forward. Concretely, define$$
 \hat{X}_s \;=\; x_{T - s}
\quad
\text{for } 0 \le s \le T.
$$
That is, $$\hat{X}_0 = x_T$$, $$\hat{X}_T = x_0$$. Our goal is to find a stochastic differential equation of the form(2)$$
 \tag{2}
d\hat{X}_s
\;=\; \hat{f}\bigl(\hat{X}_s, s\bigr)\,ds
\;+\;
g(T-s)\,d\overline{W}_s
$$
(where $$\overline{W}_s$$ is another Wiener process) such that $$\hat{X}_s$$ has the *same* distribution as $$x_{T-s}$$.In other words, if $$q_s(x)$$ denotes the density of $$\hat{X}_s$$, we want $$q_s(x) = p_{T-s}(x)$$. We must determine the drift $$\hat{f}(x,s)$$.

---


## 3. Matching Probability Densities via Fokker–Planck

### 3.1. PDE for the reverse process
If $$\hat{X}_s$$ satisfies$$
 d\hat{X}_s
= \hat{f}(\hat{X}_s, s)\,ds
\;+\;
g(T-s)\,d\overline{W}_s,
$$
then its density $$q_s(x)$$ obeys$$
 \frac{\partial q_s(x)}{\partial s}
\;=\;
-\nabla \cdot \bigl[\hat{f}(x, s)\;q_s(x)\bigr]
\;+\;
\frac{1}{2}\,g(T-s)^2\;\Delta\,q_s(x).
$$
We require $$q_s(x) = p_{T-s}(x)$$. Substitute $$q_s(x) = p_{T-s}(x)$$ into the PDE. Note that$$
 \frac{\partial}{\partial s}\,p_{T-s}(x)
\;=\;
-\,\frac{\partial}{\partial t}\,p_{t}(x)\Bigr|_{\,t=T-s}.
$$

Hence
$$
 \frac{\partial q_s}{\partial s}(x)
\;=\;
-\,\frac{\partial}{\partial t}\,p_{t}(x)\Bigr|_{t=T-s}.
$$
But from the *forward* Fokker–Planck equation, we know$$
 \frac{\partial}{\partial t}\,p_{t}(x)
\;=\;
-\nabla\!\cdot\!\bigl[f(x,t)\,p_t(x)\bigr]
+\tfrac{1}{2}\,g(t)^2\,\Delta\,p_t(x).
$$
Putting $$t = T-s$$ and changing the sign, we get$$
 \frac{\partial q_s}{\partial s}(x)
\;=\;
+\nabla\!\cdot\!\bigl[f(x,T-s)\,p_{T-s}(x)\bigr]
\;-\;
\tfrac{1}{2}\,g(T-s)^2\,\Delta\,p_{T-s}(x).
$$
3.2. Identifying the reversed drift $$\hat{f}$$Meanwhile, from the PDE for $$q_s$$ we also have$$
 \frac{\partial q_s}{\partial s}(x)
\;=\;
-\,\nabla\!\cdot\!\bigl[\hat{f}(x,s)\,q_s(x)\bigr]
\;+\;
\tfrac12\,g(T-s)^2\,\Delta\,q_s(x).
$$
Because $$q_s(x) = p_{T-s}(x)$$, we can equate the two expressions. Matching terms gives$$
 \nabla\!\cdot\!\bigl[\hat{f}(x,s)\,p_{T-s}(x)\bigr]
\;=\;
-\,\nabla\!\cdot\!\bigl[f(x,T-s)\,p_{T-s}(x)\bigr]
\;+\;
g(T-s)^2 \,\nabla \!\cdot\!\bigl[\tfrac12\,p_{T-s}(x)\bigr],
$$
where we used that $$\Delta\,p_{T-s}(x) = \nabla \cdot (\nabla\,p_{T-s}(x))$$. Rearrange:$$
 \nabla\!\cdot\!\Bigl[\hat{f}(x,s)\,p_{T-s}(x)\Bigr]
\;=\;
-\nabla\!\cdot\!\Bigl[f(x,T-s)\,p_{T-s}(x)\Bigr]
\;+\;
\tfrac12\,g(T-s)^2\,\nabla \cdot\!\Bigl[\nabla\,p_{T-s}(x)\Bigr].
$$
We can factor out $$p_{T-s}(x)$$ to write$$
 \hat{f}(x,s)\,p_{T-s}(x)
\;=\;
f(x,T-s)\,p_{T-s}(x)
\;-\;
\tfrac12\,g(T-s)^2\,\nabla\,p_{T-s}(x)
\;+\;
\text{(gradient‐free constant in } x\text{)},
$$
but the divergence‐free “constant” in $$x$$ must be zero if we want the correct boundary conditions at infinity and a well‐defined velocity field. Dividing both sides by $$p_{T-s}(x)$$ then yields$$
 \hat{f}(x,s)
\;=\;
f(x,T-s)
\;-\;
\tfrac12\,g(T-s)^2\,\nabla \log p_{T-s}(x).
$$

This is precisely the extra “score‐function” correction term discovered by Nelson and Anderson.


---


## 4. Final Reverse SDE
Hence the *reversed* SDE can be written as$$
 \boxed{
d\hat{X}_s
\;=\;
\Bigl[
f(\hat{X}_s,\,T-s)
\;-\;
\tfrac12\,g(T-s)^2\,\nabla \log p_{T-s}(\hat{X}_s)
\Bigr]\;ds
\;+\;
g(T-s)\,d\overline{W}_s.
}
$$
Often one writes the time variable in the same forward direction and just says “the reverse‐time SDE from $$t$$ down to 0 is$$
 dx \;=\;
\bigl[f(x,t)\;-\;g(t)^2\,\nabla_x \log p_{t}(x)\bigr]\;dt \;+\; g(t)\,d\overline{W}_t,
$$

”
where $$\overline{W}_t$$ is a standard Wiener process *when viewed backward* in the original clock. In either notation, the key extra piece is $$\,-\,g^{2}\nabla\log p_{t}(\cdot)$$, which ensures the *distributions* truly run in reverse.

---


### Key Intuition

- The drift in the reverse process must “compensate” for how diffusion was spreading mass forward in time.

- This compensation appears as $$\,-\,\tfrac12\,g^2 \nabla \log p_t(x)$$.

- Equivalently, one can say that in the backward direction, the random walk “knows” how to concentrate probability back into regions from which it had been dispersing forward.
This completes the derivation (sometimes known as *Anderson’s theorem*).


hought about Fokker-Planck equation for 4 秒**Understanding the Fokker–Planck Equation: A Gentle Introduction** In many physical, biological, and financial systems, we are often interested not only in the behavior of individual particles or agents but also in how *probability distributions* evolve over time. If you have ever encountered a situation where randomness (noise) and drift (directed motion) both play roles in the dynamics, you have likely brushed up against the *Fokker–Planck equation (FP equation)*.
In this blog, we will explore:

1. **What the Fokker–Planck equation is.**

2. **When and why the probability density follows it.**

3. **An intuitive explanation to help you understand it deeply.**


---


## 1. What is the Fokker–Planck Equation?
Put simply, the Fokker–Planck equation describes how the *probability density function* (PDF) of a stochastic (random) process evolves over time. More formally, if we let $p(x,t)$ be the probability density of finding the system in state $x$ at time $t$, the Fokker–Planck equation reads:$$
 \frac{\partial p(x,t)}{\partial t} = -\frac{\partial}{\partial x}\big[\,\mu(x)\,p(x,t)\big]
+ \frac{1}{2}\,\frac{\partial^2}{\partial x^2}\big[\,D(x)\,p(x,t)\big],
$$

where

- $\mu(x)$ represents the *drift* (systematic motion) of the process in state $x$,

- $D(x)$ represents the *diffusion* (random fluctuations) at state $x$.

For simplicity, we often see the equation in the case of constant diffusion $D$ and constant drift $v$:
$$
 \frac{\partial p(x,t)}{\partial t} = -v\,\frac{\partial p(x,t)}{\partial x} + \frac{D}{2}\,\frac{\partial^2 p(x,t)}{\partial x^2}.
$$

This partial differential equation tells us how the probability density changes because of both deterministic drift ($v$) and random diffusion ($D$).


---


## 2. When Does the Probability Follow the Fokker–Planck Equation?

The Fokker–Planck equation is typically valid under these conditions:

1. **Continuous Markov processes in one dimension (or higher).**
  - The process must be *Markovian*: the future depends only on the present state, not on the history.

2. **Stochastic differential equations (SDEs) of the form** $$
 dX_t = \mu(X_t)\,dt + \sqrt{D(X_t)}\,dW_t,
$$

  - where $W_t$ is a standard Brownian motion (Wiener process).

3. **Small, Gaussian-like noise.**
  - The derivation of the Fokker–Planck relies on the assumption that increments of noise are small over short intervals (leading to the second-order derivative term in the FP equation).

4. **No long-range jumps (no Lévy flights).**
  - If jumps or large discontinuous moves exist, then we would use *generalized* kinetic equations (like fractional Fokker–Planck or Master equations) rather than the classical FP equation.
In essence, if your system follows an Itô-type (or Stratonovich-type) SDE with drift $\mu$ and diffusion coefficient $D$, then the *probability density* of where that system might be at time $t$ will satisfy the Fokker–Planck equation.

---


## 3. An Intuitive Explanation

### (a) Drift Term: Moving the Probability Along
Imagine you have a pile of sand on a table. If you tilt the table slightly, the sand starts sliding in a certain direction (the *drift*). In probability terms, this systematic motion is captured by $-\frac{\partial}{\partial x}[\mu(x) p(x,t)]$. It tells us that if there is some drift velocity $\mu(x)$, it will *transport* the probability density along $x$.
- **In words** : “If the drift at position $x$ is large, probability quickly flows away from $x$ to neighboring states.”

### (b) Diffusion Term: Spreading the Probability Out
Returning to the sand analogy: in addition to sliding, random vibrations (like small shakes) cause the sand to spread out. This spreading effect is the *diffusion* and is captured by the second term $\frac{1}{2}\frac{\partial^2}{\partial x^2}[D(x) p(x,t)]$.
- **In words** : “If there is diffusion at position $x$, it causes probability to spread out from regions of higher density to lower density nearby.”

### (c) Combining Drift and Diffusion

The Fokker–Planck equation brings these two effects together. First, it shifts the distribution in the direction of the drift, and second, it broadens (or contracts) the distribution due to diffusion. Over time, the competition or combination of these two dynamics shapes the overall probability distribution in your system.


---


## 4. Why is the Fokker–Planck Equation Important?

- **Physical systems** : Think of particles in a fluid (like pollen particles in water—Brownian motion). The drift might come from fluid flow or external forces, while the diffusion comes from thermal fluctuations.

- **Financial mathematics** : Stock prices are often modeled as stochastic processes with drift and diffusion terms. The FP equation describes the time evolution of the probability density of stock prices.

- **Neuroscience** : Membrane potentials of neurons under random synaptic inputs sometimes follow a Fokker–Planck-like dynamics to describe the distribution of firing times.

- **Population genetics** : Allele frequency distributions in a population can be governed by drift (selection) and diffusion (random genetic drift).
Whenever you have a random variable subject to a systematic force (drift) plus random noise (diffusion), the Fokker–Planck is *the* PDE that details how that random variable’s distribution evolves.
