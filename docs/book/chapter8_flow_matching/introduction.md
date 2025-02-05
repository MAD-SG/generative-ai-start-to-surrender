# Introduction of flow matching

The goal of the generative model is sampling from a real data distribution $p_{data}(x)$. However, it's not possible to unkow the density function of $p_{data}(x)$.

- If we know the density function $p_{data}(x)$ or the score function of $p_{data}(x)$, we can use the sampling algorithm to generate samples from $p_{data}(x)$ like the Langevin Dynamics as described in the engery based function.

- If we say, approximate the original density function is not easy, we can work around this problem by building a path from the original data distribution to a easy distribution. Then we can sample from the easy distribution and follow the path to sample the original data distribution. Like the diffusion process, either DDPM or SDE based model, we build a path that the original data distribution is gradually transformed to a standard normal distribution. The path is called the probability density path.

Questions
Problem
1. no scalable CNF training algorithms are known

Terms
- probability density path

- Continuous Normalizing Flow

Problem:
- experiment about the log likelihood estimation