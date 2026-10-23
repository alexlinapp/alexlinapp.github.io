---
# Uses the 2018-12-22-distill.md layout!
layout: distill
title: Not all Gradient Noise is Equal
description: The structure of gradient noise relative to loss structure may improve generalization
tags: Optimization
# giscus_comments: true
date: 2026-10-22
featured: true
mermaid:
  enabled: true
  zoomable: true
code_diff: true
map: true
chart:
  chartjs: true
  echarts: true
  vega_lite: true
tikzjax: true
typograms: true

authors:
  - name: Alex Lin
    # url: "https://en.wikipedia.org/wiki/Albert_Einstein"
    # affiliations:
    #   name: IAS, Princeton

# Add Bibliography Here
# use <d-cite key="mandt2017stochastic"></d-cite>
bibliography: 2026-10-16-Anisotropic.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Intro
  - name: Preliminaries
  - name: Minima Escaping Behavior
  - name: Interpreting the Noise
  - name: Experiments
  - name: Conclusion
  - name: Future Directions
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Appendix
  - name: Citations

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
# _styles: >
#   .fake-img {
#     background: #bbb;
#     border: 1px solid rgba(0, 0, 0, 0.1);
#     box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
#     margin-bottom: 12px;
#   }
#   .fake-img p {
#     font-family: monospace;
#     color: white;
#     text-align: left;
#     margin: 12px 0;
#     text-align: center;
#     font-size: 16px;
#   }
_styles: |
  d-byline .authors-affiliations {
    grid-template-columns: 1fr !important;
  }

  d-byline .authors-affiliations > h3:nth-child(2),
  d-byline .authors-affiliations .affiliation {
    display: none !important;
  }

  d-byline .byline {
    grid-template-columns: 1fr 1fr !important;
  }
---


## Intro

SGD never sees the exact gradient of the training loss. At every update, it computes a gradient from a randomly selected minibatch, and each minibatch gives a slightly different answer. The resulting optimization path is noisy rather than deterministic.

One common explanation is that this noise acts as an implicit regularizer<d-cite key="pmlr-v119-wu20c"></d-cite>. By continually perturbing the parameters, minibatch sampling may prevent SGD from remaining in sharp regions of the loss landscape and bias it toward flatter solutions that sometimes generalize better.

This story is not necessarily wrong, but it treats noise as though it were controlled by a single volume knob. In reality, noise has directions. Two noise distributions can have the same expected magnitude while pushing the parameters along completely different parts of the loss landscape. A perturbation along a high-curvature direction may change the loss substantially, while an equally large perturbation along a flat direction may do almost nothing.

To make this distinction precise, we need a model that describes both the geometry of the loss and the geometry of the noise. A useful starting point is the local quadratic model. The Hessian $H$ describes the curvature of the loss, while the covariance matrix $\Sigma$ describes how the noise variance is distributed across parameter directions. This model is simple enough to analyze exactly, yet rich enough to expose the interaction between curvature, noise magnitude, anisotropy, and alignment.

Following Zhu et al.<d-cite key="zhu2018anisotropic"></d-cite>, we will see that this interaction is summarized locally by

$$
\operatorname{tr}(H\Sigma)
$$

We will eventually see that there is a connection between this analysis and the empirical observations of power-law eigenspectra in neural-network Hessians and stochastic-gradient covariance matrices<d-cite key="pmlr-v267-tang25d,xie2023overlooked"></d-cite> and may have an impact on the generalization of real model training.



## Preliminaries

We first setup our problem and give a brief review of some minibatch gradient properties.

Fix our dataset $\mathcal{D}$. Let $\theta$ be our model parameters, $\mathcal{L}(\theta, \mathcal{D})$ be our empirical loss. We cond

Standard GD (gradient descent) typically involves optimizing:

$$
\min_{\theta} \mathcal{L}(\theta, \mathcal{D}) = \min_{\theta} \frac{1}{N} \sum_{i=1}^N \ell(x_i; \theta)
$$

Where $x_i \in \mathcal{D}$ and for brevity $\ell(x_i, \theta)$ is the loss associated with the observation $x_i$, incorporating the model's computationi ($f_\theta(x_i)$ is left implicit). 

Standard gradient descent does:

$$
\begin{equation}
\theta_{t+1} = \theta_t - \eta_t \nabla_\theta \mathcal{L}(\theta_t) 
\end{equation}
$$

However due to practical reasons, it is infeasible to calculate the gradient over the whole dataset of $N$ examples, hence stochastic gradient descent (SGD) is often used. Defining the unbiased estimator of the true gradient of the empirical loss $\nabla_\theta \mathcal{L}(\theta)$:

$$
g(\theta_t) := \frac{1}{m} \sum_{x \in B_t} \nabla_\theta \ell(x; \theta_t)
$$

Here $B_t$ is a randomly selected minibatch of size $m$ selected uniformly from $\mathcal{D}$. The update rule for stochastic gradient descent (SGD) is:

$$
\begin{equation}
\theta_{t+1} = \theta_t - \eta \hat{g}(\theta_t) \label{eq:sgd-update}
\end{equation}
$$

Assuming that $m$ is large enough that the central limit theorem holds:

$$
\hat{g}(\theta_t) \sim \mathcal{N}(\nabla L(\theta_t), \Sigma^{\text{sgd}}(\theta_t))
$$

where

$$
\Sigma^{\text{sgd}}(\theta_t) \approx \frac{1}{m} \left[\frac{1}{N} \sum_{i=1}^N \nabla \ell(x_i; \theta_t) \nabla \ell(x_i; \theta_t)^\top - \nabla \mathcal{L}(\theta_t) \nabla \mathcal{L}(\theta_t)^\top \right]
$$

Hence defining:

$$
\varepsilon_t := \hat{g}(t) - \nabla \mathcal{L}(\theta_t) \sim \mathcal{N}(0, \Sigma^{\text{sgd}}(\theta_t))
$$

we may rewrite $\eqref{eq:sgd-update}$ as:

$$
\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t) + \eta \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \Sigma^{\text{sgd}}(\theta_t))
$$

At this point, one can consider gradient Langevin dynamics, (GLD), where the update is of the form:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t) + \eta \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \sigma_t^2 I)
$$

In fact GLD can be considered a discretization of the continuous time SDE<d-cite key="li2017stochastic"></d-cite> taking the time between the discrete time steps to be $h := \eta$. 

$$
\begin{equation}
d \theta_t = -\nabla_\theta \mathcal{L}(\theta_t) dt + \sqrt{\eta\Sigma_t} dW_t \label{eq:SDE-GLD}
\end{equation}
$$

Refer to the appendix for more details.


## Minima Escaping Behavior
<!-- 
Following Zhu et al. <d-cite key="zhu2018anisotropic"></d-cite> -->

Consider the second order approximation of the loss:

$$
\begin{equation}
\mathcal{L}(\theta) \approx \mathcal{L}(\theta_0) + (\theta - \theta_0) \nabla_\theta L(\theta_0) + \frac{1}{2} (\theta- \theta_0)^\top \nabla^2_\theta L(\theta_0) (\theta- \theta_0)
\end{equation}
$$

Here we assume that $\theta_0$ is fixed and that $\theta_t$ is a random variable, as it depends on the noisy gradients. 

Near a stationary point $\theta_0$, i.e.:

$$
\nabla_\theta L(\theta_0) = 0
$$

Near this stationary point, under the second-order approximation the loss is:

$$
\begin{equation}
\mathcal{L}(\theta) \approx \mathcal{L}(\theta_0)  + \frac{1}{2} (\theta- \theta_0)^\top \nabla^2_\theta L(\theta_0) (\theta- \theta_0) \label{eq:stationary-second-order-approx}
\end{equation}
$$

That is the loss should be well approximated by a quadratic. In order to study generalization, we instead turn to a more tractable indicator and study how noise pushes the optimizer **away** from the minimum. Following the formulation of Zhu et al.<d-cite key="zhu2018anisotropic"></d-cite>, define the **escape efficiency**:

$$
\begin{equation}
\mathbb{E}_{\theta_t}\left[ \mathcal{L}(\theta_t) - \mathcal{L}(\theta_0)\right] \label{eq:escape-efficiency}
\end{equation}
$$

Again since $\theta_t$ is a random variable, we can consider the random variable:

$$
\begin{equation}
X := \mathcal{L}(\theta_t) - \mathcal{L}(\theta_0)
\end{equation}
$$

If $\theta_0$ is a local minimum, then for sufficiently small $t$, by Markov's inequality, for any $\delta > 0$:

$$
\begin{equation}
\mathbb{P}(\mathcal{L}(\theta_t) - \mathcal{L}(\theta_0) \geq \delta) \leq \frac{\mathbb{E}[\mathcal{L}(\theta_t) - \mathcal{L}(\theta_0)]}{\delta}
\end{equation}
$$

The idea is that sharp minima attained during training may often generalize wrose than "flatter" solutioins. Here sharper means that smaller weight pertubataions in some directions may increase training loss substantially while a flatter solution, as its name implies is flat so small weight movement around this flatter minima, i.e. robustness to weight pertubations **may** (not always!) lead to accuracy on unseen examples.

Continuing, under sufficiently smooth assumptions, after applying Ito's Lemma, $\eqref{eq:escape-efficiency}$ becomes:

$$
\mathbb{E}_{\theta_t}\left[ \mathcal{L}(\theta_t) - \mathcal{L}(\theta_0)\right] = - \int_0^t \mathbb{E}[\nabla \mathcal{L}^\top \nabla L] + \int_0^t \frac{1}{2} \mathbb{E} \operatorname{tr} (H_t \Sigma_t) dt
$$

where $H_t := \nabla_\theta^2 \mathcal{L}(\theta_t)$. 

Under the assumption that $\theta_0$ is a local minimum (hence a stationary point), applying $\eqref{eq:stationary-second-order-approx}$ gives a tractable equation:

$$
\begin{equation}
\mathbb{E}_{\theta_t}[\mathcal{L}(\theta_t) - \mathcal{L}(\theta_0)] = \frac{1}{4} \operatorname{tr}\left((I - e^{-2Ht})\Sigma \right) \label{eq:tractable-escape-efficiency}
\end{equation}
$$

For small $t$, $\eqref{eq:tractable-escape-efficiency}$ gives:

$$
\operatorname{tr}\left((I - e^{-2Ht})\Sigma \right) \approx \frac{t}{2} \operatorname{tr}(H \Sigma)
$$

However as $t \to \infty$ we instead have:

$$
\operatorname{tr}\left((I - e^{-2Ht})\Sigma \right) \approx \frac{1}{4} \operatorname{tr}(\Sigma)
$$

Hence there are two different regimes for how we want to measure escape efficiency. Note here $t$ is for the continuuous time representation of our discrete problem! That is for constant learning rate $\eta$, the next iteration of GLD is at the timestep:

$$
t_{i+1} = t_i + \eta
$$

Keep this in mind when intepreting the graphs and the experiments as adjusting the learning rate can change the regime we are operating in and which approximation holds.


## Interpreting the Noise

Before continuing, the whole goal is to understand how the **structure** of noise introduced from SGD impacts generalization. Hence the effect of noise magnitude must be eliminated to create a fairer comparison. One reasonable metric for the noise magnitude is the expected square norm of the noise vector<d-cite key="li2017stochastic"></d-cite>. Consider $QDQ^\top = \Sigma \succeq 0$. Let $z \sim \mathcal{N}(0, I)$. Then define $\varepsilon := QD^{1/2} z \sim \mathcal{N}(0, \Sigma)$. Therefore:

$$
\begin{gather*}
\mathbb{E}_{\varepsilon}[\|\varepsilon\|_2^2]= \mathbb{E}_{\varepsilon} [\varepsilon^\top \varepsilon] = \mathbb{E}_{z}[z^\top D^{1/2} Q^\top Q D^{1/2} z] \\
= \mathbb{E}_{z}[\operatorname{tr}(z^\top D z)] = \mathbb{E}_{z}[\operatorname{tr}(D z^\top z)] = \operatorname{tr}(D) = \operatorname{tr}(\Sigma)
\end{gather*}
$$

Therefore given time $t$, $\operatorname{tr}(\Sigma_t)$ should be constant. That is we impose the constraint:

$$
\begin{equation}
\operatorname{tr}(\Sigma) = c, \qquad c > 0 \label{eq:sigma-constraint}
\end{equation}
$$

However, the structure of $\Sigma_t$ and the induced curvature from $\Sigma_t$ may change. For example, consider $\Sigma_t \in \mathbb{R}^{2 \times 2}$ and $\operatorname{tr}(\Sigma_t) = 1$. Then these two matrices satisfy the magnitude constraint, however lead to very different dynamics (as will be seen later):

$$
\Sigma_1 = \begin{bmatrix}
0.5 & 0 \\
0 & 0.5
\end{bmatrix}, \qquad 
\Sigma_2 = \begin{bmatrix} 
0.99999 & 0 \\

0 & 10^{-5}
\end{bmatrix}, \qquad 
\Sigma_3 = \begin{bmatrix} 
10^{-5} & 0 \\
0 & 0.99999
\end{bmatrix}
$$

There is an intuitive interpretation for these "noise" matrices. For a fixed Hessian $H \in \mathbb{R}^{2 \times 2}$, say:

$$
H = \begin{bmatrix}
100 & 0 \\
0 & 1
\end{bmatrix}
$$

There is an orthonormal basis of eigenvectors for $H$, particularly, $e_1, e_2$, with eigenvalues $\lambda_1 = 100, \lambda_2 = 1$. Computing $\operatorname{tr}(H \Sigma)$ for each fo the noise matrices:

$$
\operatorname{tr}(H \Sigma_1) = 50.5, \qquad \operatorname{tr}(H \Sigma_2) = 99.99901, \qquad \operatorname{tr}(H \Sigma_3) = 1.00099
$$

and recall that $\operatorname{tr}(H \Sigma)$ is directly tied to the escape effieciency, it is clear that the structure of noise could have an impact for escaping stationary points. 

Generalizing the example further, suppose that $H \in \mathbb{R}^{D \times D}$ and $H \succeq 0$. So the eigen decomposition of $H = Q D Q^\top$ and the columns of $Q$ is an orthornomal basis of eigenvectors, say $q_1, \ldots, q_D$ with eigenvalues $\lambda_1, \ldots, \lambda_D$. Let $\varepsilon \in \mathbb{R}^D$ be a noise vector. Since $q_1, \ldots, q_D$ is an orthornomal basis of eigenvectors for $\mathbb{R}^D$, $\exists c_1, \ldots, c_D \in \mathbb{R}^D$ such that:

$$
\varepsilon = \sum_{i=1}^D c_i q_i
$$

In fact since $q_1, \ldots, q_D$ are orthonormal we have $\forall j$:

$$
q_j^\top \varepsilon = c_i
$$

Assuming $\varepsilon \sim \mathcal{N}(0, \Sigma)$, we can consider the variances of each component of $\varepsilon$ in the direction of $q_j$:

$$
\begin{align}
\operatorname{Var}(c_j) = \mathbb{E}_{\varepsilon}[c_j^2] &= \mathbb{E}_{\varepsilon}[(q_j^\top \varepsilon) (q_j^\top \varepsilon)^\top] \notag \\
&= q_j^\top \mathbb{E}_{\varepsilon}[\varepsilon \varepsilon^\top ] q_j \notag \\
&= q_j^\top \Sigma q_j \label{eq:variance-in-direction}
\end{align}
$$


That is $q_j^\top \Sigma q_j$ can be interpreted as the variance of $\varepsilon$ in the direction of $q_j$.

This allows us to decompose $\operatorname{tr}(H\Sigma)$. To see why:

$$
\begin{align*}
\operatorname{tr}(H\Sigma) &= \operatorname{tr}\left(\sum_{i=1}^D \lambda_i q_i q_i^\top \Sigma \right) \\
&= \sum_{i=1}^D \lambda_i \operatorname{tr}\left(q_i q_i^\top \Sigma \right) \\
&= \sum_{i=1}^D \lambda_i q_i^\top \Sigma q_i \\
&= \sum_{i=1}^D \lambda_i \operatorname{Var}(c_i)
\end{align*}
$$

This precisely measures the total variance of $\varepsilon$ weighted by the eigenvalues of $H$. Intuitively, this means the more larger curvature directions with the noise having high variance in that direction, the more we can expect to escape local minima.

Going one step further, since $\Sigma \succeq 0$, we have an eigendecomposition of $\Sigma = U D_\Sigma U^\top$ where say the eigenvalues are $s_1, \ldots, s_D$ with eigenvectors $u_1, \ldots, u_D$, the columns of $U$. Then rewriting $\eqref{eq:variance-in-direction}$:

$$
\begin{align}
  \operatorname{tr}(\Sigma H) &= \operatorname{tr}(\sum_{i, j} \lambda_i s_j q_i q_i^\top u_i u_i^\top) \notag \\
  &= \sum_{i, j} \lambda_i s_j \operatorname{tr}(q_i^\top u_i (q_i^\top u_i)) & \text{ cylic property of trace} \notag \\
  &= \sum_{i, j} \lambda_i s_j (q_i^\top u_j)^2 \label{eq:escape-efficiency-alignment}
\end{align}
$$

As it can be oberserved, it does not suffice for the eigenvalues alone of $\Sigma$ and $H$ to be similar. We also need that the eigenvectors of $\Sigma$ and $H$ to be pointing in the same direction, otherwise the term $q_i^\top u_j = 0$ when they are perpendicular. This will become important in the experimental section when testing on real networks as even if the eigenvalues of the Hessian and eigenvalues of the covariance matrix of the noise follow a power law<d-cite key="pmlr-v267-tang25d,xie2023overlooked"></d-cite>, misalignment in their directions may still lead to a low value of $\operatorname{tr}(H \Sigma)$.





### Extreme Cases

Okay, so now that we have a better picture of what part of SGD makes it so that it can escape local minima, we wouuld like to have a baseline comparison. Consider the isotropic minima. That is $H= \lambda I$. Then $\operatorname{tr}(H \Sigma) = \lambda \operatorname{tr}(\Sigma)$ which is invariant under our constraint. 

How about when $\operatorname{tr}(H \Sigma)$ achieves the maximal value? For a fixed $H$ what should $\Sigma$ with $\operatorname{tr}(\Sigma) = c$ for some $c > 0$ look like in this case? That is we want to solve:

$$
\underset{\Sigma \succeq 0, \operatorname{tr}(\Sigma)= c}{\max}\operatorname{tr}(H \Sigma)
$$

In fact taking the eigendecomposition of $H = QDQ^\top$ and where $D$ contains the eigenvalues $\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_D$, we claim that $\Sigma^* := c q_1 q_1^\top$ indeed attains the maximum. 

To see this, consider any $\Sigma \succeq 0$ with $\operatorname{tr}(\Sigma)  =c$. Then:

$$
\begin{align*}
\operatorname{tr}(H \Sigma) &= \sum_{i=1}^D \underbrace{\geq 0}_{\lambda_i q_i^\top \Sigma q_i} \\
&\leq \sum_{i=1}^D \lambda_1 q_i^\top \Sigma q_i \\
&= \lambda_1 \sum_{i=1}^D \operatorname{tr}(q_i^\top \Sigma q_i) \\
&= \lambda_1 \operatorname{tr}(\sum_{i=1}^D q_i q_i^\top \Sigma) \\
&= \lambda_1 \operatorname{tr}(\Sigma) \\
&= \lambda_1 c
\end{align*}
$$

Note that $\operatorname{tr}(c q_1 q_1^\top) = c \operatorname{tr}(q_1^\top q) = c$. Also:

$$
\begin{align*}
\operatorname{tr}(H \Sigma^*) &= \sum_{i=1}^D \lambda_i c q_i^\top q_1 q_1^\top q_i \\
&= \lambda_1 c
\end{align*}
$$

In the context of $\eqref{eq:escape-efficiency-alignment}$, $H = c q_1 q_1^\top$ is perfectly aligned with $H$ in the direction of $q_1$, with all of its variance being pushed in the direction $q_1$, which allows for faster escape from a stationary point.



## Experiments

We can transform the $\eqref{eq:escape-efficiency}$ into a quadratic problem centered at $\theta_0$. See the [Appendix](#transformation-of-variables) for more details. Originally we have:

$$
\mathcal{E}_k := \mathbb{E}_{\theta_k \sim P_k} [L(\theta_k) - L(\theta_0)]
$$

Consider the displacement from $\theta_0$ as:

$$
\begin{equation}
\delta_k := \theta_k - \theta_0 \label{eq:update-rule-delta}
\end{equation}
$$

Define:

$$
\begin{equation}
F(d) := \mathcal{L}(\theta_0 + d) - \mathcal{L}(\theta_0) = \frac{1}{2} d^\top H d
\end{equation}
$$

Then assuming the second order approximation and stationary point, we can consider the equivalent problem:

$$
F(\delta) = \frac{1}{2} (\theta- \theta_0)^\top \nabla_\theta^2 L(\theta_0) (\theta - \theta_0) = \frac{1}{2} \delta^\top \nabla_theta^2 \mathcal{L}(\theta_0) \delta 
$$

Defining $H := \nabla_\theta^2 \mathcal{L}(\theta_0)$, we would like to see observe the effects of different structure of noise and Hessians on the training dynamics of GLD near this stationary point $\theta_0$. And ideally, give an approximation of how the relationship between the noise in training and the Hessian of the true empirical loss function affect average escape from minima. 


### Constant Learning Rate, Hessian and Noise Structure

Consider the most basic case. A constant learning rate $\eta$, constant hessian $H$ and constant $\Sigma^{\text{sgd}}$ where $\varepsilon \sim \mathcal{N}(0, \Sigma^{\text{sgd}})$. It has been observed that Hessians of neural networks, particularly CNNs and LLMs follow some sort of power law<d-cite key="pmlr-v267-tang25d,xie2023overlooked"></d-cite>. This motivates us to test spectral distributions for $H$, $\Sigma$ that follows a power law. Additional spectral distributions were added as well for comparison.

Additionally, as GLD is standard GD with added noise, a stable learning rate is chosen so that the observed loss growth is caused by the injected noise and its intercation with the curvature of $H$, rahter than the instability of the numerical update. 

Following the update rule induced by $\eqref{eq:update-rule-delta}$, we use the following update rule:

$$
\delta_{k+1} = \delta_k - \eta \nabla F(\delta_k) + \eta \varepsilon_k, \quad \varepsilon_k \sim \mathcal{N}(0, \Sigma_k)
$$

Since $\delta_0 = 0$, we have for the first step:

$$
\delta_1 = \delta_0 - \eta H \delta_0 + \eta \varepsilon_0 = \eta \varepsilon_0
$$

Hence:

$$
\begin{align}
  \mathbb{E}_{\delta_1}[F(\delta_1) - F(\delta_0)] = \mathbb{E}_{\delta_1}[F(\delta_1)] &= \mathbb{E}_{\varepsilon_0}[F(\eta \varepsilon_0)] \notag \\
  &= \mathbb{E}_{\varepsilon_0}[\frac{1}{2}(\eta \varepsilon_0)^\top H (\eta \varepsilon_0)] \notag \\
  &= \frac{\eta^2}{2} \mathbb{E}_{\varepsilon_0}[\varepsilon_0^\top H \varepsilon_0] \notag \\
  &= \frac{\eta^2}{2} \mathbb{E}_{\varepsilon_0}[\operatorname{tr}(\varepsilon_0^\top H \varepsilon_0)] \notag \\
  &= \frac{\eta^2}{2} \mathbb{E}_{\varepsilon_0}[\operatorname{tr}( H \varepsilon_0 \varepsilon_0^\top)] \notag \\
  &= \frac{\eta^2}{2} \operatorname{tr}(H \mathbb{E}_{\varepsilon_0}[\varepsilon_0 \varepsilon_0^\top]) \notag  \\
  &= \frac{\eta^2}{2} \operatorname{tr}(H \Sigma)
\end{align}
$$

So in fact the very first update is impacted by the escape efficiency. And in turn we can characterize the escape efficiency in terms of the expected loss from the first update. That is:

$$
\begin{equation}
\frac{2}{\eta^2} \mathbb{E}_{\delta_1}[F(\delta_1)] = \operatorname{tr}(H \Sigma)
\end{equation}
$$

To isolate the effect of noise structure, we enforce the constraint from $\eqref{eq:sigma-constraint}$. Further for simplicity, we assume that both $H$ and $\Sigma$ are diagonal, so then the orthonormal eigenvectors is just the standard basis for both. To highlight the importance of alignment between the noise covariance and hessian, we introduce an **aligned** and **reversed** configuration. For the aligned configuration, the assigned diagonal index of the larger eigenvalues coincide with the large Hessian eigenvalues. For the reversed assignment, the same noise eigenvalues are assigned in opposite order. I.e.:

$$
\Sigma_{\text{aligned}} = \begin{bmatrix}
\lambda_1 & \cdots & 0 \\
\vdots & \ddots & \vdots \\
0 & \cdots & \lambda_D
\end{bmatrix}
\qquad
\Sigma_{\text{reversed}} =
\begin{bmatrix}
\lambda_D & \cdots & 0 \\
\vdots & \ddots & \vdots \\
0 & \cdots & \lambda_1
\end{bmatrix}
$$


<div id="fig-trace-scaled" class="l-page">
  {% include figure.liquid
    path="assets/img/anisotropic-gradient-noise/trace_scaled_first_20_steps.png"
    alt="Scaled expected quadratic loss during the first twenty noisy-gradient updates for several Hessian and noise spectra."
    caption="<strong>Figure 1.</strong> Scaled expected loss during the first twenty updates. Solid curves show aligned noise and dashed curves show reversed noise."
    class="img-fluid"
    zoomable=true
  %}
</div>


<div id="fig-spectra" class="l-page">
  {% include figure.liquid
    path="assets/img/anisotropic-gradient-noise/eigenvalue_and_noise_spectra.png"
    alt="Scaled expected quadratic loss during the first twenty noisy-gradient updates for several Hessian and noise spectra."
    caption="<strong>Figure 2.</strong> Hessian eigenvalue spectra (left) and noise-variance spectra under aligned (center) and reversed (right) assignments. Spectra are ordered by Hessian-eigenvector index. Alignment places the largest noise variances along the highest-curvature directions, while reversal places them along the lowest-curvature directions. Every noise spectrum is normalized to have the same trace."
    class="img-fluid"
    zoomable=true
  %}
</div>

As can be seen in [Figure 1](#fig-trace-scaled) depends strongly on the pairing between curvature of the Hessian and noise variance. We can see that for the reversed alignment, the expected escape efficiency is very low, highlighting the fact that the $(u_i^\top q_i)^2$ term which measures alignment, really does matter. 

For the one-small spectra distribution, it makes sense that many of the reversed assignments work well. This is because most of the eigenvalues of the Hessian are the same (and are large), hence any assignment that places little of the total curvature in the first direction will still result in a high escape efficiency. I.e. for one-small spectra distribution, the hessian is:

$$
H = \text{diag}(h_{max}, \ldots, h_{max}, h_{min})
$$

So:

$$
\operatorname{tr}(H \Sigma_{reversed}) = h_{max} \sum_{i=2}^D s_i + h_{min} s_1 = h_{max}(1-s_1) + h_{min} s_1
$$

assuming that $\operatorname{tr}(\Sigma) = 1$. 

If $h_{min} \ll h_{min}$, then:

$$
\operatorname{tr}(H \Sigma_{reversed}) = h_{max}(1-s_1) + h_{min s_1} \approx h_{max}(1-s_1)
$$


### When Alignment Matters

Previously we showed that alignment is crucial for escape efficiency. However there are actually cases when it does not matter! Consider the two dimensional cases:

$$
H_\kappa=
\begin{bmatrix}
\kappa&0\\
0&1
\end{bmatrix}, \qquad
S=
\begin{bmatrix}
\rho&0\\
0&1
\end{bmatrix}
$$

Define:

$$

\Sigma_{\mathrm{aligned}}
= \begin{bmatrix}
\rho&0\\
0&1
\end{bmatrix}, \qquad
\Sigma_{\mathrm{reversed}} = \begin{bmatrix}
1&0\\
0&\rho
\end{bmatrix}

$$

Then

$$
\operatorname{tr}(H_\kappa\Sigma_{\mathrm{aligned}})=\kappa\rho+1, \qquad \operatorname{tr}(H_\kappa\Sigma_{\mathrm{reversed}})=\kappa + \rho
$$

Hence their ratio is:

$$
\frac{\kappa\rho+1}{\kappa+\rho}.
$$

This gives a a pretty clean explanation:

- If $$\kappa = 1$$: The Hessian is isotropic and the orientation of the noise does not matter.
- If $$\rho = 1$$: The noise is isotropic and the orientation of the noise does not matter.
- If both $H$ and $\Sigma$ are strongly anisotropic, then alignment matters substantially. 

However as stated before, empirical results<d-cite key="pmlr-v267-tang25d,xie2023overlooked"></d-cite> show that in practical training $H$ and $\Sigma$ indeed are highly anisotropic. 



### CNN Experiements

In practice, we are not using a constant laerning rate, the Hessian is not constant and the noise may not necessarily be drawn repeatedly from the same type of Gaussian. However assuming the genereal structure remains the same, we see if we can apply the results obtained on quadratics to CNNs.

In the following we train on Fahsion-MNIST and first try to reproduce some key results<d-cite key="zhu2018anisotropic"></d-cite>

<div id="fig-spectra" class="l-page">
  {% include figure.liquid
    path="assets/img/anisotropic-gradient-noise/test_accuracy_paper_style.png"
    alt="Scaled expected quadratic loss during the first twenty noisy-gradient updates for several Hessian and noise spectra."
    caption="<strong>Figure 3.</strong> Test Accuracy after training Fashion-MNIST to 100% training accuracy using GD"
    class="img-fluid"
    zoomable=true
  %}
</div>

Unfortunately, we do not obtain the exact same long term improvement claimed by the paper across multiple seeds. For short term durations however, we do indeed see similarities with the paper. Below we present the results.

| Method | Endpoint test accuracy | Difference from GD | Validation-selected test accuracy | Selected updates by seed |
|---|---:|---:|---:|---:|
| Dynamic Top-1 Hessian (every 10)* | **71.42 ± 0.95%** | +3.28 % | **72.52 ± 0.69%** | 600, 400, 500 |
| Dynamic covariance GLD (every 10) | 70.07 ± 0.56% | +1.93 % | 70.71 ± 0.15% | 300, 300, 700 |
| Dynamic Hessian GLD (every 10) | 69.84 ± 1.24% | +1.70 % | 71.52 ± 1.25% | 300, 400, 500 |
| SGD (batch 20) | 68.28 ± 0.43% | +0.14 % | 70.61 ± 1.04% | 700, 400, 800 |
| Constant isotropic GLD | 68.19 ± 0.12% | +0.05 % | 68.55 ± 0.29% | 0, 0, 0 |
| Dynamic isotropic GLD | 68.15 ± 0.19% | +0.01 % | 68.55 ± 0.29% | 0, 0, 0 |
| Full-batch GD | 68.14 ± 0.09% | +0.00 % | 68.55 ± 0.29% | 0, 0, 0 |
| Diagonal GLD | 67.87 ± 0.26% | -0.27 % | 68.55 ± 0.29% | 0, 0, 0 |


Additionally we consider the power scaling run and add noise whose covariance matrix has a spectral distribution based upon the power law:

$$
\lambda_k = C s^{-k}
$$

for some constants $C, s > 0$ and where $k$ is the index of the diagonal of the covariance of the noise. 

<div id="fig-spectra" class="l-page">
  {% include figure.liquid
    path="assets/img/anisotropic-gradient-noise/power_law_accuracy.png"
    alt="Scaled expected quadratic loss during the first twenty noisy-gradient updates for several Hessian and noise spectra."
    caption="<strong>Figure 4.</strong> Fashion-MNIST test accuracy during GLD continuation from a shared full-batch GD checkpoint with 100% training accuracy. GLD noise has a power-law spectrum whose eigenvalues are either aligned with the Hessian eigenvalues (A) or assigned in reverse order as a control (B). We compare frozen and dynamically refreshed Hessian directions under fixed and dynamically refreshed noise-trace policies. The black curve shows continued full-batch GD. Dynamic quantities are recomputed every 10 updates; curves and shaded regions show the mean and standard error over three runs"
    class="img-fluid"
    zoomable=true
  %}
</div>


<div id="fig-spectra" class="l-page">
  {% include figure.liquid
    path="assets/img/anisotropic-gradient-noise/powerlaw_comparison.png"
    alt="Scaled expected quadratic loss during the first twenty noisy-gradient updates for several Hessian and noise spectra."
    caption="<strong>Figure 5.</strong> Fashion-MNIST test accuracy after 1,000 continuation updates for each power-law GLD configuration. Points show the mean over three runs, and error bars show the standard error. The vertical black line gives the accuracy obtained by continuing full-batch GD from the same checkpoint (\(68.14\%\)). Blue and orange indicate dynamic and frozen Hessian directions, respectively; each configuration additionally varies the noise-trace policy and whether the power-law noise spectrum is aligned or reversed relative to the Hessian spectrum"
    class="img-fluid"
    zoomable=true
  %}
</div>

The fixed trace uses the SGD noise trace at at the GD-3000 checkpoint. It does not update the magnitude of the covariance of the noise and is most faithful in the sense to our quadratic tests. Dynamic trace estimates the expected sqaure magnitude of the noise to match the noise magnitude of SGD. 

The dynamic Hessian recomputes the Hessian recomputes the Hessian $H_t = \nabla_\theta \mathcal{L}(\theta_t)$ over the whole training dataset, so there is no noise from mini-batch estimation. The frozen Hessian uses the Hessian of the model at the GD-3000 checkpoint. The results show that both the structure of the noise and magnitude of the noise matter. 



<!-- ### LLM Finetuning

As it infeasible practically to run a full GD iteration over the whole dataset, we are limited to either selecting a smaller size of dataset (often too small to converge to a local minima with relatively high training accuracy), or an alternative direction would be to finetune a pretrained model.  -->




## Conclusion

I began with a simple question: What effect does the noise in SGD have on training, and which properties of that noise produce the effect? The answer is more complicated than saying that more noise is better.
We began by expressing minibatch SGD as full-batch gradient descent with an additional zero-mean noise term. Near a stationary point, a quadratic approximation of the loss then gave us a precise measure of the initial response to this noise:

$$
\mathbb{E}_{\varepsilon_0}[F(\delta_1)]
=
\frac{\eta^2}{2}\operatorname{tr}(H\Sigma).
$$

Here, $$H$$ describes the local curvature of the loss and $$\Sigma $$ describes the covariance of the noise. Decomposing this trace gives

$$
\operatorname{tr}(H\Sigma)
=
\sum_{i,j}\lambda_i s_j(q_i^\top u_j)^2.
$$

The eigenvalues $\lambda_i$ and $s_j$ tell us how the curvature and noise variance are distributed, while $(q_i^\top u_j)^2$ measures the alignment between their directions. Consequently, two noise distributions with the same total variance and even the same eigenvalues can interact with the loss landscape very differently.

The quadratic experiments illustrate this distinction directly. Under a fixed noise trace, placing more variance in high-curvature directions produces a larger initial expected loss increase. Reversing the same noise spectrum can greatly reduce this effect. Alignment becomes irrelevant when either the Hessian or the noise covariance is isotropic, since there is then no preferred direction with which to align.

The Fashion-MNIST experiments suggest that these ideas remain relevant outside the exact quadratic setting. Structured GLD noise based on Hessian or gradient-covariance information could outperform isotropic noise and continued full-batch gradient descent in our runs. In the power-law experiments, the best-performing configuration used dynamically refreshed Hessian directions, a dynamically adjusted noise magnitude, and an aligned spectrum. This suggests that the structure, orientation, and magnitude of the noise can all matter during training.

A further practical limitation is the cost of measuring curvature in modern neural networks. The assumption that useful information is contained in the Hessian spectrum is not new, but computing its leading eigenvalues and eigenvectors becomes increasingly difficult as model size grows. Recent work is beginning to develop methods for studying Hessian structure at the scale of models with tens of billions of parameters<d-cite key="granziol2025hessformer,granziol2026hessian"></d-cite>. Such methods may eventually make it possible to test whether the relationships studied here persist at a much larger scale.

## Future Directions
Several questions remain open. One is whether the relationship between curvature, noise structure, and alignment persists during the fine-tuning of pretrained models. With methods such as LoRA, only a low-dimensional parameter subspace is trainable, so the relevant quantities may be the Hessian and noise covariance projected into that subspace.

Another direction is extending the analysis to AdamW. If $P_t$ denotes its time-dependent preconditioner, then the effective noise covariance is approximately

$$
\Sigma_t^{\mathrm{effective}}
=
P_t\Sigma_tP_t^\top.
$$

The relevant alignment may therefore involve the preconditioned loss geometry rather than the raw Hessian and noise covariance.
Momentum also accumulates a weighted history of gradient noise:

$$
\varepsilon_t+\beta\varepsilon_{t-1}
+\beta^2\varepsilon_{t-2}+\cdots.
$$

Pan et al.<d-cite key="pan2024accelerated"></d-cite> analyze convergence of stochastic heavy-ball momentum under anisotropic noise on quadratic objectives. However, it remains unclear how momentum transforms time-varying or correlated noise in neural networks and whether the resulting alignment affects generalization. Studying these effects during large-scale training and fine-tuning would bring the analysis closer to modern optimization practice.






















































<!-- 
We define ESCAPE efficieciny as a number:

$$
TODO
$$

which helps us measure on avergae how much is our loss. This is a measure of how mcuh we move away from the bottom! (And potentially escape it??). We use the Brownian motion to calculate the expectation with respect to $\theta_t$

However this is intractable, so oinstead we assume a QUADRATIC APPROXMATION!

Why are we studying this? Because we want better generalization. It is possible we get stuck in a local basin, and we want to leave this basin. In general moving uphill is how we leave the basin. The faster we leave the basin, the more likely we can move to another basin and perhaps converge better to a local minima! We end uup with something like

$$
\operatorname{tr}(H \Sigma)
$$

Then proposition shows us, under some assumptions (typically ill conditioned assumptions), that the rate of leaving such minima is based upon how ill-conditioned the problem is! -->



---

## Appendix
In the proofs below, we assume the following:
- The training dataset $\mathcal{D} := \{(x_i, y_i)\}_{i=1}^N$
- Our model parameters $\theta$
- $B_t$ is a uniformly sampled subset of size $m$:

$$
\mathbb{P}(B_t = b) := \frac{1}{\binom{N}{m}}, \qquad b \subseteq \{1, \ldots, N \}, \; |B| = m
$$

- Our per example loss, empirical losses and mini-batch losses are:

$$
\ell_i(\theta) := \ell(f_\theta(x_i), y_i) \qquad \mathcal{L}(\theta) := \frac{1}{N} \sum_{i=1}^N \ell_i(\theta) \qquad \widehat{\mathcal{L}}_{B_t}(\theta) := \frac{1}{m} \sum_{i \in B_t} \ell_i(\theta)
$$

where $f_\theta(x_i)$ is our model prediction


- Gradient & Mini Batch Gradients:

$$
g_i := \nabla_\theta \ell_i(\theta), \qquad g := \nabla_\theta L(\theta) = \frac{1}{N} \sum_{i=1}^N g_i \qquad \hat{g} := \frac{1}{m} \sum_{i \in B} \nabla_\theta \ell_i(\theta)
$$

We indeed have that $\hat{g}$ is an unbiased estimator of the gradient of the empirical loss:

$$
\begin{align*}
  \mathbb{E}_{B_t}[ \hat{g}] &= \mathbb{E}_{B_t} \left[ \frac{1}{m} \sum_{i \in B} \nabla_\theta \ell_i(\theta) \right] \\
  &= \mathbb{E}_{B_t} \left[\frac{1}{m} \sum_{i=1}^N {\mathbf{1}}_{i \in B_t} g_i \right] \\
  &= \frac{1}{m} \sum_{i=1}^N \mathbb{P}(i \in B_t) g_i \\
  &= \frac{1}{m} \sum_{i=1}^N \frac{\binom{N-1}{m-1}}{\binom{N}{m}} g_i \\
  &= \frac{1}{N} \sum_{i=1}^N g_i = g
\end{align*}
$$

- Define the noise as:

$$
\varepsilon := g - \hat{g} = -\frac{1}{m} \sum_{i=1}^N {\mathbf{1}}_{i \in B_t} \left(g_i - g\right)
$$

We make no assumptions about the distribution of the noise. Note however that:

$$
\mathbb{E}_{B_t}[\varepsilon] = 0
$$


### Covariance of Mini-Batch Noise
To find:

$$
\operatorname{Cov}_{B_t}(\hat{g}(\theta_t))
$$

by central limit theorem $\hat{g}(\theta_t) \sim \mathcal{N}(g, \Sigma^{\text{sgd}})$. Define 

$$
\varepsilon := g - \hat{g}_t

$$

$$
\begin{align*}
\Sigma^{\text{sgd}} := \operatorname{Cov}_{B_t}(\hat{g}(\theta_t))& = \mathbb{E}_{B_t} \left[(\hat{g}(\theta_t) - \nabla_\theta L(\theta) )(\hat{g}(\theta_t) - \nabla_\theta L(\theta))^\top \right] \\
&= \mathbb{E}_{B_t} \left[(\varepsilon)(\varepsilon)^\top \right]
\end{align*}
$$

Define:

$$
  C := \frac{1}{N}\sum_{i=1}^N (g_i -g)(g_i-g)^\top
$$

Note some additional properties to be used are:

$$
\sum_{i=1}^N (g_i -g) = Ng - \sum_{i=1}^N = g_i
$$

$$
\begin{align*}
  0 &= \left(\sum_{i=1}^N (g_i - g) \right)\left(\sum_{j=1}^N (g_j - g)^\top \right) \\
    &= \sum_{i, j} (g_i -g)(g_j-g)^\top \\
    &= \sum_i (g_i -g)(g_i-g)^\top + \sum_{i \neq j} (g_i -g)(g_j- g)^\top 
\end{align*}
$$

Hence:

$$
\sum_{i \neq j} (g_i -g)(g_j-g)^\top = -\sum_{i=1}^N (g_i -g)(g_i-g)^\top = -NC \tag{2}
$$

Also:

$$
\begin{align*}
  C &= \frac{1}{N} \sum_{i=1}^N (g_i -g)(g_i-g)^\top \\
    &= \frac{1}{N} \sum_{i=1}^N g_i g_i^\top - \left(\frac{1}{N} \sum_{i=1}^N g_i\right)g^\top - g\left(\frac{1}{N} \sum_{i=1}^N g_i\right)^\top + g g^\top \\
    &= \frac{1}{N} \sum_{i=1}^N g_i g_i^\top - gg^\top - gg^\top + gg^\top \\
    &= \frac{1}{N} \sum_{i=1}^N g_i g_i^\top - g g^\top \tag{3}
\end{align*}
$$

Computation of $\varepsilon \varepsilon^\top$:

$$
\begin{align*}
  \varepsilon \varepsilon^\top &= \frac{1}{m^2} \sum_{i=1}^N \sum_{j=1}^N {\mathbf{1}}_{i \in B_t} {\mathbf{1}}_{j \in B_t} (g_i - g)(g_j - g)^\top \\
\end{align*}
$$

Hence:

$$
\begin{align*}
  \text{Cov}_{B_t}(\varepsilon \varepsilon^\top) &= \mathbb{E}_{B_t}(\varepsilon \varepsilon^\top) \\
  &= \frac{1}{m^2} \sum_{i=1}^N \sum_{j=1}^N \mathbb{E}_{B_t} \left[ {\mathbf{1}}_{i \in B_t} {\mathbf{1}}_{j \in B_t} \right] (g_i - g)(g_j - g)^\top \\
  &= \frac{1}{m^2} \left[\sum_i \underbrace{\mathbb{P}(i \in B_t)}_{\frac{m}{N}} (g_i - g)(g_i - g)^\top + \sum_{i \neq j} \underbrace{\mathbb{P}(i ,j \in B_t)}_{\frac{m(m-1)}{N(N-1)}} (g_i -g)(g_j-g)^\top  \right] \\
  &= \frac{1}{m^2} \left[mC - \frac{m(m-1)}{N(N-1)} (NC) \right] \\
  &= \frac{C}{m}\left( 1 - \frac{m-1}{N-1} \right) \\
  &= \frac{N-m}{m(N-1)} C
\end{align*}
$$

Therefore:

$$
\operatorname{Cov}_{B_t}(\epsilon_{B_t}) =
\frac{N-m}{m(N-1)}
\left[
\frac1N\sum_{i=1}^{N}
\nabla_\theta\ell_i(\theta)\nabla_\theta\ell_i(\theta)^\top
-
\nabla_\theta L(\theta)\nabla_\theta L(\theta)^\top
\right]
$$

When the size of the mini batch is small compared to the total size of the dataset, then:

$$
\frac{N-m}{N-1} = 1 - \frac{m-1}{N-1} \approx 1
$$


### Time Discretization of SDE
In this section, we show that GLD is a time discritization of the following SDE with constant timestep difference $h = \eta$.

$$
d \theta_t = -\nabla_\theta \mathcal{L}(\theta_t) dt + \sqrt{\eta \Sigma_t} dW_t
$$

Consider $h > 0$ (numerical time step size) and consider:

$$
t_k = kh, \qquad k = 0, 1, 2, \ldots
$$

Note $W_t \in \mathbb{R}^D$ is a standard Brownian motion so it satisfies, $\forall b > a$:

$$
W_b - W_a \sim \mathcal{N}(0, (b-a) I_D)
$$

Defining:

$$
\begin{gather*}
\Delta W_k := W_{t_{k+1}} - W_{t_k} \mathcal{N}(0, h I_D) \\
\Delta W_k = \sqrt{h} z_k, \qquad z_k \sim \mathcal{N}(0, I_D)
\end{gather*}
$$

We can just use Euler-Maruyama approximation and obtain:

$$
\theta_{k+1} = \theta_k - h \nabla L(\theta_k) + \sqrt{\eta h} \Sigma(\theta_k)^{1/2} z_k
$$

Taking $h = \eta$ and using the fact that:

$$
\varepsilon_k = \Sigma(\theta_k)^{1/2} z_k
$$

and we obtain the GLD update rule and this shows that GLD is the discretizaton of $\eqref{eq:SDE-GLD}$ with $h = \eta$. 





### Transformation of Variables

We want to that translating the probelm preserves the noisy gradient dynamics and expected loss increases. So fix $ \theta_0 \in \mathbb{R}^D$ and let $\eta > 0$. Define:

$$
F: \mathbb{R}^D \to \mathbb{R}, \quad F(d) := \mathcal{L}(\theta_0 + d) - \mathcal{L}(\theta_0)
$$

Let $(\Omega, \mathcal{F}, \mathbb{P})$ be a probability space and $(\varepsilon_k)_{k \geq 0}$ be random vectors on $(\Omega, \mathcal{F}, \mathbb{P})$. We consider two updates:

$$
\begin{align*}
\theta_0 &= \theta_0, &\theta_{k+1} &= \theta_k - \eta \nabla \mathcal{L}(\theta_k) + \eta \varepsilon_k \\
\delta_0 &= 0, &\delta_{k+1} &= \delta_k - \eta \nabla F(\delta_k) + \eta \varepsilon_k
\end{align*}
$$

For every $k \geq 0$ we have:

$$
\delta_k = \theta_k - \theta_0 \tag{1}
$$

Hence:

$$
\forall k \geq 0, \; \forall \omega \in \Omega, \; F(\delta_k(\omega)) = \mathcal{L}(\theta_k(\omega)) - \mathcal{L}(\theta_0)
$$

Furthermore writing:

$$
P_k := \mathbb{P} \circ \theta_k^{-1} \qquad Q_k := \mathbb{P} \circ \delta_k^{-1}
$$

we have that the escape efficiency for both parameters are the same. That is:

$$
\mathbb{E}_{\delta_k \sim Q_k} [F(\delta_k)] = \mathbb{E}_{\theta_k \sim P_k}[\mathcal{L}(\theta_k) - \mathcal{L}(\theta_0)]
$$

Hence it suffices to consider the dynamics on $\delta_k$ and the quadratic centered at $0$.


*Proof*: First by chain rule, it is clear that 

$$
\nabla F (d) = \nabla \mathcal{L}(\theta_0 + d)
$$

We can show (1) by induction on $k$.

$k = 0$: True

$k \geq 0$: Let $\omega \in \Omega$. We have:

$$
\begin{align*}
  \delta_{k+1}(\omega) &= \delta_{k}(\omega) - \eta \nabla F(\delta_k(\omega)) + \eta \varepsilon_k \\
  &= \theta_{k}(\omega) - \theta_0 - \eta \nabla \mathcal{L}(\theta_k(\omega)) + \eta \varepsilon_k \\
  &= \theta_{k+1}(\omega) - \theta_0
\end{align*}
$$


To see the expectations are equal, this is a standard pushforward measure argument. First define:

$$
\begin{gather*}
g: \mathbb{R}^D \to \mathbb{R} \\
g(\theta) = \mathcal{L}(\theta) - \mathcal{L}(\theta_0)
\end{gather*}
$$

Hence $g(\theta_k) = \mathcal{L}(\theta_k)- \mathcal{L}(\theta_0)$. 

We have:

$$
\begin{align*}
  \mathbb{E}_{\delta_k \sim Q_k} [F(\delta_k)] &= \int_{\mathbb{R}^D} F dQ_k \\
  &= \int_\Omega (F \circ \delta_k) d \mathbb{P} & \text{pushforward measure theorem} \\
  &= \int_\Omega (g \circ \theta_k) d \mathbb{P} & F \circ \delta_k = g \circ \theta_k \\
  &= \int_{\mathbb{R}^D} g d P_k \\
  &= \mathbb{E}_{\theta_k \sim P_k}\left[\mathcal{L}(\theta_k) - \mathcal{L}(\theta_0) \right]

\end{align*}
$$

<!-- # Rewriting of Standard Gaussian

Let $\varepsilon \sim \mathcal{N}(0, \Sigma)$ where $\Sigma \succeq 0$ and $\Sigma = Q DQ^\top$. Then $\exists z \sim \mathcal{N}(0, I)$ such that  -->






<!-- 
### Appendix



#### Simple Quadratics
We first consider the effect of this noise on simple quadratics. Consider the function:

$$
f(W) = \frac{1}{2} \operatorname{tr}{(W-W_*)^\top A (W-W_*)}
$$
where $W \in \mathbb{R}^{p \times q}$ is the parameters (weights) of our model that we train. We want to solve:

$$
W_* = \underset{W}{\operatorname{argmin}} f(W)
$$
Which in this case, is indeed $W_*$. In order to test generalization however, we need some sort of prediction function. In order to do this, we can rewrite this minimization problem in an more familiar equivalent formulation of linear regression. To do so we define the following:
- $X \in \mathbb{R}^{n \times p}$: $n$ input training examples. $p$ features. 
- $Y \in \mathbb{R}^{n \times q}$: $n$ observed outputs, each with $q$ output features. 
- $W \in \mathbb{R}^{p \times q}$: Parameters we train

Linear regression minimizes the loss:

$$
L_{train}(W) := \frac{1}{2n} \|XW - Y \|_F^2
$$
where $\| A \|_F := \operatorname{tr}(A^\top A) = \sqrt{\sum_{i, j} A_{ij}^2}$ is the Frobenius norm. We want to find:

$$
W_* = \operatorname{argmin} f(W)
$$

There is a well known solution, particularly since $f(W)$ is convex and differentiable and so it attains a global minimum (not necessarily unique, unless $X^\top X \succ 0$, i.e. $X$ has full column rank):

$$
W_* = (X^\top X)^\dagger X^\top Y = X^\dagger Y
$$

Note that taking $X = U \Sigma V^\top$ to be the SVD of $X$ (omitting the zero singular values)

$$
\begin{align*}
X^\top X W_* &= V \Sigma^2 V^\top V \Sigma^{-1} U^\top Y \\
&= V \Sigma U^\top Y \\
&= X^\top Y
\end{align*}
$$

Hence:

$$
\begin{align*}
L(W) &= \frac{1}{2n} \|XW - Y \|_F^2 \\
    &= \frac{1}{2n} \| X\underbrace{E}_{(W - W_*)} - \underbrace{R}_{(Y - X W_*)} \|_F^2 \\
    &= \frac{1}{2n} \left(\|R\|_F^2 + 2 \operatorname{tr}(E^\top X^\top R) + \| XE \|_F^2 \right) \\
    &= L(W_*) + \frac{1}{2n} \|XE\|_F^2
\end{align*}
$$

Writing $A = X^\top X / n$ we have:

$$
L(W) - L(W^*) = \frac{1}{2} \operatorname{tr}\left( (W- W_*)^\top A (W-W_*) \right)
$$

Since $L(W^*)$ is constant, we can see that minimizing the RHS is equivalent to minimizing the LHS for well chosen $A$.  -->

<!-- 
Experimental Setup for LLM:
- LoRA FineTuning: on pretrained GPT2 that used SGD and then on pretrained GPT2 that used AdamW or Adam
- Fintune using SGD
- Finetune using GLD with the different variants of the noise added
- Then run on test set and test generalization
- Then for each, if needed change the constants in the power law/the Sigma so we can obtain good results.  -->







<!-- ### Sources
- SGD as Bayesian Inference: https://arxiv.org/pdf/1704.04289
  - See assumption 4: Assumes that the stionary distribution of the iterates is constrained to a region well approximated by a quadratic function

- Papers on the Structure of the Hessian wrt Loss
  - Investigating the Overlooked Hessian Structure: From CNNs to LLMs: 
    - https://proceedings.mlr.press/v267/tang25d.html
  - What does it Mean to Be a Transformer: Theoretical Hessian Analysis: https://arxiv.org/abs/2410.10986
  - Why Transformers need Adam: Hessian Perspective: https://arxiv.org/abs/2402.16788
  - Analytic Insights into Structure and Rank of Neural Network Hessian Maps: https://arxiv.org/abs/2106.16225
- Papers on the Structure of the Covariance Matrix of the Noise from SGD:
  - On the Overlooked Structure of Stochastic Gradients: https://arxiv.org/pdf/2212.02083
-- -->

