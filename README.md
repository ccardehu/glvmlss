# Generalized Latent Variable Models for Location, Scale, and Shape parameters

### Overview

An **`R`** package for fitting Generalized Latent Variable Models for Location,
Scale, and Shape parameters (GLVM-LSS, Cárdenas-Hurtado et al., 2025+).

### Background

The GLVM-LSS framework extends traditional latent variable models (LVMs) by allowing distributional parameters beyond the mean (such as variance, skewness, and kurtosis) to be functions of latent variables.

Consider the following LVM:

$$f(\mathbf{y}) = \int_{\mathbb{R}^q} \prod_{i=1}^p f_i(y_i \mid \mathbf{z}; \mathbf{\theta}_i) ~ p(\mathbf{z}; \Phi) ~ d\mathbf{z} , $$

where:

- $$\mathbf{y} = (y_1, ... , y_p)^\top \in \mathbb{R}^p$$ are the observed variables,
- $$\mathbf{z} = (z_1, ... , z_q)^\top \in \mathbb{R}^q$$ are latent variables with distribution $$p(\mathbf{z}; \Phi) \sim \mathbb{N}(0,\Phi)$$, where $$\Phi$$ is a covariance matrix,
- $$\mathbf{\theta}_i = (\mu_i, \sigma_i, \tau_i, \nu_i)^\top$$ is a vector of distributional parameters -- location $$\mu_i$$, scale $$\sigma_i$$, and shape $$(\tau_i,\nu_i)$$ -- for item $$i$$,
  characterizing the conditional distribution $$f_i(y_i \mid \mathbf{z}; \mathbf{\theta}_i)$$.

In the GLVM-LSS framework, an arbitrary distributional parameter $$\varphi_i(\mathbf{z}) \in \theta_i$$ is expressed as a function of the latent variables through a link function and factor loadings:

$$v_{i,\varphi}(\varphi_i(\mathbf{z})) = \eta_{i,\varphi} := \alpha_{i0,\varphi} + \sum_{j=1}^{q} \alpha_{ij,\varphi} z_j, $$

The link function ($$v_{i,\varphi}$$) can be identity, log, logit, or any other suitable monotone function. $\eta_{i,\varphi}$ represents the linear combination of latent variables, with intercept $\alpha_{i0,\varphi}$
and factor loadings grouped in the $$q$$-dimensional vector $\alpha_{i, \varphi} = (\alpha_{i1,\varphi},...,\alpha_{iq,\varphi})^\top$.

By expressing the distributional parameters characterizing each $$f_i(y_i \mid \mathbf{z}; \mathbf{\theta}_i)$$, the GLVM-LSS framework allows for modeling conditional higher order moments,
such as the variance, skewness, and kurtosis, as functions of the latent variables.

Current implementation of the **``glvmlss``** package allows for mixed data following Normal, Bernoulli, Beta, and Skew-Normal distributions.

### Installation

You can install the development version of **``glvmlss``** from [GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("ccardehu/glvmlss")
```

### Literature

- Cárdenas-Hurtado, C., Moustaki, I., Chen, Y., & Marra, G. (2025).
  “Generalized Latent Variable Models for Location, Scale, and Shape parameters” (Accepted for Publication in Psychometrika).
    
- Cárdenas-Hurtado, C., (2023), “Generalized Latent Variable Models for Location, Scale, and Shape parameters”,
  PhD Thesis. Department of Statistics, The London School of Economics and Political Science
