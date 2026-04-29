# Logistic Regression Classifier

Logistic Regression implemented from scratch in Python, utilizing Maximum Likelihood Estimation (MLE) through Gradient Descent, featuring $L_2$ Regularization and Z-score standardization to optimize convergence.

---

## Overview

This project implements a binary Logistic Regression model aimed at finding an approximating function $h:\mathcal{X}\rightarrow[0,1]$ that minimizes the model's generalization error. The implementation transforms a linear signal $s = w^T x + b$ into a probability measure using the sigmoid function, providing a robust framework for supervised learning tasks.

---

## Table of Contents

- [Mathematical Foundation](#mathematical-foundation)
- [Methodology](#methodology)
- [Key Challenges & Mitigations](#key-challenges--mitigations)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Key Learnings](#key-learnings)
- [Next Steps](#next-steps)

---

## Mathematical Foundation

The implementation relies on the following dimensional structure for matrix operations:

| Component | Symbol | Dimension | Description |
| :--- | :--- | :--- | :--- |
| **Weight Vector** | $w$ | $\mathbb{R}^{n \times 1}$ | Model parameters. |
| **Input Matrix** | $X$ | $\mathbb{R}^{m \times n}$ | $m$ examples with $n$ features. |
| **Linear Signal** | $s$ | $\mathbb{R}^{m \times 1}$ | $s = Xw + b$. |
| **Prediction** | $\hat{y}$ | $\mathbb{R}^{m \times 1}$ | $\hat{y} = \sigma(s)$. |
| **Cost ($E_{in}$)** | $E_{in}$ | $\mathbb{R}$ | Average Binary Cross-Entropy. |

---

## Methodology

### Algorithm
* **Hypothesis Space:** Defined by the composition of a linear operation and the logistic function $\sigma(s)=\frac{1}{1+e^{-s}}$.
* **Optimization:** Parameters are found by minimizing the Negative Log-Likelihood (Binary Cross-Entropy).
* **Gradient Descent:** Iterative updates using the gradient $\nabla_{w}E_{in}=\frac{1}{N}\sum_{n=1}^{N}(\hat{y}_{n}-y_{n})x_{n}$.

### Preprocessing
* **Standardization:** Applies Z-score normalization $x'=\frac{x-\mu}{\sigma}$ to prevent the "bouncing" effect during training and ensure faster convergence.

---

## Key Challenges & Mitigations

* **Perfect Linear Separation:** Mitigated using **$L_2$ Regularization (Ridge)** to prevent weight divergence to infinity.
* **Multicollinearity:** Addressed through dimensionality reduction or **$L_1$ Regularization (Lasso)** to stabilize parameter estimation.
* **Class Imbalance:** Solved via **Weighted Cross-Entropy** or resampling techniques like **SMOTE**.

---

## Getting Started

```bash
# Clone the repository
git clone [https://github.com/arojocz/machine-learning-Projects.git](https://github.com/arojocz/machine-learning-Projects.git)
cd ml-portfolio/07-logistic-regression