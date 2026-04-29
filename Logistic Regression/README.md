# Logistic Regression Classifier

> [cite_start]Logistic Regression implemented from scratch in Python, utilizing Maximum Likelihood Estimation (MLE) through Gradient Descent [cite: 53, 146][cite_start], featuring $L_2$ Regularization and Z-score standardization to optimize convergence[cite: 170, 184].

---

## Overview

[cite_start]This project implements a binary Logistic Regression model aimed at finding an approximating function $h:\mathcal{X}\rightarrow[0,1]$ that minimizes the model's generalization error[cite: 9]. [cite_start]The implementation transforms a linear signal $s = w^T x + b$ into a probability measure using the sigmoid function [cite: 21, 28][cite_start], providing a robust framework for supervised learning tasks[cite: 8].

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

[cite_start]The implementation relies on the following dimensional structure for matrix operations[cite: 226]:

| Component | Symbol | Dimension | Description |
| :--- | :--- | :--- | :--- |
| **Weight Vector** | $w$ | $\mathbb{R}^{n \times 1}$ | [cite_start]Model parameters[cite: 226]. |
| **Input Matrix** | $X$ | $\mathbb{R}^{m \times n}$ | [cite_start]$m$ examples with $n$ features[cite: 226]. |
| **Linear Signal** | $s$ | $\mathbb{R}^{m \times 1}$ | [cite_start]$s = Xw + b$[cite: 226]. |
| **Prediction** | $\hat{y}$ | $\mathbb{R}^{m \times 1}$ | [cite_start]$\hat{y} = \sigma(s)$[cite: 226]. |
| **Cost ($E_{in}$)** | $E_{in}$ | $\mathbb{R}$ | [cite_start]Average Binary Cross-Entropy[cite: 226]. |

---

## Methodology

### Algorithm
* **Hypothesis Space:** Defined by the composition of a linear operation and the logistic function $\sigma(s)=\frac{1}{1+e^{-s}}$[cite: 19, 20].
* [cite_start]**Optimization:** Parameters are found by minimizing the Negative Log-Likelihood (Binary Cross-Entropy)[cite: 69, 71].
* [cite_start]**Gradient Descent:** Iterative updates using the gradient $\nabla_{w}E_{in}=\frac{1}{N}\sum_{n=1}^{N}(\hat{y}_{n}-y_{n})x_{n}$[cite: 153, 157].

### Preprocessing
* **Standardization:** Applies Z-score normalization $x'=\frac{x-\mu}{\sigma}$ to prevent the "bouncing" effect during training and ensure faster convergence[cite: 173, 178].

---

## Key Challenges & Mitigations

* [cite_start]**Perfect Linear Separation:** Mitigated using **$L_2$ Regularization (Ridge)** to prevent weight divergence to infinity[cite: 184, 185].
* **Multicollinearity:** Addressed through dimensionality reduction or **$L_1$ Regularization (Lasso)** to stabilize parameter estimation[cite: 191].
* **Class Imbalance:** Solved via **Weighted Cross-Entropy** or resampling techniques like **SMOTE**[cite: 196, 200].

---

## Project Structure
02-logistic-regression/
│
├── implementation.py       # Computational graph: Forward & Backward pass ├── preprocessing.py        # Z-score normalization logic 
├── requirements.txt
└── README.md  

---

## Getting Started

```bash
# Clone the repository
git clone [https://github.com/arojocz/ml-portfolio.git](https://github.com/arojocz/ml-portfolio.git)
cd ml-portfolio/02-logistic-regression

# View the original implementation
# [https://colab.research.google.com/drive/1icvrKqeRMo29B_0iSrf4Bcq-sXFHn15q](https://colab.research.google.com/drive/1icvrKqeRMo29B_0iSrf4Bcq-sXFHn15q)
