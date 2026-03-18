# Gradient Descent & Linear Regression

> Gradient descent implemented from scratch using symbolic differentiation (SymPy) and applied to linear regression on two datasets — Diabetes (sklearn) and Auto MPG (UCI) — optimizing MSE over 30,000 epochs with Hold-Out 70/30 validation.

---

## Overview

This project covers two things: a symbolic gradient descent explorer that accepts any mathematical function and finds its minimum interactively, and a full linear regression pipeline trained via batch gradient descent. The regression model is evaluated on two real datasets, tracking cost convergence across epochs and reporting MSE on unseen test data.

---

## What's Included

| Module | Description |
|--------|-------------|
| Symbolic gradient descent | Accepts any f(x) via input, computes derivative with SymPy, iterates to minimum |
| Linear regression — Diabetes | sklearn built-in dataset, 10 features, 70/30 split |
| Linear regression — Auto MPG | UCI dataset, 5 features, standardized, 70/30 split |

---

## Datasets

| Dataset | Instances | Features | Target | Source |
|---------|-----------|----------|--------|--------|
| Diabetes | 442 | 10 | Disease progression | sklearn |
| Auto MPG | ~392 | 5 (cylinders, displacement, horsepower, weight, acceleration) | Miles per gallon | [UCI](https://archive.ics.uci.edu/ml/datasets/auto+mpg) |

---

## Methodology

**Hypothesis:** ŷ = θᵀx

**Cost function (MSE):**

J(θ) = (1/2m) Σ (θᵀx⁽ⁱ⁾ − y⁽ⁱ⁾)²

**Parameter update:**

θ := θ − α · ∇J(θ)

| Parameter | Diabetes | Auto MPG |
|-----------|----------|----------|
| Learning rate (α) | 0.005 | 0.0005 |
| Epochs | 30,000 | 30,000 |
| Validation | Hold-Out 70/30 | Hold-Out 70/30 |
| Features scaled | No (sklearn normalized) | Yes (StandardScaler) |

---

## Results

| Dataset | MSE (test) |
|---------|------------|
| Diabetes | ~2900 |
| Auto MPG | ~17–18 |

Cost curves plotted across all epochs confirm stable convergence without oscillation in both cases.

---

## Getting Started

```bash
git clone https://github.com/your-username/ml-portfolio.git
cd ml-portfolio/06-gradient-descent-regression

python -m venv venv
source venv/bin/activate

# Run the notebook or script
jupyter notebook gradient_descent.ipynb
```

---

## Key Learnings

- **Learning rate is critical and dataset-dependent.** Auto MPG requires α=0.0005 (10x smaller than Diabetes) because its raw feature scales — even after standardization — produce larger gradients that cause divergence at higher rates.
- **Symbolic differentiation with SymPy makes gradient descent interpretable.** By computing the exact derivative analytically rather than numerically, the explorer exposes exactly what gradient the algorithm is following at each step.
- **30,000 epochs is sufficient for convergence on both datasets.** The cost curves flatten well before the epoch limit, suggesting that early stopping could reduce compute without sacrificing accuracy.

---

## Author

**Luis Angel Rojo Chavez** — [GitHub](https://github.com/arojocz)
