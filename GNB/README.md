# Gaussian Naïve Bayes Classifier

> Gaussian Naïve Bayes implemented from scratch using log-likelihood and Gaussian PDF, evaluated on three datasets with Leave-One-Out and 10-Fold Cross-Validation — achieving 96.5% balanced accuracy on Iris and 82.9% on Heart Disease.

---

## Overview

This project implements Gaussian Naïve Bayes from scratch without using scikit-learn's GaussianNB. The classifier estimates per-class mean and standard deviation during training and predicts using log-posterior probabilities under a Gaussian density assumption. Evaluated across three benchmark datasets of increasing complexity using two validation strategies, with per-class Sensitivity, Specificity, and Balanced Accuracy reported.

---

## Datasets

| Dataset | Instances | Classes | Type |
|---------|-----------|---------|------|
| Iris | 150 | 3 | Multiclass |
| Heart Disease | 270 | 2 | Binary |
| LED7Digit | 500 | 10 | Multiclass |

---

## Results

### Leave-One-Out

| Dataset | Balanced Accuracy | Sensitivity | Specificity |
|---------|-------------------|-------------|-------------|
| Iris | 96.5% | 95.3% | 97.7% |
| Heart | 82.9% | 82.9% | 82.9% |
| LED7Digit | 82.8% | 69.2% | 96.5% |

### 10-Fold Cross-Validation

| Dataset | Balanced Accuracy | Sensitivity | Specificity |
|---------|-------------------|-------------|-------------|
| Iris | 96.0% | 94.7% | 97.3% |
| Heart | 83.6% | 83.6% | 83.6% |
| LED7Digit | 80.0% | 64.1% | 96.0% |

**Key finding:** GNB handles binary and low-class problems well (Iris ~96%, Heart ~83%) but sensitivity drops to ~69% on LED7Digit's 10-class problem — when classes share similar Gaussian distributions, the conditional independence assumption breaks down and misclassifications increase. LOO consistently outperforms 10-Fold, suggesting the model benefits from maximum training data.

---

## Getting Started

```bash
git clone https://github.com/your-username/ml-portfolio.git
cd ml-portfolio/03-gaussian-naive-bayes

python -m venv venv
source venv/bin/activate

# Place iris.data, heart.dat, led7digit.dat in this folder, then:
python multiclass_cm.py
```

---

## Key Learnings

- Implementing GNB from scratch reinforces how the Gaussian PDF and log-prior combine into a log-posterior — using log space avoids numerical underflow when multiplying many small probabilities across 13+ features.
- Heart Disease shows identical Sensitivity and Specificity in both validation methods, which happens in balanced binary datasets where errors are distributed symmetrically across both classes.
- LED7Digit's high Specificity (~96%) alongside low Sensitivity (~69%) reveals a classic imbalance in one-vs-rest evaluation: the model is good at rejecting wrong classes but struggles to correctly identify the right one in a 10-class space.

---

## Author

**Luis Angel Rojo Chavez** — [GitHub](https://github.com/arojocz)
