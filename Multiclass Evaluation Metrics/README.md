# Multiclass Evaluation — KNN vs Gaussian Naive Bayes

> Head-to-head comparison of KNN (k=3) and Gaussian Naive Bayes on the UCI Wine dataset using Leave-One-Out cross-validation, with full per-class metrics and Macro, Weighted, and Micro averaging. GaussianNB achieves 97.8% macro F1 vs 71.5% for KNN.

---

## Overview

This project builds a multiclass evaluation pipeline that goes beyond accuracy. For each classifier, a 3×3 global confusion matrix is decomposed into three binary (one-vs-rest) matrices to compute per-class Sensitivity, Specificity, Balanced Accuracy, Precision, and F1-Score, aggregated with three averaging strategies.

---

## Dataset

| Property | Details |
|----------|---------|
| Source | [UCI Wine](https://archive.ics.uci.edu/dataset/109/wine) |
| Instances | 178 |
| Features | 13 chemical properties |
| Classes | 3 wine cultivars (59 / 71 / 48 samples) |
| Validation | Leave-One-Out |

---

## Results

### KNN (k=3)

| Class | Sensitivity | Specificity | Balanced Acc | Precision | F1-Score |
|-------|-------------|-------------|--------------|-----------|----------|
| Class 1 | 86.4% | 88.2% | 87.3% | 78.5% | 82.3% |
| Class 2 | 69.0% | 86.0% | 77.5% | 76.6% | 72.6% |
| Class 3 | 60.4% | 84.6% | 72.5% | 59.2% | 59.8% |
| **Macro** | 72.0% | 86.3% | 79.1% | 71.4% | **71.5%** |
| **Micro** | 72.5% | 72.5% | 72.5% | 72.5% | **72.5%** |

### Gaussian Naive Bayes

| Class | Sensitivity | Specificity | Balanced Acc | Precision | F1-Score |
|-------|-------------|-------------|--------------|-----------|----------|
| Class 1 | 96.6% | 100.0% | 98.3% | 100.0% | 98.3% |
| Class 2 | 97.2% | 98.1% | 97.7% | 97.2% | 97.2% |
| Class 3 | 100.0% | 98.5% | 99.2% | 96.0% | 98.0% |
| **Macro** | 97.9% | 98.9% | 98.4% | 97.7% | **97.8%** |
| **Micro** | 97.8% | 97.8% | 97.8% | 97.8% | **97.8%** |

**Key finding:** GaussianNB dominates KNN by ~26 F1 points. Wine features are continuous chemical measurements that naturally fit Gaussian distributions per class — ideal for GNB. KNN's poor performance is largely a normalization problem: features like proline (~700) overwhelm alcohol (~12) in Euclidean distance, biasing neighbor selection entirely.

---

## Getting Started
```bash
git clone https://github.com/your-username/ml-portfolio.git
cd ml-portfolio/02-multiclass-evaluation

python -m venv venv
source venv/bin/activate

# Place wine.data in this folder, then:
python multiclass_cm.py

---

## Requirements

```
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
tabulate>=0.9.0
```

---

## Key Learnings

- GaussianNB nearly perfectly classifies chemical data because its Gaussian assumption matches the data distribution almost exactly.
- KNN without normalization is unreliable when features have very different scales — normalizing would likely close most of the 26-point gap.
- Micro average equals overall accuracy in multiclass settings, confirming both metrics as cross-checks.

---

## Author

**Luis Angel Rojo Chavez** — [GitHub](https://github.com/your-username)
