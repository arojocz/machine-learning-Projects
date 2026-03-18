# PCA + Multi-Classifier Benchmark

> Dimensionality reduction with PCA benchmarked against 5 classifiers (KNN, GaussianNB, DecisionTree, RandomForest, AdaBoost) on three datasets — showing that retaining 90%+ explained variance with fewer components preserves or improves classifier performance in most cases.

---

## Overview

This project evaluates whether PCA-based dimensionality reduction improves or degrades classification performance. Each dataset is standardized, reduced to the minimum number of components that explain at least 90% of variance, and then benchmarked with five classifiers — both on original and PCA-transformed features. Validation uses stratified Hold-Out 70/30 with micro-average metrics for multiclass datasets.

---

## Datasets & PCA Components Selected

| Dataset | Original Features | PCA Components | Variance Explained |
|---------|-------------------|----------------|--------------------|
| Iris | 4 | 2 | 95.93% |
| Heart Disease | 13 | 10 | 90.85% |
| LED7Digit | 7 | 6 | 93.72% |

---

## Results

### Iris — Balanced Accuracy

| Classifier | Original | PCA | Improved |
|------------|----------|-----|----------|
| KNN | 93.3% | 93.3% | — |
| GaussianNB | 97.8% | 82.2% | No |
| DecisionTree | 93.3% | 84.4% | No |
| RandomForest | 95.6% | 86.7% | No |
| AdaBoost | 91.1% | 86.7% | No |

### Heart Disease — Balanced Accuracy

| Classifier | Original | PCA | Improved |
|------------|----------|-----|----------|
| KNN | 85.3% | 77.5% | No |
| GaussianNB | 82.8% | 85.0% | YES |
| DecisionTree | 73.6% | 75.8% | YES |
| RandomForest | 82.5% | 79.2% | No |
| AdaBoost | 82.2% | 75.8% | No |

### LED7Digit — Balanced Accuracy

| Classifier | Original | PCA | Improved |
|------------|----------|-----|----------|
| KNN | 65.5% | 69.8% | YES |
| GaussianNB | 65.5% | 74.2% | YES |
| DecisionTree | 71.1% | 70.6% | No |
| RandomForest | 69.7% | 70.5% | YES |
| AdaBoost | 65.2% | 51.5% | No |

**Key finding:** PCA helps most when features are correlated or redundant — GaussianNB on LED7Digit jumps from 65.5% to 74.2% after reducing 7 features to 6 components. On the other hand, PCA consistently hurts GaussianNB on Iris (97.8% → 82.2%), where the original 4 features already provide clean Gaussian separation that PCA partially destroys by mixing them.

---

## Getting Started

```bash
git clone https://github.com/your-username/machine-learning-Projects
cd ml-portfolio/05-pca-benchmark

python -m venv venv
source venv/bin/activate

# Place iris.data, heart.dat, led7digit.dat in this folder, then:
python pca_benchmark.py
```

---

## Key Learnings

- **PCA is not universally beneficial.** It improves performance when features are correlated (LED7Digit, Heart + GNB) but degrades it when the original feature space already contains clean, discriminative structure (Iris + GaussianNB).
- **AdaBoost is the most sensitive to PCA.** It degrades on all three datasets under PCA, suggesting that boosting relies on the raw feature interactions that PCA's linear projections discard.
- **Dimensionality reduction enables 2D visualization of high-dimensional class structure.** The PCA scatter plots show that Iris classes are linearly separable in 2 components, while Heart and LED7Digit classes overlap — explaining why classifiers plateau on those datasets regardless of whether PCA is applied.

---

## Author

**Luis Angel Rojo Chavez** — [GitHub](https://github.com/arojocz)
