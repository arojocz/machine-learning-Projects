# K-Nearest Neighbors Classifier

> KNN implemented from scratch in Python, evaluated across three benchmark datasets using 5-Fold Cross-Validation and Leave-One-Out, achieving up to 97.5% balanced accuracy on Iris.

---

## Overview

This project implements the K-Nearest Neighbors algorithm from scratch using Euclidean distance, without relying on scikit-learn's KNN implementation. The classifier is evaluated on three datasets of increasing difficulty — Iris, LED7Digit, and Heart Disease — using two cross-validation strategies. The goal is to understand how KNN behaves across different feature spaces, class distributions, and validation methods.

---

## Table of Contents

- [Datasets](#datasets)
- [Methodology](#methodology)
- [Results](#results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Key Learnings](#key-learnings)
- [Next Steps](#next-steps)

---

## Datasets

| Dataset | Instances | Classes | Type |
|---------|-----------|---------|------|
| Iris | 150 | 3 | Multiclass |
| LED7Digit | 500 | 10 | Multiclass |
| Heart Disease | 270 | 2 | Binary |

All datasets were loaded from `.data` / `.dat` files with no additional preprocessing or normalization applied.

---

## Methodology

### Algorithm
KNN implemented from scratch — custom Euclidean distance function, manual neighbor voting, and class-weighted confusion matrix aggregation across folds.

### Hyperparameters
- **k = 5** neighbors (fixed across all experiments)
- **Distance metric:** Euclidean

### Validation strategies
- **5-Fold Cross-Validation** — dataset split into 5 equal partitions, model trained on 4 and tested on 1 each round
- **Leave-One-Out (LOO)** — each sample used once as a test case; equivalent to N-Fold where N = dataset size

### Evaluation metrics
Since accuracy alone is misleading for multiclass and imbalanced datasets, all results are reported as:

- **Recall** = TP / (TP + FN)
- **Specificity** = TN / (TN + FP)
- **Balanced Accuracy** = (Recall + Specificity) / 2

For multiclass problems, TP/TN/FP/FN were aggregated across all classes before computing metrics.

---

## Results

### 5-Fold Cross-Validation

| Dataset | TP | TN | FP | FN | Recall | Specificity | Balanced Accuracy |
|---------|----|----|----|----|--------|-------------|-------------------|
| Iris | 145 | 295 | 5 | 5 | 96.7% | 98.3% | **97.5%** |
| LED7Digit | 347 | 4347 | 153 | 153 | 69.4% | 96.6% | **83.0%** |
| Heart | 65 | 107 | 43 | 55 | 54.2% | 71.3% | **62.8%** |

### Leave-One-Out

| Dataset | TP | TN | FP | FN | Recall | Specificity | Balanced Accuracy |
|---------|----|----|----|----|--------|-------------|-------------------|
| Iris | 145 | 295 | 5 | 5 | 96.7% | 98.3% | **97.5%** |
| LED7Digit | 355 | 4355 | 145 | 145 | 71.0% | 96.8% | **83.9%** |
| Heart | 67 | 117 | 33 | 53 | 55.8% | 78.0% | **66.9%** |

**Key finding:** KNN achieves near-perfect results on linearly separable data (Iris 97.5%) but degrades significantly on overlapping class distributions (Heart ~63–67%), directly reflecting the algorithm's sensitivity to feature space geometry.

---

## Project Structure

```
01-knn/
│
├── knn.py               # Full implementation: distance, voting, k-fold, LOO
├── requirements.txt
└── README.md
```

Datasets (`iris.data`, `led7digit.dat`, `heart.dat`) are not included in the repository. Download them from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/) and place them in the project root.

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/your-username/ml-portfolio.git
cd ml-portfolio/01-knn

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Add your datasets to this folder, then run
python knn.py
```

---

## Requirements

```
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
```

---

## Key Learnings

- **KNN is highly sensitive to feature space geometry.** It excels on clean, separable data like Iris (97.5% balanced accuracy) but struggles with noisy, overlapping distributions like Heart Disease (~63%), where Euclidean distance loses discriminative power.

- **5-Fold and LOO produce nearly identical results on stable datasets.** Iris and LED7Digit showed virtually no difference between both strategies, confirming robust decision boundaries. Heart Disease improved ~4 points with LOO, suggesting the model benefits from maximizing training data on harder problems.

- **When k=1 and training equals test set, accuracy is trivially 100%.** The distance from any point to itself is 0, so the classifier always votes for its own class — a clean illustration of overfitting by data leakage.

- **Small k = irregular decision boundary; large k = smoother but potentially class-eliminating boundary.** When k exceeds the number of samples in a minority class, that class can be completely voted out — a practical risk in imbalanced datasets.

---

## Next Steps

- [ ] Add feature normalization (Min-Max / Z-score) and measure its impact, especially on Heart Disease
- [ ] Experiment with k values from 1 to 30 and plot balanced accuracy vs. k
- [ ] Compare Euclidean vs. Manhattan vs. Minkowski distance metrics
- [ ] Test weighted KNN (closer neighbors vote more) to improve performance on overlapping classes
- [ ] Benchmark against scikit-learn's KNeighborsClassifier for validation

---

## Author

**Luis Angel Rojo Chavez** — [GitHub](https://github.com/arojocz)

*IPN — Centro de Investigación en Computación | Clasificación Inteligente de Patrones*
