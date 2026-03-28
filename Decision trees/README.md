# Decision Tree Classifier — Gini Impurity

> Binary decision tree implemented from scratch using Gini impurity, depth-limited to 6 levels, evaluated on three datasets with stratified Hold-Out 70/30 — achieving 86.7% balanced accuracy on Iris and 89.2% on LED7Digit.

---

## Overview

This project implements a full binary decision tree from scratch without scikit-learn's DecisionTreeClassifier. The tree splits recursively using Gini impurity, evaluating every possible threshold across all features to find the optimal split at each node. Depth is limited to 6 to control overfitting. All datasets are reduced to binary classification and evaluated using stratified Hold-Out 70/30, with results compared against the full dataset to expose overfitting behavior.

---

## Datasets

| Dataset | Instances (binary) | Classes | Split |
|---------|--------------------|---------|-------|
| Iris (versicolor vs virginica) | 100 | 2 | 70/30 stratified |
| Heart Disease | 270 | 2 | 70/30 stratified |
| LED7Digit (classes 1 & 2) | ~100 | 2 | 70/30 stratified |

---

## Results — Hold-Out 70/30

| Dataset | Balanced Accuracy | Sensitivity | Specificity |
|---------|-------------------|-------------|-------------|
| Iris | 86.7% | 80.0% | 93.3% |
| Heart Disease | 73.6% | 69.4% | 77.8% |
| LED7Digit | 89.2% | 87.5% | 90.9% |

## Results — Full Dataset (train = test)

| Dataset | Balanced Accuracy | Sensitivity | Specificity |
|---------|-------------------|-------------|-------------|
| Iris | 100.0% | 100.0% | 100.0% |
| Heart Disease | 91.7% | 90.8% | 92.7% |
| LED7Digit | 97.1% | 94.1% | 100.0% |

**Key finding:** The gap between full-dataset and hold-out results reveals classic overfitting — Heart Disease drops 18 points (91.7% → 73.6%) when evaluated on unseen data, showing the tree memorizes training patterns rather than generalizing. LED7Digit holds better (~8 point drop) because its binary features produce cleaner, more generalizable splits.

---

## Getting Started

```bash
git clone https://github.com/your-username/ml-portfolio.git
cd ml-portfolio/04-decision-tree

python -m venv venv
source venv/bin/activate

# Place iris.data, heart.dat, led7digit.dat in this folder, then:
python decision_tree.py
```

---

## Key Learnings

- **Gini impurity implemented from scratch exposes the cost of exhaustive search.** Every node evaluates all possible thresholds across all features — O(n · d · n log n) per split — which is manageable on small datasets but would be prohibitive at scale without vectorization or pruning.
- **Full-dataset evaluation is misleading by design.** Training and testing on the same data inflates metrics artificially (Heart Disease: 91.7%), making hold-out validation essential for any honest performance claim.
- **Depth limiting is a blunt but effective regularization tool.** Capping at depth 6 prevents complete memorization, but Heart Disease still overfits — suggesting that min_samples_leaf or cost-complexity pruning would produce a better-generalizing tree.

---

## Author

**Luis Angel Rojo Chavez** — [GitHub](https://github.com/arojocz)
