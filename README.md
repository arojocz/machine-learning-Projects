# machine-learning-Projects
# Machine Learning Portfolio

> A curated collection of supervised learning projects covering classical and modern classification algorithms, built with real-world datasets, rigorous evaluation, and production-ready code.

---

## Projects

| # | Project | Algorithm | Dataset | Accuracy (best) |
|---|---------|-----------|---------|-----------------|
| 01 | [K-Nearest Neighbors](#01-k-nearest-neighbors) | KNN | UCI Iris |
| 02 | [Multiclass Evaluation — KNN vs GaussianNB](#02-multiclass-evaluation--knn-vs-gaussian-naive-bayes) | KNN / GaussianNB | UCI Wine | 97.8% (GNB) |
| 03 | [Gaussian Naïve Bayes](#03-gaussian-naïve-bayes) | GaussianNB | Iris / Heart / LED7Digit | 96.5% (Iris LOO) |
| 04 | [Decision Tree — Gini Impurity](#04-decision-tree--gini-impurity) | Decision Tree | Iris / Heart / LED7Digit | 89.2% (LED7Digit) |
| 05 | [PCA + Multi-Classifier Benchmark](#05-pca--multi-classifier-benchmark) | KNN / GNB / DT / RF / AdaBoost | Iris / Heart / LED7Digit | 97.8% (GNB Iris orig.) |

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![pandas](https://img.shields.io/badge/pandas-2.0-green)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-red)

- **Language:** Python 3.10+
- **Core libraries:** scikit-learn, pandas, NumPy, Matplotlib, Seaborn
- **Boosting:** XGBoost, LightGBM
- **Deep learning:** TensorFlow / PyTorch (project 08)
- **Environment:** Jupyter / Google Colab / local venv

---

## Repository Structure

```
ml-portfolio/
│
├── README.md 
├── .gitignore
│
├── 01-KNN/
│   ├── knn-classifier.pdf
│   ├── knn.py
│   ├── knn2.py
│   ├── led7digit.dat
│   ├── iris.data
│   └── README.md
│
├── 02-Multiclass Evaluation Metrics/
│   ├── multiclass_cm.py
│   ├── multiclass_eval_metrics.txt
│   └── README.md
│
├── 03-gaussian-naive-bayes/
│   └── gaussian_naive_bayes.pdf
│   └── heart.dat
│   └── iris.dat
│   └── led7digit.dat
│   └── multiclass_cm.py
│   └── README.md
│
├── 04-decision-tree/
│   └── decision_tree.pdf
│   └── decision_tree.py
│   └── heart.dat
│   └── iris.dat
│   └── led7digit.dat
│   └── README.md
│
├── 05-PCA/
│   └── ...

```

---

### 01 K-Nearest Neighbors
**Problem:** Multiclass and binary pattern classification across three benchmark datasets (Iris, LED7Digit, Heart Disease).  
**Approach:** KNN implemented from scratch using Euclidean distance with custom class-weighted confusion matrix aggregation. Evaluated with 5-Fold Cross-Validation and Leave-One-Out, fixed k=5.  
**Key finding:** Strong on separable data (Iris 97.5% balanced accuracy) but drops to 62.8% on Heart Disease — performance is dictated entirely by feature space geometry, not model complexity.

---

### 02 Multiclass Evaluation — KNN vs Gaussian Naive Bayes
**Problem:** Wine cultivar classification (3 classes) with full per-class metric reporting.  
**Approach:** Head-to-head benchmark of KNN (k=3) vs GaussianNB using Leave-One-Out, with per-class confusion matrices and Macro, Weighted, and Micro averaging.  
**Key finding:** GaussianNB achieves 97.8% macro F1 vs 71.5% for KNN — the gap is almost entirely explained by the absence of feature normalization, which distorts Euclidean distance on Wine's mixed-scale features.

---

### 03 Gaussian Naïve Bayes
**Problem:** Multiclass and binary classification across Iris, Heart Disease, and LED7Digit.  
**Approach:** GNB implemented from scratch using log-likelihood with Gaussian PDF. Evaluated with LOO and 10-Fold CV, reporting per-class Sensitivity, Specificity, and Balanced Accuracy.  
**Key finding:** Strong on simple distributions (Iris 96.5%) but sensitivity drops to 69% on LED7Digit's 10-class problem — the conditional independence assumption weakens as class overlap increases.

---

### 04 Decision Tree — Gini Impurity
**Problem:** Binary classification on three datasets (Iris versicolor vs virginica, Heart Disease, LED7Digit).  
**Approach:** Decision tree implemented from scratch using Gini impurity with exhaustive threshold search, depth-limited to 6. Evaluated with stratified Hold-Out 70/30 and compared against full-dataset results.  
**Key finding:** Heart Disease drops 18 points between full-dataset (91.7%) and hold-out (73.6%) evaluation — a textbook overfitting case showing the tree memorizes training patterns. Depth limiting alone is insufficient; pruning strategies would help.

---

### 05 PCA + Multi-Classifier Benchmark
**Problem:** Evaluate whether PCA-based dimensionality reduction improves classification across three datasets and five algorithms.  
**Approach:** Features standardized, reduced to minimum components explaining ≥90% variance (2–10 components), then benchmarked with KNN, GaussianNB, DecisionTree, RandomForest, and AdaBoost using stratified Hold-Out 70/30.  
**Key finding:** PCA helps when features are correlated — GaussianNB on LED7Digit jumps from 65.5% to 74.2% — but consistently hurts when the original feature space already provides clean class separation, as with GaussianNB on Iris (97.8% → 82.2%).

---
## How to Run Any Project

```bash
# Clone the repo
git clone https://github.com/arojocz/machine-learning-Projects
cd ml-portfolio

# Go into any project
cd 01-knn

# Create environment and install dependencies
python -m venv venv
source venv/bin/activate

# Run
python main.py
```



## About Me

**Luis Angel Rojo Chavez**  
Machine Learning and Artificial Intelligence enthusiast focused on building interpretable, well-evaluated models.

[LinkedIn](https://www.linkedin.com/in/luis-angel-rojo-chavez/) · [GitHub](https://github.com/arojocz) · [Email](arojocz@gmail.com)
