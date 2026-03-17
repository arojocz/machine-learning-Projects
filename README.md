# machine-learning-Projects
# Machine Learning Portfolio

> A curated collection of supervised learning projects covering classical and modern classification algorithms, built with real-world datasets, rigorous evaluation, and production-ready code.

---

## Projects

| # | Project | Algorithm | Dataset |
|---|---------|-----------|---------|
| 01 | [K-Nearest Neighbors](#01-k-nearest-neighbors) | KNN | UCI Iris |
| 01 | [K-Nearest Neighbors](#01-k-nearest-neighbors) | KNN | UCI Iris |

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
├── 03-logistic-regression/
│   └── ...
│
├── 04-decision-tree/
│   └── ...
│
├── 05-random-forest/
│   └── ...
│
├── 06-svm/
│   └── ...
│
├── 07-gradient-boosting/
│   └── ...
│
└── 08-neural-network/
    └── ...
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
pip install -r requirements.txt

# Run
python main.py
```



## About Me

**Luis Angel Rojo Chavez**  
Machine Learning and Artificial Intelligence enthusiast focused on building interpretable, well-evaluated models.

[LinkedIn](https://www.linkedin.com/in/luis-angel-rojo-chavez/) · [GitHub](https://github.com/your-username) · [Email](arojocz@gmail.com)
