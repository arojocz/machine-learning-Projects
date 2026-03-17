# machine-learning-Projects
# Machine Learning Portfolio

> A curated collection of supervised learning projects covering classical and modern classification algorithms, built with real-world datasets, rigorous evaluation, and production-ready code.

---

## Projects

| # | Project | Algorithm | Dataset | Accuracy | Status |
|---|---------|-----------|---------|----------|--------|
| 01 | [K-Nearest Neighbors](#01-k-nearest-neighbors) | KNN | UCI Iris | 96.7% | ✅ Complete |
| 02 | [Naive Bayes Classifier](#02-naive-bayes-classifier) | Gaussian NB | SMS Spam | 98.1% | ✅ Complete |
| 03 | [Logistic Regression](#03-logistic-regression) | LogReg | Titanic | 82.3% | ✅ Complete |
| 04 | [Decision Tree](#04-decision-tree) | CART | Heart Disease | 87.5% | ✅ Complete |
| 05 | [Random Forest](#05-random-forest) | Ensemble | Credit Risk | 91.2% | ✅ Complete |
| 06 | [Support Vector Machine](#06-support-vector-machine) | SVM / SVC | Breast Cancer | 97.4% | ✅ Complete |
| 07 | [Gradient Boosting](#07-gradient-boosting) | XGBoost | Customer Churn | 93.8% | ✅ Complete |
| 08 | [Neural Network](#08-neural-network) | MLP | MNIST Digits | 98.6% | ✅ Complete |

> Update this table as you add projects. Replace accuracy values with your real results.

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
├── README.md                    ← You are here
├── .gitignore
│
├── 01-knn/
│   ├── knn.py
│   ├── requirements.txt
│   └── README.md
│
├── 02-naive-bayes/
│   ├── naive_bayes.py
│   ├── requirements.txt
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

## Project Summaries

### 01 K-Nearest Neighbors
**Problem:** Multiclass flower species classification.  
**Approach:** Explored the effect of K neighbours and distance metrics (Euclidean) on performance. Final model uses K=5 with standardized features.  
**Key finding:** 

---

### 02 Naive Bayes Classifier
**Problem:** Binary spam detection on SMS messages.  
**Approach:** Applied TF-IDF vectorization with Multinomial Naive Bayes. Compared Gaussian, Bernoulli, and Multinomial variants.  
**Key finding:** Multinomial NB outperformed the others due to the frequency-based nature of text features.

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
