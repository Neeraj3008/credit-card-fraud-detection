# Credit Card Fraud Detection using Machine Learning

Detecting fraudulent credit card transactions on a highly imbalanced dataset, with a focus on Recall and F1-score rather than raw accuracy.

**Author:** Neeraj Patil

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Overview

Credit card fraud is one of the biggest challenges faced by financial institutions. Fraudulent transactions are extremely rare compared to legitimate ones, which makes fraud detection a severely imbalanced classification problem.

This project builds and compares five models — from a naive baseline through to a tuned gradient boosting model — and evaluates each on metrics that actually matter for fraud: Precision, Recall, F1-score, and ROC-AUC.

---

## Problem Statement

Only 0.173% of transactions in the dataset are fraudulent. A model that predicts "not fraud" for every single transaction would score 99.83% accuracy while catching zero fraud. The goal is therefore to **maximise the number of fraudulent transactions caught (Recall) without generating an unmanageable volume of false alarms (Precision)**.

---

## Dataset

| Property | Value |
| --- | --- |
| Source | Kaggle — Credit Card Fraud Detection |
| Rows | 284,807 |
| Columns | 31 |
| Features | `Time`, `Amount`, `V1`–`V28` (PCA-transformed) |
| Target | `Class` (0 = genuine, 1 = fraud) |
| Genuine transactions | 284,315 (99.827%) |
| Fraudulent transactions | 492 (0.173%) |
| Missing values | None |
| Duplicate rows | 1,081 |

`V1`–`V28` are anonymised principal components; the original features were not released for confidentiality reasons. `Time` and `Amount` are the only two untransformed features.

---

## Project Structure

```
credit-card-fraud-detection/
├── credit_card_fraud_detection.ipynb   # Main notebook — EDA, modelling, evaluation
├── data/
│   └── creditcard.csv                  # Dataset (not committed — see Usage)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt**

```
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
jupyter
```

---

## Usage

1. Download `creditcard.csv` from Kaggle and place it in the `data/` folder.
2. Update the file path in the data-loading cell of the notebook:

   ```python
   df = pd.read_csv("data/creditcard.csv")
   ```

3. Launch the notebook and run all cells:

   ```bash
   jupyter notebook credit_card_fraud_detection.ipynb
   ```

Note that the notebook was originally written on Kaggle, so the default path points at `/kaggle/input/`.

---

## Methodology

### 1. Exploratory Data Analysis
- Class distribution plot confirming the 99.83 / 0.17 imbalance
- Distribution of transaction `Amount` (heavily right-skewed)
- Boxplot of `Amount` by class to test for a fraud–amount relationship
- Correlation heatmap and per-feature correlation with the target

### 2. Train / Test Split
An 80/20 split with `stratify=y`, so the fraud ratio is preserved in both sets.

| Split | Rows | Genuine | Fraud |
| --- | --- | --- | --- |
| Train | 227,845 | 227,451 | 394 |
| Test | 56,962 | 56,864 | 98 |

### 3. Feature Scaling
`StandardScaler` is applied for Logistic Regression, which is sensitive to feature magnitude. Tree-based models (Random Forest, XGBoost) are trained on the unscaled data since they are scale-invariant.

### 4. Models Trained

| # | Model | Type | Notes |
| --- | --- | --- | --- |
| 1 | Dummy Classifier | Baseline | `strategy="most_frequent"` |
| 2 | Logistic Regression | Linear | `max_iter=1000`, scaled input |
| 3 | Random Forest | Bagging ensemble | `n_estimators=50` |
| 4 | Random Forest + SMOTE | Bagging + oversampling | Minority class upsampled to 227,451 |
| 5 | XGBoost | Boosting ensemble | `n_estimators=100`, `lr=0.1`, `max_depth=6` |
| 6 | XGBoost (tuned) | Boosting ensemble | `GridSearchCV`, 3-fold, scored on F1 |

### 5. Hyperparameter Tuning
Grid search over `n_estimators` [100, 200], `max_depth` [3, 5, 7], and `learning_rate` [0.01, 0.1].

Best parameters: `learning_rate=0.1`, `max_depth=5`, `n_estimators=200` (CV F1 = 0.8411).

---

## Results

All metrics are on the held-out test set (56,962 transactions, 98 of them fraudulent).

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Dummy Classifier | 0.9983 | 0.0000 | 0.0000 | 0.0000 | — |
| Logistic Regression | 0.9991 | 0.8267 | 0.6327 | 0.7168 | 0.8162 |
| **Random Forest** | **0.9996** | **0.9412** | 0.8163 | **0.8743** | 0.9081 |
| Random Forest + SMOTE | 0.9994 | 0.8351 | **0.8265** | 0.8308 | **0.9131** |
| XGBoost (default) | 0.9995 | 0.9157 | 0.7755 | 0.8398 | 0.8877 |
| XGBoost (tuned) | 0.9995 | 0.9059 | 0.7857 | 0.8415 | 0.8928 |

### Fraud cases caught (out of 98)

| Model | Caught | Missed |
| --- | --- | --- |
| Dummy Classifier | 0 | 98 |
| Logistic Regression | ~62 | ~36 |
| Random Forest | ~80 | ~18 |
| Random Forest + SMOTE | ~81 | ~17 |
| XGBoost (tuned) | ~77 | ~21 |

**Final model: Random Forest**, selected for the best overall balance of Precision, Recall, and F1-score.

---

## Key Findings

**Accuracy is meaningless here.** The Dummy Classifier reached 99.83% accuracy by predicting "genuine" every time, catching zero fraud. Any model on this dataset will look excellent on accuracy alone.

**Non-linear models clearly beat the linear baseline.** Random Forest lifted Recall from 63.3% to 81.6% over Logistic Regression — roughly 18 additional frauds caught out of 98, which is a substantial difference at a bank's transaction volume.

**SMOTE traded precision for recall.** Oversampling the minority class raised Recall by about 1 percentage point (one extra fraud caught) but dropped Precision from 94.1% to 83.5%, pulling F1 down by around 4 points. Random Forest was already robust enough that synthetic samples added more false positives than genuine signal.

**More complexity did not mean better performance.** XGBoost, even after grid search, did not overtake Random Forest. Likely reasons: the search grid was small relative to XGBoost's full parameter space, the PCA-transformed features are already clean and low-noise, and the dataset is not large enough for boosting to pull ahead.

**ROC-AUC alone can mislead on imbalanced data.** Random Forest + SMOTE posted the highest ROC-AUC while having a clearly worse precision/F1 profile, which is why the final model choice was based on the full metric picture rather than a single number.

**Feature importance.** Random Forest impurity-based importances identify which of the PCA components contribute most to fraud detection, though the anonymisation means these cannot be mapped back to real-world transaction attributes.

---

## Limitations

- The 1,081 duplicate rows were identified but not removed, so a small amount of leakage between train and test is possible.
- Feature anonymisation prevents any domain-level interpretation of the importance rankings.
- The classification threshold was left at the default 0.5; no cost-sensitive threshold tuning was performed.
- Model selection used a single train/test split rather than repeated cross-validation, so the reported metrics carry some variance.

---

## Future Work

1. Explore LightGBM and CatBoost for tabular fraud detection.
2. Apply SHAP for model explainability.
3. Experiment with threshold optimisation for different business requirements.
4. Build a real-time fraud detection pipeline using streaming transaction data.
5. Investigate deep learning and transformer-based fraud detection models.

---

## Tech Stack

| Category | Tools |
| --- | --- |
| Language | Python 3.12 |
| Data handling | NumPy, Pandas |
| Visualisation | Matplotlib, Seaborn |
| Modelling | scikit-learn, XGBoost |
| Imbalance handling | imbalanced-learn (SMOTE) |
| Environment | Jupyter Notebook / Kaggle |

---


