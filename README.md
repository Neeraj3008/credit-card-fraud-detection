# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using multiple classification models and techniques for handling highly imbalanced data.

## Overview

Credit card fraud detection is a highly imbalanced classification problem. In this dataset, fraudulent transactions represent only a very small fraction of all transactions.

This project compares different machine learning approaches and evaluates them using metrics that are more meaningful for fraud detection, particularly:

- Precision
- Recall
- F1-Score
- ROC-AUC

The project also explores **SMOTE** for handling class imbalance and **GridSearchCV** for XGBoost hyperparameter tuning.

## Dataset

The project uses the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.

### Dataset Statistics

- **Total transactions:** 284,807
- **Legitimate transactions:** 284,315
- **Fraudulent transactions:** 492
- **Fraud rate:** ~0.173%
- **Target variable:** `Class`

Where:

- `0` → Legitimate transaction
- `1` → Fraudulent transaction

The dataset contains transactions made by European cardholders during September 2013 over a period of two days.

Most features are anonymized PCA-transformed variables (`V1`–`V28`), along with `Time` and `Amount`.

## Project Workflow

The notebook follows the following workflow:

1. Dataset inspection
2. Exploratory data analysis
3. Class imbalance analysis
4. Data preprocessing
5. Stratified train-test split
6. Baseline evaluation
7. Model training
8. SMOTE experimentation
9. XGBoost hyperparameter tuning
10. Model comparison and evaluation

## Exploratory Data Analysis

The notebook performs analysis of:

- Dataset structure and statistics
- Class distribution
- Transaction amounts
- Feature correlations
- Fraudulent vs. legitimate transactions
- Feature importance
- ROC curves

The analysis highlights the severe class imbalance in the dataset and why accuracy alone is not sufficient for evaluating fraud detection models.

## Data Preprocessing

The dataset was split into training and testing sets using an **80/20 stratified split** to preserve the proportion of fraudulent transactions in both sets.

The preprocessing workflow includes:

- Checking for missing values
- Separating features and target
- Feature scaling where required
- Stratified train-test splitting

## Models

### 1. Dummy Classifier

A `DummyClassifier` was used as a baseline.

It achieved approximately **99.83% accuracy**, while detecting no fraudulent transactions.

This provides a baseline demonstrating why accuracy can be misleading for highly imbalanced datasets.

### 2. Logistic Regression

Logistic Regression was used as a baseline machine learning model.

| Metric | Score |
|---|---:|
| Accuracy | 99.91% |
| Precision | 82.67% |
| Recall | 63.27% |
| F1-Score | 71.68% |
| ROC-AUC | 81.62% |

### 3. Random Forest

A Random Forest classifier with 50 trees was trained to capture non-linear relationships in the transaction data.

| Metric | Score |
|---|---:|
| Accuracy | 99.96% |
| Precision | **94.12%** |
| Recall | 81.63% |
| F1-Score | **87.43%** |
| ROC-AUC | 90.81% |

### 4. Random Forest + SMOTE

SMOTE was applied to the training data to address the severe class imbalance.

| Metric | Score |
|---|---:|
| Accuracy | 99.94% |
| Precision | 83.51% |
| Recall | **82.65%** |
| F1-Score | 83.08% |
| ROC-AUC | **91.31%** |

SMOTE increased recall compared with the original Random Forest model, while precision and F1-score decreased.

### 5. XGBoost

XGBoost was evaluated as a boosting-based model.

Initial parameters:

```text
n_estimators = 100
learning_rate = 0.1
max_depth = 6

Results:

Metric	Score
Accuracy	99.95%
Precision	91.57%
Recall	77.55%
F1-Score	83.98%
ROC-AUC	88.77%
6. Tuned XGBoost

GridSearchCV was used to search for better XGBoost hyperparameters.

The best configuration found was:

learning_rate = 0.1
max_depth = 5
n_estimators = 200

Results:

Metric	Score
Accuracy	99.95%
Precision	90.59%
Recall	78.57%
F1-Score	84.15%
ROC-AUC	89.28%
Model Comparison
Model	Precision	Recall	F1-Score	ROC-AUC
Dummy Classifier	0.00%	0.00%	0.00%	—
Logistic Regression	82.67%	63.27%	71.68%	81.62%
Random Forest	94.12%	81.63%	87.43%	90.81%
Random Forest + SMOTE	83.51%	82.65%	83.08%	91.31%
XGBoost	91.57%	77.55%	83.98%	88.77%
Tuned XGBoost	90.59%	78.57%	84.15%	89.28%
Key Observations
The dataset is highly imbalanced, with fraud representing only ~0.173% of transactions.
The Dummy Classifier demonstrates that very high accuracy can be achieved without detecting any fraud.
Random Forest achieved a 94.12% precision and 87.43% F1-score.
Applying SMOTE increased Random Forest recall from 81.63% to 82.65%.
SMOTE also achieved the highest ROC-AUC among the evaluated models at 91.31%.
Hyperparameter tuning improved the XGBoost F1-score from 83.98% to 84.15%.
Precision, recall, F1-score and ROC-AUC provide more useful insight into fraud detection performance than accuracy alone.
Feature Analysis

Feature importance was analyzed using the Random Forest model to identify features that contributed most to its predictions.

ROC curves were also generated to compare model performance across different classification thresholds.

Technologies Used
Python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
Imbalanced-learn
XGBoost
Jupyter Notebook
Project Structure
credit-card-fraud-detection/
│
├── notebooks/
│   └── credit card fraud detection.ipynb
│
├── requirements.txt
└── README.md
How to Run
Clone the repository
git clone https://github.com/neerajpatil3008/credit-card-fraud-detection.git
cd credit-card-fraud-detection
Install dependencies
pip install -r requirements.txt
Run the notebook
jupyter notebook

Open:

credit card fraud detection.ipynb

The notebook contains the complete data analysis, preprocessing, model training, SMOTE experimentation, hyperparameter tuning, and evaluation workflow.

Future Improvements
Experiment with additional ensemble models
Optimize classification thresholds based on fraud detection requirements
Add SHAP-based model explainability
Evaluate the models on newer transaction datasets
Develop a real-time fraud detection pipeline
Author

Neeraj Patil

GitHub:
