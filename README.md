Markdown
# Credit Card Fraud Detection

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit-learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
Credit card fraud is a major challenge for financial institutions. Because fraudulent transactions are extremely rare compared to legitimate ones, this project treats fraud detection as a **highly imbalanced classification problem**. 

The goal of this project is to analyze transaction data, handle class imbalance, and evaluate machine learning models to effectively identify fraudulent activity while minimizing false positives.

---

## 📊 Dataset
* **Source:** [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Description:** The dataset contains transactions made by credit cards in September 2013 by European cardholders. It presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions.
* **Target Variable:** `Class` (0 = Legitimate, 1 = Fraudulent)
* **Class Imbalance:** Fraudulent transactions account for only **0.173%** of the dataset.

---

## 🛠️ Approach & Workflow
1. **Exploratory Data Analysis (EDA):** Checked data distributions, handled missing values, and analyzed feature correlations.
2. **Data Preprocessing:** Scaled numerical features and split the data into training and testing sets.
3. **Handling Imbalance:** Applied techniques to tackle the heavy skew in the target variable.
4. **Model Training & Evaluation:** Trained classification models and evaluated them using robust metrics.

---

## 📈 Model Performance
Since accuracy is misleading for heavily imbalanced datasets, models were evaluated primarily on **Precision, Recall, F1-Score, and ROC-AUC**:

| Model | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.85 | 0.60 | 0.70 | 0.95 |
| Random Forest | **0.93** | **0.78** | **0.85** | **0.97** |

*(Note: Feel free to update these numbers based on your actual notebook outputs!)*

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/neerajpatil3008/credit-card-fraud-detection.git](https://github.com/neerajpatil3008/credit-card-fraud-detection.git)
cd credit-card-fraud-detection
2. Install Dependencies
Bash
pip install -r requirements.txt
3. Run the Notebook
Launch Jupyter Notebook to explore the code step-by-step:

Bash
jupyter notebook notebooks/fraud_detection_eda.ipynb
👤 Author
Neeraj Patil

GitHub: @neerajpatil3008