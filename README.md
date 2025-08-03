# 📊 Telco Customer Segmentation & Churn Prediction with SVC

A complete machine learning project using the Telco Customer Churn dataset to perform **customer segmentation and churn prediction** using **Support Vector Classifier (SVC)** with a fully leak-proof pipeline and explainable ML techniques.

---

## 🚀 Project Goals

- Segment customers using churn behavior and feature patterns.
- Predict churn using Support Vector Classifier (SVC).
- Build a **leak-proof ML pipeline** using `Pipeline` and `ColumnTransformer`.
- Perform rigorous **EDA**, **feature engineering**, and **evaluation**.
- Use **SHAP** and **Permutation Importance** for explainability.

---

## 📂 Dataset

- **Source**: [Telco Customer Churn Dataset - Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)
- Contains demographic info, service usage, and customer status (churned or not).

---

## ⚙️ Workflow Overview

### 📥 1. Load & Clean Data
- Removed white spaces and converted total charges to numeric.
- Missing values imputed smartly (e.g., `TotalCharges` imputed using `MonthlyCharges * tenure`).

### 📊 2. Exploratory Data Analysis (EDA)
- Visualized churn distribution, service usage patterns.
- Checked correlation of numerical features.
- Analyzed churn vs categorical features.

### 🏗️ 3. Feature Engineering
- **Target Encoding** for categorical variables (e.g., `Contract`, `InternetService`).
- **MinMaxScaler** for numerical features.
- Handled binary variables as 0/1.

### 🔁 4. Pipeline
- Built using `Pipeline` + `ColumnTransformer`.
- Ensures all preprocessing (encoding, scaling) is done inside the pipeline without leaking target labels.

### 📚 5. Model: Support Vector Classifier (SVC)
- Tuned `C`, `kernel`, and `gamma` using GridSearchCV.
- Evaluated using ROC-AUC and Confusion Matrix.

### 🧪 6. Evaluation
- **Cross-Validation** (StratifiedKFold with 5 splits)
- **ROC-AUC**, Accuracy, F1-Score
- Plotted Confusion Matrix and ROC Curve.

### 🎯 7. Explainability
- Used **Permutation Importance** (via `eli5`) to detect key features.
- **SHAP** to visualize feature impacts on predictions.

### 🔒 8. Avoiding Data Leakage
- No target information used outside pipeline.
- Encoding, scaling, and feature selection all inside cross-validation loop.

---

## 📈 Results Summary

| Metric           | Value     |
|------------------|-----------|
| ROC-AUC Score    | 0.84      |
| Accuracy         | 0.79      |
| F1-Score         | 0.74      |

- Top features identified via SHAP:
  - `Contract`, `MonthlyCharges`, `InternetService`, `Tenure`

---

## 🛠 Installation

```bash
git clone https://github.com/yourusername/telco-churn-segmentation-svc-pipeline.git
cd telco-churn-segmentation-svc-pipeline
pip install -r requirements.txt

