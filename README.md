# 🧠 Loan Default Prediction ML Project

This project aims to predict whether a borrower will default on a loan based on tabular data including credit score, income, employment type, loan purpose, and more. The dataset used is the Lending Club public dataset.

---

## 🔍 Problem Statement

The objective is to build a machine learning model that can predict loan default risk based on customer and loan attributes.

---

## 📦 Dataset

**Dataset Source**: [Lending Club Loan Data](https://www.kaggle.com/wordsforthewise/lending-club)

You’ll need to download the dataset and place it in the `data/` directory as `loan_data.csv`.

---

## 🔧 ML Concepts Covered

- Handling missing values (drop + imputation)
- Encoding categorical variables (Ordinal + One-Hot)
- Feature engineering (e.g., ratios, bins)
- ML Pipelines
- Cross-validation
- Model building: XGBoost, Random Forest
- Model evaluation: Confusion Matrix, ROC AUC
- Feature Importance: SHAP, Model-based
- Data leakage detection and prevention

---

## 🗂️ Milestones

1. **EDA**
   - Distribution analysis
   - Data leakage checks

2. **Preprocessing**
   - Imputation
   - Encoding (One-hot + Ordinal)
   - Pipeline integration

3. **Modeling**
   - Training with XGBoost and Random Forest
   - Cross-validation

4. **Evaluation**
   - Confusion Matrix
   - ROC AUC
   - Precision/Recall

5. **Interpretability**
   - SHAP analysis
   - Feature importance plots

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/loan-default-prediction-ml.git
cd loan-default-prediction-ml
