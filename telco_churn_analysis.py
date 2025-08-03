import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from src.model import build_pipeline
from src.utils import load_data



def main():
    path = './data/Telco-Customer-Churn.csv'
    data = load_data(path)
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['Churn'] = data['Churn'].map({'No': 0, 'Yes': 1})
    telco_data = data.copy()
    X = telco_data.drop('Churn', axis=1)
    y = telco_data['Churn']
    X.drop('customerID', axis=1, inplace=True, errors='ignore')

    pipeline = build_pipeline(X)
    scores = cross_val_score(pipeline, X, y, cv=5)

    print(f"Cross val score: {scores}")
    print(f"Cross val Avg score: {scores.mean()}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    roc_auc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')
    print(f'Average ROC AUC score: {np.mean(roc_auc_scores):.3f}')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("Test ROC-AUC:", roc_auc_score(y_test, y_proba))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()


if __name__ == "__main__":
    main()