from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer


def preprocess_features(X):
    # Define binary columns Yes/No or Male/Female
    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

    for col in binary_cols:
        X[col] = X[col].map(binary_map)

    # 2. Define complex mapping as a dictionary of column: map_dict
    multi_map = {
        'MultipleLines': {'Yes': 1, 'No': 0, 'No phone service': 2},
        'InternetService': {'DSL': 1, 'Fiber optic': 2, 'No': 0},
        'OnlineSecurity': {'No': 0, 'Yes': 1, 'No internet service': 2},
        'OnlineBackup': {'No': 0, 'Yes': 1, 'No internet service': 2},
        'DeviceProtection': {'No': 0, 'Yes': 1, 'No internet service': 2},
        'TechSupport': {'No': 0, 'Yes': 1, 'No internet service': 2},
        'StreamingTV': {'No': 0, 'Yes': 1, 'No internet service': 2},
        'StreamingMovies': {'No': 0, 'Yes': 1, 'No internet service': 2},
        'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
        'PaymentMethod': {
            'Electronic check': 0,
            'Mailed check': 1,
            'Bank transfer (automatic)': 2,
            'Credit card (automatic)': 3
        }
    }

    for col, mapper in multi_map.items():
        X[col] = X[col].map(mapper)

    categorical_cols = [x for x in X.select_dtypes(include='object').columns.tolist() if
                        x != 'Churn' and x != 'CustomerID']

    numerical_cols = [x for x in X.select_dtypes(include=['int64', 'float64']).columns.tolist() if
                      x != 'Churn' and x != 'CustomerID']

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
    ])

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler()),
    ])

    preprocessor = ColumnTransformer([
        ('categorical', categorical_pipeline, categorical_cols),
        ('numerical', numerical_pipeline, numerical_cols),
    ])

    return preprocessor