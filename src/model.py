from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from src.data_preprocessing import preprocess_features


def build_pipeline(X):
    preprocessor = preprocess_features(X)

    svc_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(kernel='rbf', C=1, gamma='scale', probability=True, class_weight='balanced'))
    ])
    return svc_pipeline

