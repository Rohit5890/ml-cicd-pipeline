import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Test data loading
def test_load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    assert X.shape == (150, 4)  # 150 samples, 4 features
    assert len(np.unique(y)) == 3  # 3 classes in the target variable


# Test train-test split
def test_train_test_split():
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Ensure the train-test split is correct
    assert X_train.shape[0] == 120  # 120 samples in the training set
    assert X_test.shape[0] == 30  # 30 samples in the test set
    assert (
        len(np.unique(y_train)) == 3
    )  # All classes should be present in the training set
    assert len(np.unique(y_test)) == 3  # All classes should be present in the test set


# Test model training and prediction
def test_model_training():
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = SVC()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    assert (
        y_pred.shape == y_test.shape
    )  # Ensure predictions are the same shape as the test labels
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.9  # Ensure accuracy is reasonable


# Test model saving
def test_model_saving():
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = SVC()
    model.fit(X_train, y_train)

    # Save the model to a file
    joblib.dump(model, "model.pkl")

    # Check if the model file is created
    loaded_model = joblib.load("model.pkl")
    assert isinstance(loaded_model, SVC)  # Ensure the loaded model is of type SVC
