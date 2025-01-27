# model.py
import pandas as pd

# from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV

# from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn

# Load the dataset
iris = pd.read_csv("./data/iris.csv")
X = iris.drop("variety", axis=1)
y = iris["variety"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
mlflow.set_tracking_uri("http://localhost:5000")
# Train the model
# model = RandomForestClassifier()
mlflow.set_experiment("log experiments")
input_example = X_test[0:1]

model = SVC()
param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"],
}

with mlflow.start_run():
    mlflow.log_param("dataset", "iris")
    mlflow.log_param("model", "SVC")
    # Perform GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2
    )
    grid_search.fit(X_train, y_train)
    # Log the best parameters found
    best_params = grid_search.best_params_
    mlflow.log_params(best_params)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred = best_model.predict(X_test)
    accuracy_value = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy_value}")
    mlflow.sklearn.log_model(best_model, "iris-SVC-model")
    mlflow.log_metric("accuracy", accuracy_value)
    model_path = f"runs:/{mlflow.active_run().info.run_id}/iris-SVC-model"
    mlflow.register_model(model_path, "iris-SVC-model")
# Save the model
joblib.dump(model, "model.pkl")
