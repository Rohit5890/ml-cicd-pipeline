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
X = iris.drop('variety', axis=1)
y = iris['variety']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
mlflow.set_tracking_uri("http://localhost:5000")
# Train the model
# model = RandomForestClassifier()
mlflow.set_experiment("log experiments")
model = SVC()
model.fit(X_train, y_train)

#     # Make predictions and evaluate the model
y_pred = model.predict(X_test)
input_example = X_test[0:1]

# Log model with input example to infer the model signature automatically

model = SVC()
# model = RandomForestClassifier()
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# # Perform GridSearchCV
# grid_search = GridSearchCV(
# estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
# grid_search.fit(X_train, y_train)

# # Get best hyperparameters and model
# best_params = grid_search.best_params_
# best_model = grid_search.best_estimator_

# print("Best hyperparameters found: ", best_params)


# model.fit(X_train, y_train)
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
    # Get the best model and its accuracy
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # Log the accuracy of the model
    # mlflow.log_metric("accuracy", accuracy)
    # Log the trained model
    # mlflow.sklearn.log_model(best_model, "model")
    # Make predictions and evaluate the model
    # Evaluate the best model on the test set
    y_pred = best_model.predict(X_test)
    accuracy_value = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy_value}")
    mlflow.sklearn.log_model(best_model, "iris-model")
    mlflow.log_metric("accuracy", accuracy_value)
    model_path = f"runs:/{mlflow.active_run().info.run_id}/iris-model"
    mlflow.register_model(model_path, "iris-model")
# Save the model
joblib.dump(model, "model.pkl")
