import mlflow
import mlflow.pyfunc
from flask import Flask, request, jsonify
from mlflow.tracking import MlflowClient
import numpy as np

# import joblib

# Initialize the Flask app
app = Flask(__name__)


def get_latest_version(model_name, client):
    return client.get_latest_versions(model_name, stages=["None"])[0]


def load_model():
    try:
        mlflow.set_tracking_uri("http://host.docker.internal:5000")
        model_name = "iris-model"
        client = MlflowClient()

        # Get the latest version of the model
        latest_model_version = get_latest_version(model_name, client)
        print(f"Latest model version: {latest_model_version.version}")

        # Construct the model URI for MLflow registry
        model_uri = f"models:/{model_name}/{latest_model_version.version}"
        # Load the model using MLflow
        model = mlflow.pyfunc.load_model(model_uri)
        # model = joblib.load('model.pkl')
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


# Load the model once at app startup
model = load_model()


# Define the predict endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model could not be loaded"}), 500

    try:
        # Get the input data from the request (assuming JSON format)
        data = request.get_json()

        # Check the 'features' key is present in the request data
        if "features" not in data:
            return jsonify({"error": "Missing 'features' in inputs."}), 400

        features = np.array(data["features"]).reshape(1, -1)

        prediction = model.predict(features)
        return jsonify({"prediction": prediction[0]})
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
