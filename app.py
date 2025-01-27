# from flask import Flask, request, jsonify
# import joblib
# import mlflow
# import numpy as np

# # Initialize the Flask app
# app = Flask(__name__)

# # Load the trained model
# model_versions = mlflow.search_registered_models(filter_string="name='iris-model'")

# print(f"model_v:{model_versions}")
# # Get the latest model version (this should not raise an error if the model exists)
# latest_version = model_versions[0].latest_versions[0].version
# # latest_version = model_versions
# print(f"models:{latest_version}")
# model_uri = f"models:/iris-model/{latest_version}"

# # Load the model
# model = mlflow.pyfunc.load_model(model_uri)
# # model = joblib.load('./mlruns/139729248635119691/e1108faf6e264368b182ea0a29061aab/artifacts/model/model.pkl')

# # Define the predict endpoint
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the input data from the request (assuming JSON format)
#     data = request.get_json()

#     # Extract the features from the incoming JSON data
#     features = np.array(data['features']).reshape(1, -1)
    
#     # Make the prediction using the model
#     prediction = model.predict(features)

#     # Return the prediction as a JSON response
#     return jsonify({"prediction": prediction[0].tolist()})  # Convert prediction to a list if necessary

# if __name__ == '__main__':
#     app.run(debug=True)

# --------------------
import mlflow
import mlflow.pyfunc
from flask import Flask, request, jsonify
from mlflow.tracking import MlflowClient
import numpy as np
import joblib

# Initialize the Flask app
app = Flask(__name__)

def load_model():
    try:
        # Set the MLflow tracking URI to the local MLflow server running on the host machine (in case of Docker)
        mlflow.set_tracking_uri("http://host.docker.internal:5000")  # Change if using a different URI
        
        # Define the model name
        model_name = "iris-model"

        # Connect to the MLflow Client
        client = MlflowClient()

        # Get the latest version of the model
        latest_model_version = client.get_latest_versions(model_name, stages=["None"])[0]  # Use the first/latest version
        print(f"Latest model version: {latest_model_version.version}")

        # Construct the model URI for MLflow registry
        model_uri = f"models:/{model_name}/{latest_model_version.version}"
        # model_uri= r'/C:/Users/roar_/Videos/MLOPS/ml-cicd-pipeline/mlruns/139729248635119691/264844a42f484e449c48db02dacf3432/artifacts/model/.'
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
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model could not be loaded. Please try again later."}), 500

    try:
        # Get the input data from the request (assuming JSON format)
        data = request.get_json()

        # Ensure the 'features' key is present in the request data
        if 'features' not in data:
            return jsonify({"error": "Missing 'features' in the input data."}), 400

        # Extract the features from the incoming JSON data
        features = np.array(data['features']).reshape(1, -1)

        # Make the prediction using the model
        prediction = model.predict(features)

        # Return the prediction as a JSON response
        return jsonify({"prediction": prediction[0]})  # Convert prediction to a list if necessary
    
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

if __name__ == '__main__':
    # Run the app, make it listen on all interfaces (for Docker or production usage)
    app.run(host='0.0.0.0', port=5001, debug=True)
# from flask import Flask, request, jsonify
# import joblib
# import mlflow
# from mlflow.tracking import MlflowClient
# import numpy as np
# import os

# # Initialize the Flask app
# app = Flask(__name__)

# def load_model():
#     try:
#         # Set the MLflow tracking URI to the local MLflow server running on the host machine (in case of Docker)
#         mlflow.set_tracking_uri("http://host.docker.internal:5000")  # Change if using a different URI
        
#         # Search for the registered model in MLflow
#         # model_versions = mlflow.search_registered_models(filter_string="name='iris-model'")

#         # if not model_versions:
#         #     raise ValueError("No registered models found in MLflow.")

#         # print(f"model_v:{model_versions}")
        
#         # # Get the latest model version
#         # latest_version = model_versions[0].latest_versions[0].version
#         # print(f"Latest model version: {latest_version}")

#         # # Construct the model URI for MLflow registry
#         # model_uri = f"models:/iris-model/{latest_version}"

#         # # Load the model using MLflow
#         # model = mlflow.pyfunc.load_model(model_uri)
#         # Define the model name
#         model_name = "iris-model"

#         # Connect to the MLflow Client
#         client = MlflowClient()

#         # Get the latest version of the model
#         latest_model_version = client.get_latest_versions(model_name)
#         print(latest_model_version)
#         # Load the model
#         model_uri = f"models:/{model_name}/{latest_model_version[0].version}"
#         model = mlflow.pyfunc.load_model('./mlruns/139729248635119691/264844a42f484e449c48db02dacf3432/artifacts/model/')
#         return model
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         return None

# # Load the model once at app startup
# model = load_model()
# print(f"model:{model}")

# # Define the predict endpoint
# @app.route('/predict', methods=['POST'])
# def predict():
#     if model is None:
#         return jsonify({"error": "Model could not be loaded. Please try again later."}), 500

#     try:
#         # Get the input data from the request (assuming JSON format)
#         data = request.get_json()

#         # Ensure the 'features' key is present in the request data
#         if 'features' not in data:
#             return jsonify({"error": "Missing 'features' in the input data."}), 400

#         # Extract the features from the incoming JSON data
#         features = np.array(data['features']).reshape(1, -1)

#         # Make the prediction using the model
#         prediction = model.predict(features)

#         # Return the prediction as a JSON response
#         return jsonify({"prediction": prediction[0]})  # Convert prediction to a list if necessary
    
#     except Exception as e:
#         return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

# if __name__ == '__main__':
#     # Run the app, make it listen on all interfaces (for Docker or production usage)
#     app.run(host='127.0.0.1', port=5001, debug=True)

# from flask import Flask, request, jsonify
# import joblib
# import mlflow
# import numpy as np
# import os

# # Initialize the Flask app
# app = Flask(__name__)

# # Load the model only once at the beginning to avoid reloading on every request
# def load_model():
#     try:
#         # Search for the registered model in MLflow
#         model_versions = mlflow.search_registered_models(filter_string="name='iris-model'")

#         if not model_versions:
#             raise ValueError("No registered models found in MLflow.")

#         print(f"model_v:{model_versions}")
        
#         # Get the latest model version
#         latest_version = model_versions[0].latest_versions[0].version
#         print(f"Latest model version: {latest_version}")

#         model_uri = f"models:/iris-model/{latest_version}"

#         # Load the model
#         model = mlflow.pyfunc.load_model(model_uri)
#         return model
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         return None

# # Load the model once at app startup
# model = load_model()

# # Define the predict endpoint
# @app.route('/predict', methods=['POST'])
# def predict():
#     if model is None:
#         return jsonify({"error": "Model could not be loaded. Please try again later."}), 500

#     try:
#         # Get the input data from the request (assuming JSON format)
#         data = request.get_json()

#         # Ensure the 'features' key is present in the request data
#         if 'features' not in data:
#             return jsonify({"error": "Missing 'features' in the input data."}), 400

#         # Extract the features from the incoming JSON data
#         features = np.array(data['features']).reshape(1, -1)

#         # Make the prediction using the model
#         prediction = model.predict(features)

#         # Return the prediction as a JSON response
#         return jsonify({"prediction": prediction[0]})  # Convert prediction to a list if necessary
    
#     except Exception as e:
#         return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

# if __name__ == '__main__':
#     # Run the app, make it listen on all interfaces (for Docker or production usage)
#     app.run(host='0.0.0.0', port=5001, debug=False)



# --------------------




# ----------------
# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import mlflow
# import numpy as np

# # Initialize the FastAPI app
# app = FastAPI()

# # Load the trained model

# model_versions = mlflow.search_registered_models(filter_string=f"name='iris-model'")

# print(f"model_v:{model_versions}")
# # Get the latest model version (this should not raise an error if the model exists)
# latest_version = model_versions[0].latest_versions[0].version
# # latest_version = model_versions
# print(f"models:{latest_version}")
# model_uri = f"models:/iris-model/{latest_version}"

#     # Load the model
# model = mlflow.pyfunc.load_model(model_uri)
# # model = joblib.load('./mlruns/139729248635119691/e1108faf6e264368b182ea0a29061aab/artifacts/model/model.pkl')

# # Define a Pydantic model to handle input data
# class PredictionRequest(BaseModel):
#     features: list[float]

# # Define the predict endpoint
# @app.post("/predict")
# def predict(request: PredictionRequest):
#     # Convert the features list into a numpy array and reshape for prediction
#     features = np.array(request.features).reshape(1, -1)
    
#     # Make the prediction using the model
#     prediction = model.predict(features)

#     # Return the prediction as a JSON response
#     return {"prediction": prediction[0]}
