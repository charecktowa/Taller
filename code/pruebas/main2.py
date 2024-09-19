from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
import numpy as np
import onnxruntime as rt
from sklearn.datasets import load_iris

# Load Iris dataset to get target names
iris = load_iris()
target_names = iris.target_names.tolist()

# Create FastAPI instance
app = FastAPI()

# Load ONNX models
knn_sess = rt.InferenceSession("knn_iris.onnx")
gnb_sess = rt.InferenceSession("gnb_iris.onnx")

# Get input and output names for KNN model
knn_input_name = knn_sess.get_inputs()[0].name
knn_label_name = knn_sess.get_outputs()[0].name

# Get input and output names for GaussianNB model
gnb_input_name = gnb_sess.get_inputs()[0].name
gnb_label_name = gnb_sess.get_outputs()[0].name


# Define the Model Selection Enum
class ModelName(str, Enum):
    knn = "knn"
    gaussian_nb = "gaussian_nb"


# Define the Input Data Model
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Create the Prediction Endpoint
@app.post("/predict/{model_name}")
def predict_iris(model_name: ModelName, features: IrisFeatures):
    # Convert input data to numpy array
    data = np.array(
        [
            [
                features.sepal_length,
                features.sepal_width,
                features.petal_length,
                features.petal_width,
            ]
        ],
        dtype=np.float32,
    )

    if model_name == ModelName.knn:
        # Use KNN model for prediction
        pred_onx = knn_sess.run([knn_label_name], {knn_input_name: data})[0]
    elif model_name == ModelName.gaussian_nb:
        # Use GaussianNB model for prediction
        pred_onx = gnb_sess.run([gnb_label_name], {gnb_input_name: data})[0]
    else:
        return {"error": "Invalid model selected"}

    # Get the predicted class name
    class_idx = int(pred_onx[0])
    class_name = target_names[class_idx]

    return {"model_used": model_name, "prediction": class_name}
