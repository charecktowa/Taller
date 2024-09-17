from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import onnxruntime as rt
from sklearn.datasets import load_iris

# Cargar el conjunto de datos Iris para obtener los nombres de las clases
iris = load_iris()
target_names = iris.target_names.tolist()

# Crear la instancia de la aplicación FastAPI
app = FastAPI()

# Cargar el modelo ONNX al iniciar la aplicación
sess = rt.InferenceSession("knn_iris.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name


# Definir el modelo de datos de entrada
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.post("/predited")
def predict(
    sepal_width: float, sepal_length: float, petal_length: float, petal_width: float
) -> str:
    # Convertir los datos de entrada a un arreglo numpy
    data = np.array(
        [[sepal_length, sepal_width, petal_length, petal_width]], dtype=np.float32
    )

    # Realizar la predicción usando el modelo ONNX
    pred_onx = sess.run([label_name], {input_name: data})[0]

    # Obtener el nombre de la clase predicha
    class_idx = int(pred_onx[0])
    class_name = target_names[class_idx]

    # Devolver la predicción como respuesta JSON
    return class_name


# Definir el endpoint para realizar predicciones
@app.post("/predict")
def predict_iris(features: IrisFeatures):
    # Convertir los datos de entrada a un arreglo numpy
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

    # Realizar la predicción usando el modelo ONNX
    pred_onx = sess.run([label_name], {input_name: data})[0]

    # Obtener el nombre de la clase predicha
    class_idx = int(pred_onx[0])
    class_name = target_names[class_idx]

    # Devolver la predicción como respuesta JSON
    return {"prediction": class_name}
