from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import iris plan
from sklearn.datasets import load_iris
import numpy as np

# Import ONNX Runtime
import onnxruntime as ort

app = FastAPI()


class Persona(BaseModel):
    id: int
    nombre: str
    apellido: str


base_datos = []

iris = load_iris()
etiquetas = iris.target_names.tolist()


class IrisPlant(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


knn_model_path = "knn_iris.onnx"
gnb_model_path = "gnb_iris.onnx"

onnx_knn = ort.InferenceSession(knn_model_path)
onnx_gnb = ort.InferenceSession(gnb_model_path)

input_name_knn = onnx_knn.get_inputs()[0].name
label_name_knn = onnx_knn.get_outputs()[0].name

input_name_gnb = onnx_gnb.get_inputs()[0].name
label_name_gnb = onnx_gnb.get_outputs()[0].name


@app.get("/modelo")
def read_root():
    return {"message": "¡Hola, mundo!"}


@app.get("/modelo/me")
def modelo():
    return {"mensaje": "knn"}


@app.post("/persona", response_model=Persona)
def crear_persona(persona: Persona):
    base_datos.append(persona)
    return persona


@app.get("/persona/obtener", response_model=list[Persona])
def obtener_personas():
    return base_datos


@app.get("/persona/obtener/{id}", response_model=Persona)
def obtener_persona(id: int):
    for item in base_datos:
        if item.id == id:
            return item
    raise HTTPException(404, "No encontrado")


@app.delete("/persona/borrar/{id}")
def borrar_persona(id: int):
    for index, item in enumerate(base_datos):
        if item.id == id:
            del base_datos[index]
            return {"mensaje": "Persona borrada"}
    raise HTTPException(404, "No encontrado")


@app.post("/iris")
def clasificar(iris_plant: IrisPlant, model: str = "knn"):
    input_data = np.array(
        [
            [
                iris_plant.sepal_length,
                iris_plant.sepal_width,
                iris_plant.petal_length,
                iris_plant.petal_width,
            ]
        ],
        dtype=np.float32,
    )

    if model.lower() == "knn":
        pred_onx = onnx_knn.run([label_name_knn], {input_name_knn: input_data})[0]

    elif model.lower() == "gnb":
        pred_onx = onnx_gnb.run([label_name_gnb], {input_name_gnb: input_data})[0]
    else:
        raise HTTPException(404, "Modelo no encontrado")

    class_idx = int(pred_onx[0])
    class_name = etiquetas[class_idx]

    return {"clase": class_name}
