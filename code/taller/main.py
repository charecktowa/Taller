from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


class Persona(BaseModel):
    id: int
    nombre: str
    apellido: str


base_datos = []


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
