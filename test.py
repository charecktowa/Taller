import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets

# Cargar el conjunto de datos Iris
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Crear un gráfico de dispersión para longitud y ancho del pétalo
plt.figure(figsize=(8, 6))
for species in df["species"].unique():
    subset = df[df["species"] == species]
    plt.scatter(subset["petal length (cm)"], subset["petal width (cm)"], label=species)

plt.title("Gráfico de dispersión típico del conjunto Iris")
plt.xlabel("Longitud del pétalo (cm)")
plt.ylabel("Anchura del pétalo (cm)")
plt.legend()
plt.grid(True)
plt.show()
