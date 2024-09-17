# Importar las librerías necesarias
from sklearn.datasets import load_iris  # Para cargar el conjunto de datos Iris
from sklearn.model_selection import (
    train_test_split,
)  # Para dividir el conjunto de datos
from sklearn.neighbors import KNeighborsClassifier  # El clasificador KNN
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)  # Métricas para evaluar el modelo
import matplotlib.pyplot as plt  # Para visualizar resultados (opcional)
import seaborn as sns  # Para mejorar las visualizaciones (opcional)

# Paso 1: Cargar el conjunto de datos Iris
iris = load_iris()
X = iris.data  # Características (longitud y anchura de sépalos y pétalos)
y = iris.target  # Etiquetas (especies de Iris)

# Paso 2: Dividir el conjunto de datos en entrenamiento y prueba
# Usamos 80% de los datos para entrenamiento y 20% para prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Paso 3: Crear una instancia del clasificador KNN
# Elegimos un valor de k (n_neighbors), por ejemplo, k=3
k = 3
knn = KNeighborsClassifier(n_neighbors=k)

# Paso 4: Entrenar el modelo con los datos de entrenamiento
knn.fit(X_train, y_train)

# Paso 5: Realizar predicciones en el conjunto de prueba
y_pred = knn.predict(X_test)

# Paso 6: Evaluar el rendimiento del modelo
# Calculamos la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo KNN con k={k}: {accuracy:.2f}")

# Reporte de clasificación detallado
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:")
print(cm)

# Visualizar la matriz de confusión
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
)
plt.xlabel("Etiqueta predicha")
plt.ylabel("Etiqueta verdadera")
plt.title(f"Matriz de confusión para KNN con k={k}")
plt.show()

# Paso 7: Opcional - Probar diferentes valores de k
# Evaluar el modelo para diferentes valores de k y visualizar los resultados
k_values = range(1, 26)
scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_k = knn.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred_k))

# Visualizar la precisión para diferentes valores de k
plt.figure(figsize=(10, 6))
plt.plot(k_values, scores, marker="o")
plt.title("Precisión del modelo KNN para diferentes valores de k")
plt.xlabel("Valor de k")
plt.ylabel("Precisión")
plt.xticks(k_values)
plt.grid(True)
plt.show()
