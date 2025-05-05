import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# Datos de entrada (trabajo, historial) y la salida (crédito)
X = np.array([
    [0, 0],  # No tiene trabajo, no tiene buen historial
    [1, 0],  # Tiene trabajo, no tiene buen historial
    [0, 1],  # No tiene trabajo, tiene buen historial
    [1, 1],  # Tiene trabajo, tiene buen historial
])

y = np.array([0, 0, 1, 1])  # 0 = No crédito, 1 = Crédito

# Crear y entrenar el modelo
modelo = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, activation='logistic', solver='adam')
modelo.fit(X, y)

# Predecir para 3 casos nuevos
casos_prueba = np.array([[0, 0], [1, 0], [1, 1]])

# Mostrar la probabilidad de cada caso
for i, caso in enumerate(casos_prueba):
    probabilidad = modelo.predict_proba([caso])[0][1]  # La probabilidad de que sea 'Crédito'
    resultado = "Crédito" if probabilidad >= 0.5 else "No crédito"
    print(f"Caso {i + 1} -> Entrada: {caso} | Probabilidad: {probabilidad:.4f} | {resultado}")

# Graficar los datos
plt.figure(figsize=(8, 6))

# Datos de entrenamiento
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=100, label='Datos', edgecolors='black')

# Casos de prueba
plt.scatter(casos_prueba[:, 0], casos_prueba[:, 1], c='blue', marker='x', s=200, label='Casos de prueba')

# Graficar frontera de decisión
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors='black', linestyles='dashed')

# Configuración del gráfico
plt.title("Crédito según trabajo e historial")
plt.xlabel("¿Tiene trabajo? (0/1)")
plt.ylabel("¿Buen historial? (0/1)")
plt.legend()
plt.grid(True)
plt.show()
