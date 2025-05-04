import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

# Datos de entrada (trabajo, historial) y salida (crédito)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([
    [0],
    [1],
    [1],
    [1]
])

# Crear modelo
modelo = Sequential()
modelo.add(Dense(units=1, input_dim=2, activation='sigmoid'))
modelo.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
modelo.fit(X, y, epochs=200, verbose=0)

# Casos de prueba (los 3 que quieres probar)
casos_prueba = np.array([
    [0, 0],
    [1, 0],
    [1, 1]
])

# Predecir casos
for i, caso in enumerate(casos_prueba, 1):
    prob = modelo.predict(np.array([caso]), verbose=0)[0][0]
    resultado = 1 if prob >= 0.5 else 0
    print(f"Caso {i} -> Entrada: {caso} | Probabilidad: {prob:.4f} | {'✅ Crédito' if resultado else '❌ No crédito'}")

# Preparar gráfica
plt.figure(figsize=(6, 6))
plt.title("Crédito según trabajo e historial")
plt.xlabel("¿Tiene trabajo? (0/1)")
plt.ylabel("¿Buen historial? (0/1)")

# Graficar datos originales
for i in range(len(X)):
    color = 'green' if y[i] == 1 else 'red'
    plt.scatter(X[i][0], X[i][1], c=color, label='Datos' if i == 0 else "")

# Graficar casos de prueba
for caso in casos_prueba:
    plt.scatter(caso[0], caso[1], c='blue', marker='x', s=100, label='Casos de prueba' if np.array_equal(caso, casos_prueba[0]) else "")

# Graficar frontera de decisión
xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
grid = np.c_[xx.ravel(), yy.ravel()]
preds = modelo.predict(grid, verbose=0).reshape(xx.shape)
plt.contour(xx, yy, preds, levels=[0.5], colors='black', linestyles='dashed', linewidths=2)

# Leyenda
plt.legend()
plt.grid(True)
plt.show()
