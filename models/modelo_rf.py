# modelo_rf.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
import joblib

# Cargar dataset
df = pd.read_csv("sorteos_unificado.csv")

# Asegúrate de que exista la columna 'numeros'
if 'numeros' not in df.columns:
    raise ValueError("La columna 'numeros' no está en el CSV. Verifica el archivo.")

# Convertir la columna 'numeros' a lista de enteros
df['numeros'] = df['numeros'].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Función para binarizar la salida (90 posiciones → 1 si ha salido, 0 si no)
def convertir_a_binario(lista):
    binario = np.zeros(90)
    for num in lista:
        if 1 <= num <= 90:
            binario[num - 1] = 1
    return binario

# Preparar X e y
X = np.arange(len(df) - 1).reshape(-1, 1)
y = np.array([convertir_a_binario(l) for l in df['numeros'].values[1:]])

# Ajustar longitudes
X = X[:len(y)]

# Entrenar modelo
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X, y)

# Guardar modelo
joblib.dump(rf, "modelo_random_forest.pkl")

# Predecir y evaluar MAE
y_pred = rf.predict(X)
mae = mean_absolute_error(y, y_pred)
print(f"📉 MAE Random Forest: {mae:.4f} → error medio ≈ {mae*20:.2f} números por sorteo")
