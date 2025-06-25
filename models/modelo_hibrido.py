import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

# Cargar dataset
ruta_csv = r"C:\Users\pablo\OneDrive\Escritorio\italia_20\sorteo_20_unificado.csv"
df = pd.read_csv(ruta_csv)
columnas_numeros = [f"n{i}" for i in range(1, 21)]

# Preparar dataset binario
def crear_dataset(df, target_col):
    X, y = [], []
    for i in range(len(df) - 1):
        fila = df.iloc[i]
        siguiente_fila = df.iloc[i + 1]
        nums = set(fila[columnas_numeros])
        X.append([(1 if n in nums else 0) for n in range(1, 91)])
        y.append(siguiente_fila[target_col])
    return np.array(X), np.array(y)

# Último sorteo como input
ultimo_sorteo = set(df.iloc[-1][columnas_numeros])
X_pred = np.array([[(1 if n in ultimo_sorteo else 0) for n in range(1, 91)]])

# Entrenar y predecir
predicciones = {}
for i in range(1, 21):
    target = f"n{i}"
    X, y = crear_dataset(df, target)

    if i <= 13:
        model = CatBoostRegressor(verbose=0)
    else:
        model = LGBMRegressor()

    model.fit(X, y)
    pred = model.predict(X_pred)[0]
    predicciones[target] = round(pred)

# Ordenar y mostrar predicciones
valores = list(predicciones.values())
valores_ordenados = sorted(set(round(v) for v in valores if 1 <= v <= 90))

# Guardar en CSV
df_pred = pd.DataFrame([valores_ordenados], columns=[f"pred{i+1}" for i in range(len(valores_ordenados))])
output_path = r"C:\Users\pablo\OneDrive\Escritorio\italia_20\prediccion_hibrida.csv"
df_pred.to_csv(output_path, index=False)

print("✅ Predicción del próximo sorteo:")
print(valores_ordenados)
print(f"📁 Guardado en: {output_path}")
