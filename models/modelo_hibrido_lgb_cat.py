import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

# Cargar CSV
ruta = r"C:\Users\pablo\OneDrive\Escritorio\italia_20\sorteo_20_unificado.csv"
df = pd.read_csv(ruta)

# Columnas de los números
columnas_numeros = [f"n{i}" for i in range(1, 21)]

# Función para crear X binaria y y objetivo
def crear_dataset(df, target_col):
    X, y = [], []
    for i in range(len(df) - 1):
        fila = df.iloc[i]
        siguiente_fila = df.iloc[i + 1]
        nums = set(fila[columnas_numeros])
        X.append([(1 if n in nums else 0) for n in range(1, 91)])
        y.append(siguiente_fila[target_col])
    return np.array(X), np.array(y)

# Entrenar modelo para cada número
resultados = []

for i in range(1, 21):
    target = f"n{i}"
    X, y = crear_dataset(df, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if i <= 13:
        model = CatBoostRegressor(verbose=0)
        modelo_usado = "CatBoost"
    else:
        model = LGBMRegressor()
        modelo_usado = "LightGBM"

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    resultados.append({
        "número": target,
        "modelo": modelo_usado,
        "MAE": round(mae, 4)
    })

# Mostrar resultados
df_resultados = pd.DataFrame(resultados)
print(df_resultados)
