import streamlit as st
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------- Configuración de la App ---------------------- #
st.set_page_config(page_title="Italia Keno Predictor", layout="centered")
st.title("🎯 Predicción de Italia Keno")

# ---------------------- Carga de datos ---------------------- #
@st.cache_data
def cargar_datos():
    df = pd.read_csv("sorteo_20_unificado.csv")
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
    df["hora"] = df["fecha_hora"].dt.hour * 100 + (df["fecha_hora"].dt.minute // 5) * 5
    return df

df = cargar_datos()

# ---------------------- Selección de franja horaria ---------------------- #
st.subheader("🕒 Selecciona franja horaria del sorteo")
col1, col2 = st.columns(2)
hora_inicio = col1.time_input("Desde (hora:min)", value=pd.to_datetime("16:00").time())
hora_fin = col2.time_input("Hasta (hora:min)", value=pd.to_datetime("16:55").time())

# Convertimos a códigos de hora de 5 minutos
inicio_cod = hora_inicio.hour * 100 + (hora_inicio.minute // 5) * 5
fin_cod = hora_fin.hour * 100 + (hora_fin.minute // 5) * 5

horas_objetivo = list(range(inicio_cod, fin_cod + 1, 5))
st.info(f"🎯 Predicción para sorteos entre `{inicio_cod}` y `{fin_cod}`")

# ---------------------- Entrenamiento de modelos ---------------------- #
st.subheader("🔧 Entrenando modelos por número...")

maes = {}
predicciones = {"n1": [], "n2": [], "n3": []}
X = df[["hora"]]

for i in range(1, 4):
    y = df[f"n{i}"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatBoostRegressor(verbose=0) if i <= 13 else LGBMRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    maes[f"n{i}"] = round(mae, 2)

    for h in horas_objetivo:
        pred = round(model.predict([[h]])[0])
        predicciones[f"n{i}"].append(pred)

# ---------------------- Resultados ---------------------- #
st.subheader("🔮 Predicción del próximo sorteo (media por franja)")
for i in range(1, 4):
    valores = predicciones[f"n{i}"]
    pred_media = int(round(np.mean(valores)))
    st.markdown(f"**n{i}:** 🎯 `{pred_media}` — MAE: `{maes[f'n{i}']}`")

# ---------------------- Gráfica de MAE ---------------------- #
st.subheader("📉 Error Medio Absoluto (MAE)")
fig, ax = plt.subplots()
sns.barplot(x=list(maes.keys()), y=list(maes.values()), palette="Blues_d", ax=ax)
ax.set_ylabel("MAE")
ax.set_title("Error Absoluto Medio por Número")
st.pyplot(fig)

# ---------------------- Números más frecuentes ---------------------- #
st.subheader("📊 Top 3 números más frecuentes")
valores = df[[f"n{i}" for i in range(1, 21)]].values.flatten()
numeros, cuentas = np.unique(valores, return_counts=True)
top_3 = sorted(zip(numeros, cuentas), key=lambda x: x[1], reverse=True)[:3]

for num, count in top_3:
    st.write(f"🔹 Número {num}: {count} veces")