import streamlit as st
import pandas as pd
import numpy as np
import os
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------- Configuración de la App ---------------------- #
st.set_page_config(page_title="Italia Keno Predictor", layout="centered")
st.title("🎯 Predicción de Italia Keno")

# ---------------------- Carga de datos ---------------------- #
@st.cache_data
def cargar_datos():
    ruta = os.path.join("data", "sorteo_20_unificado.csv")
    df = pd.read_csv(ruta)
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
    df["hora"] = df["fecha_hora"].dt.hour * 100 + (df["fecha_hora"].dt.minute // 5) * 5
    return df

df = cargar_datos()

# ---------------------- Franja horaria ---------------------- #
st.subheader("🕒 Selecciona franja horaria del sorteo")
col1, col2 = st.columns(2)
hora_inicio = col1.time_input("Desde", value=pd.to_datetime("16:00").time())
hora_fin = col2.time_input("Hasta", value=pd.to_datetime("16:55").time())

inicio_cod = hora_inicio.hour * 100 + (hora_inicio.minute // 5) * 5
fin_cod = hora_fin.hour * 100 + (hora_fin.minute // 5) * 5
horas_objetivo = list(range(inicio_cod, fin_cod + 1, 5))
st.info(f"🎯 Predicción para sorteos entre `{inicio_cod}` y `{fin_cod}`")

# ---------------------- Preparación de datos binarios ---------------------- #
registros = []
for i in range(len(df)):
    hora = df.loc[i, "hora"]
    numeros = df.loc[i, [f"n{j}" for j in range(1, 21)]].values
    for n in range(1, 91):
        registros.append({"numero": n, "hora": hora, "salio": int(n in numeros)})

df_bin = pd.DataFrame(registros)

# ---------------------- Entrenamiento del modelo probabilístico ---------------------- #
st.subheader("🔧 Entrenando modelo de probabilidad...")
X = df_bin[["numero", "hora"]]
y = df_bin["salio"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LGBMClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# ---------------------- Predicción por franja ---------------------- #
predicciones = {}
for n in range(1, 91):
    proba = []
    for h in horas_objetivo:
        prob = model.predict_proba([[n, h]])[0][1]
        proba.append(prob)
    predicciones[n] = np.mean(proba)

# ---------------------- Resultados: Top 3 ---------------------- #
top_3_pred = sorted(predicciones.items(), key=lambda x: x[1], reverse=True)[:3]

st.subheader("🔮 Top 3 números más probables")
for n, prob in top_3_pred:
    st.markdown(f"🔢 Número `{n}` — Probabilidad estimada: `{prob:.2%}`")

st.subheader("📈 Precisión del modelo")
st.metric(label="Accuracy", value=f"{acc:.2%}")

# ---------------------- Gráfico Top 10 ---------------------- #
st.subheader("📊 Top 10 números más probables")
top_10 = sorted(predicciones.items(), key=lambda x: x[1], reverse=True)[:10]

fig1, ax1 = plt.subplots()
sns.barplot(x=[f"{n}" for n, _ in top_10], y=[p for _, p in top_10], ax=ax1)
ax1.set_ylabel("Probabilidad de salir")
ax1.set_xlabel("Número")
ax1.set_title("Top 10 Números con Mayor Probabilidad")
st.pyplot(fig1)

# ---------------------- Gráfico completo (opcional) ---------------------- #
with st.expander("📉 Ver probabilidad para todos los números (1-90)"):
    fig2, ax2 = plt.subplots(figsize=(18, 6))
    sns.barplot(x=list(predicciones.keys()), y=list(predicciones.values()), ax=ax2)
    ax2.set_xlabel("Número")
    ax2.set_ylabel("Probabilidad de salir")
    ax2.set_title("Probabilidad estimada por número (media en franja)")
    plt.xticks(rotation=90)
    st.pyplot(fig2)

# ---------------------- Sidebar: activar predicción única ---------------------- #
st.sidebar.header("🧠 Opciones del modelo")
activar_prediccion_unica = st.sidebar.checkbox("Mostrar predicción de número único", value=True)

# ---------------------- MODELO DE PREDICCIÓN ÚNICO REVISADO ---------------------- #
if activar_prediccion_unica:
    st.header("🎯 Predicción de un único número por el modelo")

    # Crear un registro por cada número observado
    registros_regresion = []
    for i in range(len(df)):
        hora = df.loc[i, "hora"]
        numeros = df.loc[i, [f"n{j}" for j in range(1, 21)]].values
        for num in numeros:
            registros_regresion.append({"hora": hora, "numero": num})

    df_uni = pd.DataFrame(registros_regresion)

    X_uni = df_uni[["hora"]]
    y_uni = df_uni["numero"]

    X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(X_uni, y_uni, test_size=0.2, random_state=42)

    modelo_unico = LGBMRegressor()
    modelo_unico.fit(X_train_u, y_train_u)

    y_pred_u = modelo_unico.predict(X_test_u)
    mae_unico = mean_absolute_error(y_test_u, y_pred_u)

    # Predicciones sobre la franja
    predicciones_franja = [round(modelo_unico.predict([[h]])[0]) for h in horas_objetivo]
    numero_final = int(pd.Series(predicciones_franja).mode()[0])

    st.success(f"📌 El número único predicho por el modelo es: **{numero_final}**")
    st.caption(f"Basado en {len(horas_objetivo)} predicciones por hora en la franja seleccionada.")
    st.metric(label="MAE del modelo", value=f"{mae_unico:.2f}")

    # Distribución de predicciones
    st.subheader("📊 Distribución de predicciones individuales")
    conteo_pred = pd.Series(predicciones_franja).value_counts().sort_values(ascending=False)

    fig_pred, ax_pred = plt.subplots()
    sns.barplot(x=conteo_pred.index, y=conteo_pred.values, ax=ax_pred)
    ax_pred.set_xlabel("Número predicho")
    ax_pred.set_ylabel("Frecuencia")
    ax_pred.set_title("Frecuencia por número en la franja horaria")
    st.pyplot(fig_pred)
