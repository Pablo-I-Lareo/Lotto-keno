# keno_prediction_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MultiLabelBinarizer

@st.cache_data

def load_data():
    df = pd.read_csv("C:/Users/pablo/OneDrive/Escritorio/italia_20/sorteos_lotto_10_20_sin_n11.csv")
    df["hora"] = pd.to_datetime(df["fecha_hora"], format="%d-%m-%Y %H:%M").dt.hour
    numeros_cols = [f"n{i}" for i in range(1, 11)]
    df["numeros"] = df[numeros_cols].values.tolist()
    return df


def build_model(X, y):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],)),
        Dense(64, activation='relu'),
        Dense(20, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=30, batch_size=16, verbose=0)
    return model


def predict_top_n(model, input_array, n):
    probas = model.predict(input_array, verbose=0)[0]
    return np.argsort(probas)[::-1][:n] + 1

st.set_page_config(page_title="Predicción Italia Keno", page_icon="🔢", layout="centered")
st.title("🔢 Predicción Italia Keno por Hora")
df = load_data()
mlb = MultiLabelBinarizer(classes=list(range(1, 21)))
df["bin"] = mlb.fit_transform(df["numeros"]).tolist()

st.sidebar.header("Opciones de visualización")
mostrar_frecuencias = st.sidebar.checkbox("📊 Ver estadísticas históricas por hora", value=True)

hora = st.selectbox("Selecciona la hora para analizar", sorted(df["hora"].unique()))
df_hora = df[df["hora"] == hora].copy()

if df_hora.empty:
    st.error("❌ No hay sorteos para esta hora.")
    st.stop()

if len(df_hora) < 10:
    st.warning("⚠️ Hay pocos sorteos para esta hora. Los resultados podrían no ser fiables.")
else:
    st.success(f"Se usarán {len(df_hora)} sorteos para la hora {hora}:00")

try:
    X = np.ones((len(df_hora), 1)) * hora
    y = mlb.transform(df_hora["numeros"])
except Exception as e:
    st.error(f"❌ Error al preparar los datos de entrenamiento: {e}")
    st.stop()

try:
    model = build_model(X, y)
except Exception as e:
    st.error(f"❌ Error al entrenar el modelo: {e}")
    st.stop()

input_array = np.array([[hora]])

st.subheader("🎯 Predicción de Números por Hora")
resultados = {}
for n in [1, 2, 3, 4]:
    topn = predict_top_n(model, input_array, n)
    resultados[n] = topn
    st.write(f"🔹 Top {n}: {list(topn)}")

st.subheader("📊 Evaluación histórica del Top-N")
for n in [1, 2, 3, 4]:
    aciertos = sum(
        any(num in sorteados for num in resultados[n])
        for sorteados in df_hora["numeros"]
    )
    total = len(df_hora)
    st.write(f"✅ Top {n} acertó al menos un número en {aciertos}/{total} sorteos → {aciertos/total:.2%}")

if mostrar_frecuencias:
    st.subheader("🏆 Número más frecuente entre los 10 sorteados a esa hora")
    todos = [num for sublist in df_hora["numeros"] for num in sublist]
    conteo = Counter(todos)
    mas_comun = conteo.most_common(1)[0]
    st.write(f"🔢 El número más frecuente a las {hora}:00 fue el **{mas_comun[0]}**, con {mas_comun[1]} apariciones.")

    st.subheader("📊 Top 10 números más frecuentes a esa hora")
    top10 = conteo.most_common(10)
    nums = [x[0] for x in top10]
    freqs = [x[1] for x in top10]

    fig, ax = plt.subplots()
    ax.bar(nums, freqs, color='skyblue')
    ax.set_xlabel("Número")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Top 10 números más frecuentes a las {hora}:00")
    st.pyplot(fig)
