import streamlit as st
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Italia Keno - Modelos ML", layout="centered")
st.title("🎯 Italia Keno - Comparador de Modelos")

@st.cache_data
def cargar_datos():
    df = pd.read_csv("sorteo_20_limpio.csv")
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
    df["hora"] = df["fecha_hora"].dt.hour * 100 + (df["fecha_hora"].dt.minute // 5) * 5
    return df

df = cargar_datos()

# Elección de modelo
modo = st.selectbox("🔍 Selecciona tipo de modelo", ["Clasificación por número", "Regresión por número"])

# Selección de franja horaria visible
st.subheader("🕒 Selecciona franja horaria")
col1, col2 = st.columns(2)
hora_inicio = col1.time_input("Desde", value=pd.to_datetime("16:00").time())
hora_fin = col2.time_input("Hasta", value=pd.to_datetime("16:55").time())

inicio_cod = hora_inicio.hour * 100 + (hora_inicio.minute // 5) * 5
fin_cod = hora_fin.hour * 100 + (hora_fin.minute // 5) * 5
horas_objetivo = list(range(inicio_cod, fin_cod + 1, 5))

if not horas_objetivo:
    st.error("⚠️ Franja horaria inválida. Asegúrate de que la hora final sea posterior a la hora de inicio.")
else:
    if modo == "Clasificación por número":
        st.subheader("🔢 Top 10 números más predecibles por hora")
        clasificacion_resultados = []

        for numero in range(1, 91):
            y = df[[f"n{i}" for i in range(1, 21)]].apply(lambda row: int(numero in row.values), axis=1)
            X = df[["hora"]]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            model = LGBMClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            proba = [model.predict_proba([[h]])[0][1] for h in horas_objetivo]
            media_proba = np.mean(proba)

            clasificacion_resultados.append((numero, acc, media_proba))

        top_10 = sorted(clasificacion_resultados, key=lambda x: x[2], reverse=True)[:10]
        df_top10 = pd.DataFrame([(n, a, p) for n, a, p in top_10], columns=["Número", "Precisión", "Probabilidad"])
        st.dataframe(df_top10)

        st.subheader("📊 Gráfico de Probabilidad")
        fig, ax = plt.subplots()
        sns.barplot(x=df_top10["Número"].astype(str), y=df_top10["Probabilidad"], ax=ax)
        ax.set_ylabel("Probabilidad media en franja")
        st.pyplot(fig)

        st.success(f"🔮 Número más probable (según modelo): {df_top10.iloc[0]['Número']}")

    else:
        st.subheader("📌 Modelo de Regresión basado en todos los números (n1 a n20)")

        # Expandir dataset para entrenar sobre todos los números
        registros = []
        for i, row in df.iterrows():
            for j in range(1, 21):
                registros.append({"hora": row["hora"], "numero": row[f"n{j}"]})
        df_reg = pd.DataFrame(registros)

        X = df_reg[["hora"]]
        y = df_reg["numero"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model_reg = LGBMRegressor()
        model_reg.fit(X_train, y_train)
        y_pred = model_reg.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        predicciones = [int(round(model_reg.predict([[h]])[0])) for h in horas_objetivo if h is not None]

        if predicciones:
            prediccion_final = int(pd.Series(predicciones).mode()[0])
            st.metric("MAE (Error Medio Absoluto)", f"{mae:.2f}")
            st.success(f"🔮 Número más probable para franja {inicio_cod}–{fin_cod}: {prediccion_final}")

            st.subheader("📊 Frecuencia de predicciones por hora")
            conteo = pd.Series(predicciones).value_counts().sort_values(ascending=False)
            fig2, ax2 = plt.subplots()
            sns.barplot(x=conteo.index.astype(str), y=conteo.values, ax=ax2)
            ax2.set_xlabel("Número predicho")
            ax2.set_ylabel("Frecuencia")
            st.pyplot(fig2)
        else:
            st.warning("⚠️ No se pudieron generar predicciones para la franja horaria seleccionada.")
