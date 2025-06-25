# Lotto
# 🎯 Italia Keno Predictor

Este repositorio contiene el desarrollo completo de una aplicación de predicción para el juego Italia Keno, que puedes consultar online desde:

🔗 [https://lottokeno.streamlit.app/](https://lottokeno.streamlit.app/)

---

## 📚 Descripción del proyecto

El objetivo principal de este proyecto es automatizar el scraping de los sorteos de Italia Keno y construir un modelo de machine learning híbrido que sea capaz de predecir los números más probables del siguiente sorteo, filtrando incluso por hora del día.

---

## 🧰 Tecnologías utilizadas

* **Python 3.10+**
* **Selenium** para scraping web
* **Pandas / NumPy** para tratamiento de datos
* **LightGBM** y **CatBoost** para modelos predictivos
* **Scikit-learn** para métricas y divisiones
* **Streamlit** para la interfaz web

---

## 🚀 Proceso completo

### 1. ✍️ Scraping de datos

Se utilizó Selenium para automatizar la navegación por la página:

* `https://lotostats.ro/toate-rezultatele-italia-keno-10e-20-90`
* Extracción de los resultados de los sorteos
* Almacenamiento en archivos `.csv`

### 2. 📊 Limpieza y unificación

* Conversión de fechas y horas a formatos legibles
* Unificación de sorteos diarios en un solo dataset: `sorteo_20_unificado.csv`
* Normalización de campos y eliminación de duplicados

### 3. 🧲 Análisis exploratorio

* Análisis de frecuencias por número
* Identificación de horas más activas para ciertos números
* MAE (Mean Absolute Error) por modelo y por número

### 4. 🤖 Modelo híbrido

Se entrena un modelo por cada número:

* Del **n1 al n13** se usa **CatBoostRegressor** (mejor MAE encontrado)
* Del **n14 al n20** se usa **LightGBMRegressor**

Para cada modelo:

* Se utiliza `hora` como feature principal
* Se calcula el MAE sobre test set
* Se predice el número para una hora determinada

### 5. 📅 App en Streamlit

Interfaz accesible desde [https://lottokeno.streamlit.app/](https://lottokeno.streamlit.app/):

* Permite seleccionar la hora deseada
* Muestra predicción de los 3 primeros números del sorteo siguiente
* Informa del MAE por modelo
* Muestra el top 3 de números históricamente más frecuentes

---

## 🚪 Estructura de carpetas destacadas

```bash
italia_20/
├── keno_app.py                  # App principal en Streamlit
├── modelo_hibrido.py           # Entrenamiento modelo mixto CatBoost/LightGBM
├── scrapear_muchas_paginas.py # Scraper con Selenium
├── sorteo_20_unificado.csv     # Dataset final
└── README.md
```

---

## 🚀 Próximos pasos

* Ampliar features: temperatura, festivos, etc.
* Guardar modelos y reutilizar sin reentrenar
* Añadir exportación de predicciones
* Mejora de interfaz visual

---

## 🌟 Autor

Pablo Iglesias Lareo

Bootcamp de Data Science y Machine Learning · 2025
