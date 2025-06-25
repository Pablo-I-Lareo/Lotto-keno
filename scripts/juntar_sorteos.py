import os
import pandas as pd
from bs4 import BeautifulSoup

# Ruta de la carpeta con los HTML
carpeta = r"C:\Users\pablo\OneDrive\Escritorio\italia_20\htmls_filas"

# Inicializamos una lista para guardar los datos
sorteos = []

# Recorremos todos los archivos HTML de la carpeta
for nombre_archivo in sorted(os.listdir(carpeta)):
    if nombre_archivo.endswith(".html"):
        ruta_completa = os.path.join(carpeta, nombre_archivo)
        with open(ruta_completa, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            filas = soup.find_all("tr", {"role": "row"})

            for fila in filas:
                celdas = fila.find_all("td")
                if len(celdas) >= 2:
                    fecha_hora = celdas[0].text.strip()
                    numeros = [int(div.text) for div in celdas[1].find_all("div", class_="nrr")]
                    if len(numeros) == 20:
                        sorteo = {"archivo": nombre_archivo, "fecha_hora": fecha_hora}
                        for i, num in enumerate(numeros, 1):
                            sorteo[f"n{i}"] = num
                        sorteos.append(sorteo)

# Convertimos a DataFrame
df = pd.DataFrame(sorteos)

# Guardamos en CSV
output = r"C:\Users\pablo\OneDrive\Escritorio\italia_20\sorteo_20_hoy.csv"
df.to_csv(output, index=False, encoding="utf-8-sig")

print(f"✅ Guardados {len(df)} sorteos en '{output}'")
