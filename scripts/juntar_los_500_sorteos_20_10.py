import os
import pandas as pd
from bs4 import BeautifulSoup
import re

# Ruta de la carpeta con los HTML
carpeta = r"C:\Users\pablo\OneDrive\Escritorio\italia_20\htmls_filas"

sorteos = []
archivos_procesados = 0

for nombre_archivo in sorted(os.listdir(carpeta)):
    if nombre_archivo.endswith(".html"):
        ruta_completa = os.path.join(carpeta, nombre_archivo)
        with open(ruta_completa, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            filas = soup.find_all("tr", {"role": "row"})

            for fila in filas:
                celdas = fila.find_all("td")
                if len(celdas) >= 2:
                    texto_fecha_hora = celdas[0].get_text(" ", strip=True)
                    match = re.search(r'(\d{2}:\d{2}).*?(\d{2}-\d{2}-\d{4})', texto_fecha_hora)
                    if not match:
                        continue
                    hora, fecha = match.groups()
                    fecha_hora = f"{fecha} {hora}"

                    numeros = [int(div.text) for div in celdas[1].find_all("div", class_="nrr")]
                    if len(numeros) == 11:
                        sorteo = {"archivo": nombre_archivo, "fecha_hora": fecha_hora}
                        for i, num in enumerate(numeros, 1):
                            sorteo[f"n{i}"] = num
                        sorteos.append(sorteo)
                        archivos_procesados += 1

if sorteos:
    df = pd.DataFrame(sorteos)
    output = r"C:\Users\pablo\OneDrive\Escritorio\italia_20\sorteos_lotto_10_20.csv"
    df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"\n✅ Guardados {len(df)} sorteos (con 11 números) en '{output}'")
else:
    print("❌ No se encontraron sorteos válidos.")
