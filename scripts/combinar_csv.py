import pandas as pd

# Cargar archivos
archivo_1 = r"C:\Users\pablo\OneDrive\Escritorio\italia_20\sorteos_000_a_099.csv"
archivo_2 = r"C:\Users\pablo\OneDrive\Escritorio\italia_20\sorteos_000_a_099_18_06.csv"
archivo_3 = r"C:\Users\pablo\OneDrive\Escritorio\italia_20\sorteos_000_a_099_nuevo.csv"

df1 = pd.read_csv(archivo_1)
df2 = pd.read_csv(archivo_2)
df3 = pd.read_csv(archivo_3)

# Unir los archivos
df_unido = pd.concat([df1, df2, df3], ignore_index=True)

# Convertir a datetime y ordenar (sin eliminar NaN)
df_unido['fecha_hora'] = pd.to_datetime(df_unido['fecha_hora'], errors='coerce')
df_unido = df_unido.sort_values(by='fecha_hora', na_position='last')

# Eliminar duplicados solo donde 'fecha_hora' NO es NaN
mask_valid = df_unido['fecha_hora'].notna()
df_unido.loc[mask_valid] = df_unido.loc[mask_valid].drop_duplicates(subset='fecha_hora')

# Guardar
df_unido.to_csv(r"C:\Users\pablo\OneDrive\Escritorio\italia_20\sorteos_unificado.csv", index=False)

print("CSV combinado y ordenado. Se conservaron registros con fecha_hora vacía.")
