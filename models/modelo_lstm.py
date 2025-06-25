# modelo_lstm.py
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import joblib

# Cargar dataset
df = pd.read_csv("sorteos_unificado.csv")

if 'numeros' not in df.columns:
    raise ValueError("La columna 'numeros' no está en el CSV.")

df['numeros'] = df['numeros'].apply(lambda x: eval(x) if isinstance(x, str) else x)

def convertir_a_binario(lista):
    binario = np.zeros(90)
    for num in lista:
        if 1 <= num <= 90:
            binario[num - 1] = 1
    return binario

X = np.arange(len(df) - 10).reshape(-1, 1)  # secuencias de 10
y = np.array([convertir_a_binario(df['numeros'].iloc[i + 10]) for i in range(len(df) - 10)])

# Input shape: (samples, timesteps, features)
X_seq = np.array([np.array([convertir_a_binario(df['numeros'].iloc[j]]) for j in range(i, i+10)]) for i in range(len(df) - 10)])

# Escalado opcional si quieres trabajar con floats
scaler = MinMaxScaler()
X_seq = X_seq.reshape(-1, 90)
X_seq = scaler.fit_transform(X_seq).reshape(-1, 10, 90)

# Guardamos el escalador por si se quiere usar después
joblib.dump(scaler, "scaler_lstm.pkl")

# Modelo LSTM
model = Sequential()
model.add(LSTM(128, input_shape=(10, 90), return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(90, activation='sigmoid'))  # salida binaria [0-1]

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mae'])

early_stop = EarlyStopping(patience=5, restore_best_weights=True)

# Entrenamiento
model.fit(X_seq, y, epochs=30, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=1)

# Guardar modelo
model.save("modelo_lstm.keras")

# Evaluación
y_pred = model.predict(X_seq)
mae = mean_absolute_error(y, np.round(y_pred))
print(f"📉 MAE LSTM: {mae:.4f} → error medio ≈ {mae*20:.2f} números por sorteo")
