import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping

# Cargar datos
df = pd.read_csv("sorteo_20.csv")

# Asegurar que las columnas de números están ordenadas
num_cols = [f"n{i}" for i in range(1, 21)]

# Verificar que existen
for col in num_cols:
    if col not in df.columns:
        raise ValueError(f"Falta la columna {col}")

# Crear secuencias: 10 sorteos anteriores → predicción del siguiente
X_seq = []
y_seq = []

for i in range(len(df) - 10):
    entrada = df[num_cols].iloc[i:i+10].values  # (10, 20)
    objetivo = df[num_cols].iloc[i+10].values   # (20,)
    X_seq.append(entrada)
    y_seq.append(objetivo)

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# ---------------------- LSTM ---------------------- #
model = Sequential()
model.add(LSTM(64, input_shape=(10, 20), return_sequences=False))
model.add(Dense(20, activation='linear'))

model.compile(optimizer='adam', loss='mae', metrics=['mae'])

early_stop = EarlyStopping(monitor="val_loss", patience=3)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=16, callbacks=[early_stop], verbose=1)

pred_lstm = model.predict(X_test)
mae_lstm = mean_absolute_error(y_test, pred_lstm)
print(f"📉 MAE modelo LSTM: {mae_lstm:.4f} → error ≈ {mae_lstm*20:.2f} números por sorteo")

# ------------------ Random Forest ------------------ #
# Aplanar las secuencias (10 sorteos x 20 números = 200 features)
X_rf = X_seq.reshape((X_seq.shape[0], -1))

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_seq, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_rf, y_train_rf)

pred_rf = rf.predict(X_test_rf)
mae_rf = mean_absolute_error(y_test_rf, pred_rf)
print(f"🌲 MAE modelo RandomForest: {mae_rf:.4f} → error ≈ {mae_rf*20:.2f} números por sorteo")
