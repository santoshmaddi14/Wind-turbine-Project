import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load the dataset
df = pd.read_csv('wind_turbine_maintenance_data.csv')

# Features and target
features = ['Wind_Speed', 'Rotor_Speed', 'Power_Output', 'Temperature', 'Vibration', 'Turbine_Age', 'Maintenance_History']
target = 'Anomaly'

# Standardize the features
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Prepare the data for LSTM
X = df[features].values
y = df[target].values

# Reshape data to 3D for LSTM [samples, time steps, features]
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Save the model
model.save('wind_turbine_anomaly_detection_model.h5')
