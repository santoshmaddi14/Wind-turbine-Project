import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('wind_turbine_anomaly_detection_model.h5')

# Load the dataset for scaler (for real-world application, use the scaler used during training)
df = pd.read_csv('wind_turbine_maintenance_data.csv')

# Features used for prediction
features = ['Wind_Speed', 'Rotor_Speed', 'Power_Output', 'Temperature', 'Vibration', 'Turbine_Age', 'Maintenance_History']

# Standardize the features
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

def predict_anomaly(data):
    """
    Predict anomaly for given data.
    :param data: list of feature values in the order of features list
    :return: 1 if anomaly, 0 if normal
    """
    data = np.array(data).reshape((1, 1, len(features)))
    prediction = model.predict(data)
    return int(prediction[0][0] > 0.5)

# Example usage
new_data = [12, 35, 1500, 25, 6, 10, 0]  # Example input
print(f'Anomaly: {predict_anomaly(new_data)}')

# For manual input
if __name__ == "__main__":
    wind_speed = float(input("Enter Wind Speed (m/s): "))
    rotor_speed = float(input("Enter Rotor Speed (RPM): "))
    power_output = float(input("Enter Power Output (kW): "))
    temperature = float(input("Enter Temperature (°C): "))
    vibration = float(input("Enter Vibration (mm/s): "))
    turbine_age = int(input("Enter Turbine Age (years): "))
    maintenance_history = int(input("Enter Maintenance History (1 for recent maintenance, 0 for no recent maintenance): "))

    manual_data = [wind_speed, rotor_speed, power_output, temperature, vibration, turbine_age, maintenance_history]
    print(f'Anomaly: {predict_anomaly(manual_data)}')
