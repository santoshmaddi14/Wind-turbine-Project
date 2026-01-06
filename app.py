import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from EDA import ExploratoryDataAnalysis

st.set_page_config(page_title="Wind Turbine Predictive Maintenance", layout="wide")
# Load the model
model = load_model('wind_turbine_anomaly_detection_model.h5')

# Load and scale data for feature scaling
df = pd.read_csv('wind_turbine_maintenance_data.csv')
features = ['Wind_Speed', 'Rotor_Speed', 'Power_Output', 'Temperature', 'Vibration', 'Turbine_Age',
            'Maintenance_History']
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

st.title('AI-Driven Predictive Maintenance for Wind Turbines')
st.image("coverpage.png")

# Website description
description = """


## Introduction

The renewable energy sector, particularly wind energy, is rapidly expanding as the world shifts towards sustainable energy sources. Wind turbines, the critical components of wind farms, require regular maintenance to ensure optimal performance and longevity. Traditional maintenance practices are often reactive, leading to unplanned downtimes and increased operational costs. Predictive maintenance leverages advanced technologies like artificial intelligence (AI) to foresee potential failures before they occur, thus optimizing maintenance schedules and reducing costs.

## Predictive Maintenance

Predictive maintenance involves the use of AI and machine learning algorithms to analyze data from various sensors installed on wind turbines. These sensors monitor parameters such as wind speed, rotor speed, power output, temperature, and vibration. By analyzing these parameters, AI models can predict anomalies and potential failures, allowing for timely maintenance actions.

## Benefits of AI-Driven Predictive Maintenance

1. **Reduced Downtime:** By predicting potential failures, maintenance can be scheduled proactively, minimizing unplanned downtimes.
2. **Cost Savings:** Timely maintenance prevents major failures and extends the lifespan of turbines, leading to significant cost savings.
3. **Improved Efficiency:** Regular maintenance ensures that turbines operate at optimal efficiency, maximizing power output.
4. **Enhanced Safety:** Predicting and addressing potential issues before they become critical ensures the safety of maintenance personnel and the surrounding environment.

## Our Approach

Our AI-driven predictive maintenance system utilizes a Long Short-Term Memory (LSTM) neural network to analyze historical and real-time data from wind turbines. The LSTM model is trained on a comprehensive dataset simulating various operational scenarios, including normal and anomalous conditions. By continuously monitoring turbine parameters, our system can accurately predict anomalies and recommend maintenance actions.

## Conclusion

Incorporating AI-driven predictive maintenance in wind turbine operations not only enhances efficiency and reduces costs but also contributes to the broader goal of sustainable energy. As technology advances, predictive maintenance systems will become increasingly sophisticated, providing even greater benefits to the renewable energy sector.

*Experience the future of wind turbine maintenance with our AI-driven predictive maintenance system.*
"""
st.markdown(description, unsafe_allow_html=True)

# Function to make prediction
def make_prediction(data):
    data = np.array(data).reshape((1, 1, len(features)))
    prediction = model.predict(data)
    return int(prediction[0][0] > 0.5)


# Streamlit app
def main():


    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if st.session_state['logged_in']:

        eda = ExploratoryDataAnalysis("wind_turbine_maintenance_data.csv")
        eda.run()

        st.subheader("Enter Wind Turbine Parameters")
        wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=30.0, value=12.0)
        rotor_speed = st.number_input("Rotor Speed (RPM)", min_value=0.0, max_value=100.0, value=35.0)
        power_output = st.number_input("Power Output (kW)", min_value=0.0, max_value=5000.0, value=1500.0)
        temperature = st.number_input("Temperature (°C)", min_value=-20.0, max_value=50.0, value=25.0)
        vibration = st.number_input("Vibration (mm/s)", min_value=0.0, max_value=20.0, value=6.0)
        turbine_age = st.number_input("Turbine Age (years)", min_value=0, max_value=50, value=10)
        maintenance_history = st.selectbox("Maintenance History", [0, 1])

        input_data = [wind_speed, rotor_speed, power_output, temperature, vibration, turbine_age, maintenance_history]
        scaled_data = scaler.transform([input_data])

        if st.button("Predict Anomaly"):
            prediction = make_prediction(scaled_data)
            if prediction == 1:
                st.markdown("<h1 style='color: red;'>Anomaly Detected!</h1>", unsafe_allow_html=True)
            else:
                st.markdown("<h1 style='color: green;'>No Anomaly Detected</h1>", unsafe_allow_html=True)

        st.markdown(
            "<footer style='text-align: center; padding: 10px;'><hr>© 2024 AI-Driven Predictive Maintenance for Wind Turbines. All rights reserved.</footer>",
            unsafe_allow_html=True)
    else:
        st.title("Login to Wind Turbine Predictive Maintenance")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username == "admin" and password == "password":  # Replace with proper authentication
                st.session_state['logged_in'] = True
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")


if __name__ == "__main__":
    main()
