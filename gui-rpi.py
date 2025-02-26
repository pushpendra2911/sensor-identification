import pandas as pd
import streamlit as st
import requests
import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Set page configuration first (must be the first Streamlit command)
st.set_page_config(layout="wide", page_title="Building Sensor Prediction Dashboard")

# Custom CSS for styling and single-page layout
st.markdown("""
    <style>
    .main {
        background-color: #f0f8ff; /* Light blue background for tech feel */
        padding: 5px; /* Reduced padding for compactness */
        max-height: 100vh; /* Limit height to viewport */
        overflow: hidden; /* Prevent scrolling */
    }
    .header {
        text-align: center;
        color: #1e90ff; /* Dodger blue for title */
        font-size: 24px; /* Reduced for compactness */
        font-family: 'Arial Black', sans-serif;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 2px; /* Reduced margin */
    }
    .prediction {
        background-color: #e6f3ff; /* Light blue for predictions */
        padding: 5px; /* Reduced padding */
        border-radius: 3px;
        margin: 2px 0;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        font-size: 14px; /* Smaller text for compactness */
    }
    .stButton>button {
        background-color: #32cd32; /* Lime green for buttons */
        color: white;
        border: none;
        padding: 6px 12px; /* Reduced padding */
        border-radius: 2px;
        font-size: 12px; /* Smaller text */
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #228b22; /* Forest green on hover */
    }
    .stSlider > div > div > div {
        padding: 0; /* Remove padding from slider */
    }
    .image-container {
        max-height: 150px; /* Reduced height for image */
        overflow: hidden;
    }
    .chart-container {
        max-height: 300px; /* Reduced height for chart to fit */
        overflow: hidden; /* Disable chart scrolling to prevent overflow */
    }
    .footer {
        text-align: center;
        color: #666;
        font-style: italic;
        font-size: 12px; /* Smaller footer text */
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Layout with columns for single-page view
col1, col2 = st.columns([1, 3])  # Simplified layout: image and content

with col1:
    # Load and display header image (compact)
    try:
        image = Image.open("image-gui.jpg")
        st.image(image, caption="Building Sensor Network", use_container_width=True, output_format="auto", width=250)  # Reduced width for compactness
    except FileNotFoundError:
        st.write("Note: Add an 'iot_sensor_image.jpg' for a header image (optional).")

with col2:
    # Title and description (compact)
    st.markdown("<h1 class='header'>Building Sensor Prediction Dashboard</h1>", unsafe_allow_html=True)
    st.write("Real-time sensor type predictions from IoT telemetry!")

    # Streaming controls (compact)
    st.write("RPi Streaming predictions...")
    speed = st.slider("Streaming Speed (seconds per row)", 0.1, 2.0, 0.5, key="speed_slider", label_visibility="collapsed")
    pause = st.button("Pause Streaming", key="pause_button")

    # Initialize data buffer for the live chart with consistent float types
    sensor_data = pd.DataFrame(columns=['co', 'humidity', 'light', 'lpg', 'motion', 'smoke', 'temp'], dtype=float)
    max_points = 150  # Reduced points to fit in chart height
    chart_placeholder = st.empty()

    # Prediction area with limited height
    prediction_placeholder = st.empty()

    def fetch_predictions(telemetry):
        # Convert NumPy types (e.g., bool_) to Python native types for JSON serialization
        clean_telemetry = {}
        for key, value in telemetry.items():
            if isinstance(value, np.bool_):
                clean_telemetry[key] = bool(value)  # Convert np.bool_ to Python bool
            elif isinstance(value, np.floating):
                clean_telemetry[key] = float(value)  # Convert np.float to Python float
            elif isinstance(value, np.integer):
                clean_telemetry[key] = int(value)  # Convert np.int to Python int
            else:
                clean_telemetry[key] = value  # Keep other types as-is

        try:
            response = requests.post("http://192.168.29.16:8000/predict", json=clean_telemetry, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            st.error(f"Error fetching predictions: {str(e)}")
            return None

    # Load inference data
    try:
        inference_data = pd.read_csv("iot_inference_data.csv")
        inference_data["ts"] = pd.to_datetime(inference_data["ts"], unit="s", errors="coerce")
        # Convert all numeric columns to float and boolean columns to Python bool
        for col in ['co', 'humidity', 'light', 'lpg', 'motion', 'smoke', 'temp']:
            if col in inference_data.columns:
                if col in ['light', 'motion']:  # Binary columns
                    inference_data[col] = inference_data[col].apply(lambda x: bool(x) if pd.notna(x) else False).astype(int)  # Convert to 0/1 int for chart
                else:  # Numeric columns
                    inference_data[col] = pd.to_numeric(inference_data[col], errors='coerce').astype(float)
    except FileNotFoundError:
        st.error("Error: 'iot_inference_data.csv' not found. Using simulated data instead.")
        inference_data = pd.DataFrame(columns=['ts', 'device', 'co', 'humidity', 'light', 'lpg', 'motion', 'smoke', 'temp'])

    # Streaming loop with pause functionality and live chart
    paused = False
    i = 0
    while True:  # Infinite loop for live streaming (until paused or app closed)
        if paused and pause:
            continue
        if i >= len(inference_data) and not inference_data.empty:
            i = 0  # Loop back to start if data runs out
        row = inference_data.iloc[i] if not inference_data.empty else None

        if row is not None:
            telemetry = {"ts": row["ts"].timestamp(), "device": row["device"]}
            sensor_cols = [col for col in inference_data.columns if col in ["co", "humidity", "light", "lpg", "smoke", "temp", "motion"]]
            for j, col in enumerate(sensor_cols):
                telemetry[f"r{j+1}"] = row[col]

            # Prepare new values for the chart (convert to float for consistency)
            new_values = {}
            for col in sensor_cols:
                value = row[col]
                if pd.isna(value):
                    value = 0.0  # Handle missing values
                elif col in ['light', 'motion']:
                    new_values[col] = int(bool(value))  # Convert to 0/1 for binary sensors
                else:
                    new_values[col] = float(value)  # Ensure numeric sensors are float

            # Add new values to the DataFrame for the chart
            new_row = pd.DataFrame([new_values])
            sensor_data = pd.concat([sensor_data, new_row], ignore_index=True)

            # Keep only the last 'max_points' entries for performance and reset periodically
            if len(sensor_data) > max_points:
                sensor_data = sensor_data.tail(max_points)
            if i % 1000 == 0 and i > 0:  # Reset every 1000 rows to prevent memory growth
                sensor_data = pd.DataFrame(columns=sensor_data.columns, dtype=float)

            # Update the live chart with specific y columns (numeric sensors only)
            numeric_sensors = ['co', 'humidity', 'lpg', 'smoke', 'temp']  # Exclude binary light/motion for chart
            with chart_placeholder.container():
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                st.line_chart(sensor_data[numeric_sensors], width=1200, height=300)  # Reduced height to fit
                st.markdown("</div>", unsafe_allow_html=True)

            # Fetch and display predictions
            result = fetch_predictions(telemetry)
            if result and "predictions" in result:
                with prediction_placeholder.container():
                    st.markdown(f"<div class='prediction'>**Row {i+1}:**</div>", unsafe_allow_html=True)
                    for pred in result["predictions"]:
                        st.markdown(f"<div class='prediction'>  Value: {pred['value']:.4f}, Predicted Type: {pred['sensor_type']}, Confidence: {pred['confidence']:.2f}</div>", unsafe_allow_html=True)
            else:
                with prediction_placeholder.container():
                    st.markdown("<div class='prediction'>**Error:** No predictions received.</div>", unsafe_allow_html=True)

        if pause:
            paused = not paused
        else:
            time.sleep(speed)

        i += 1

    # Compact footer
    st.markdown("""
        <p class='footer'>Developed for [Sensor Identification BITs Dissertation] - Pushpendra Singh 2022ac05343</p>
    """, unsafe_allow_html=True)
