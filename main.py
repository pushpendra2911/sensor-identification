from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from collections import deque
import numpy as np
from scipy.stats import skew
import logging
import uvicorn

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for detailed logs
logger = logging.getLogger(__name__)

# Load trained components
try:
    model = joblib.load("sensor_lgbm_classifier.joblib")
    encoder_sensor = joblib.load("encoder_sensor.joblib")
    scalers = joblib.load("sensor_scalers.joblib")
    transformers = joblib.load("sensor_transformers.joblib")
    imputer = joblib.load("imputer.joblib")
    logger.info("Model files loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model files: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Failed to load model files: {str(e)}")

ranges = {
    'co': {'min': 0.0011705085484379, 'max': 0.0135048080389674},
    'humidity': {'min': 1.100000023841858, 'max': 99.9000015258789},
    'light': {'min': 0.0, 'max': 1.0},
    'lpg': {'min': 0.0026934786226618, 'max': 0.0157997856147855},
    'motion': {'min': 0.0, 'max': 1.0},
    'smoke': {'min': 0.0066920963173865, 'max': 0.0442880442945307},
    'temp': {'min': 0.0, 'max': 30.600000381469727}
}

feature_columns = [
    "sensor_value", "sensor_value_rolling_mean", "sensor_value_rolling_median", "sensor_value_rolling_max",
    "sensor_value_rolling_min", "sensor_value_rolling_std", "sensor_value_rolling_variance", "sensor_value_range",
    "sensor_value_skew", "sensor_value_diff", "value_ratio", "value_relative_to_max", "hour", "day_of_week",
    "elapsed_time", "is_boolean", "value_magnitude", "is_gas_range", "is_temp_range"
]

app = FastAPI()
history = {}

def predict_sensor_types(telemetry_row, window_size=10):
    try:
        logger.debug(f"Processing telemetry: {telemetry_row}")
        row_df = pd.DataFrame([telemetry_row])
        row_df["ts"] = pd.to_datetime(row_df["ts"], unit="s", errors="coerce")
        row_df["hour"] = row_df["ts"].dt.hour
        row_df["day_of_week"] = row_df["ts"].dt.dayofweek
        row_df["elapsed_time"] = (row_df["ts"] - pd.Timestamp("1970-01-01")).dt.total_seconds()
        device_id = row_df["device"].iloc[0]
        
        # Map r1, r2, etc., to sensor values (adjust based on your data structure)
        sensor_values = {}
        for i in range(1, 7):  # Assuming 6 sensors (e.g., co, humidity, light, lpg, smoke, temp)
            key = f"r{i}"
            if key in row_df.columns:
                sensor_values[key] = float(row_df[key].iloc[0])
        
        if not sensor_values:
            raise HTTPException(status_code=400, detail="No sensor values (r1-r6) provided in request")

        melted_df = pd.DataFrame({
            "ts": [row_df["ts"].iloc[0]] * len(sensor_values), "hour": [row_df["hour"].iloc[0]] * len(sensor_values),
            "day_of_week": [row_df["day_of_week"].iloc[0]] * len(sensor_values),
            "elapsed_time": [row_df["elapsed_time"].iloc[0]] * len(sensor_values), "sensor_value": list(sensor_values.values())
        })
        melted_df["is_boolean"] = melted_df["sensor_value"].isin([0, 1]).astype(int)
        melted_df["value_magnitude"] = np.log10(melted_df["sensor_value"].apply(lambda x: abs(x) + 1e-6))
        
        if device_id not in history:
            history[device_id] = deque(maxlen=window_size)
        window = history[device_id]
        temp_df = melted_df[["sensor_value"]].copy()
        means, medians, maxs, mins, stds, vars, ranges_col, skews, diffs = [], [], [], [], [], [], [], [], []
        prev_val = None
        for val in temp_df["sensor_value"]:
            window.append(val)
            arr = np.array(window)
            means.append(np.mean(arr))
            medians.append(np.median(arr))
            maxs.append(np.max(arr))
            mins.append(np.min(arr))
            stds.append(np.std(arr) if len(arr) > 1 else 0)
            vars.append(np.var(arr) if len(arr) > 1 else 0)
            ranges_col.append(np.max(arr) - np.min(arr) if len(arr) > 1 else 0)
            skew_val = skew(arr) if len(arr) > 2 and np.var(arr) > 1e-6 else 0
            skews.append(skew_val)
            diffs.append(val - prev_val if prev_val is not None else 0)
            prev_val = val
        temp_df["sensor_value_rolling_mean"] = means
        temp_df["sensor_value_rolling_median"] = medians
        temp_df["sensor_value_rolling_max"] = maxs
        temp_df["sensor_value_rolling_min"] = mins
        temp_df["sensor_value_rolling_std"] = stds
        temp_df["sensor_value_rolling_variance"] = vars
        temp_df["sensor_value_range"] = ranges_col
        temp_df["sensor_value_skew"] = skews
        temp_df["sensor_value_diff"] = diffs
        melted_df = pd.concat([melted_df, temp_df.drop(columns="sensor_value")], axis=1)
        melted_df["is_gas_range"] = melted_df["sensor_value"].apply(lambda x: 1 if 0.001 <= abs(x) <= 0.05 else 0)
        melted_df["is_temp_range"] = melted_df["sensor_value"].apply(lambda x: 1 if 0 <= abs(x) <= 31 else 0)
        melted_df["value_ratio"] = (melted_df["sensor_value"] - melted_df["sensor_value_rolling_mean"]) / (melted_df["sensor_value_range"] + 1e-6)
        melted_df["value_relative_to_max"] = melted_df["sensor_value"] / (melted_df["sensor_value_rolling_max"] + 1e-6)
        
        X = melted_df[feature_columns].copy()
        logger.debug(f"Features shape: {X.shape}")
        for i, val in enumerate(X["sensor_value"]):
            if val in [0, 1]:
                pass
            else:
                distances = {sensor: min(abs(val - ranges[sensor]["min"]), abs(val - ranges[sensor]["max"])) 
                             for sensor in ranges if sensor not in ["light", "motion"]}
                closest_sensor = min(distances, key=distances.get)
                if closest_sensor in transformers:
                    X.iloc[i, X.columns.get_loc("sensor_value")] = transformers[closest_sensor].transform([[val]])[0][0]
                elif closest_sensor in scalers and scalers[closest_sensor]:
                    X.iloc[i, X.columns.get_loc("sensor_value")] = scalers[closest_sensor].transform([[val]])[0][0]
        
        X = pd.DataFrame(imputer.transform(X), columns=feature_columns)
        pred_proba = model.predict(X)
        
        predictions = []
        for i, proba in enumerate(pred_proba):
            confidence = max(proba)
            sensor_type = encoder_sensor.inverse_transform([np.argmax(proba)])[0] if confidence >= 0.5 else "unknown"
            predictions.append({"value": melted_df["sensor_value"].iloc[i], "sensor_type": sensor_type, "confidence": confidence})
        
        logger.info(f"Predictions: {predictions}")
        return predictions, history
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)  # Include full stack trace
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict")
async def predict(telemetry: dict):
    global history
    predictions, history = predict_sensor_types(telemetry)
    return {"predictions": predictions}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)