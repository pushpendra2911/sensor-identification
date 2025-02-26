import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from collections import deque
from scipy.stats import skew

# Load and preprocess training data
data = pd.read_csv("iot_train_data.csv")
for col in ['light', 'motion']:
    if col in data.columns:
        data[col] = data[col].astype(int)
data["ts"] = pd.to_datetime(data["ts"], unit="s", errors="coerce")
data["hour"] = data["ts"].dt.hour
data["day_of_week"] = data["ts"].dt.dayofweek
data["elapsed_time"] = (data["ts"] - data["ts"].min()).dt.total_seconds()

sensor_columns = ["co", "humidity", "light", "lpg", "motion", "smoke", "temp"]
train_df = data.melt(id_vars=["ts", "device", "hour", "day_of_week", "elapsed_time"],
                     value_vars=sensor_columns, var_name="sensor_type", value_name="sensor_value")
train_df["sensor_value"] = pd.to_numeric(train_df["sensor_value"], errors="coerce")
train_df["is_boolean"] = train_df["sensor_value"].isin([0, 1]).astype(int)
train_df["value_magnitude"] = np.log10(train_df["sensor_value"].abs() + 1e-6)

# Feature engineering
def streaming_feature_engineering(df, window_size=10):
    df = df.copy()
    grouped = df.groupby("device")
    for name, group in grouped:
        window = deque(maxlen=window_size)
        means, medians, maxs, mins, stds, vars, ranges, skews, diffs = [], [], [], [], [], [], [], [], []
        prev_val = None
        for val in group["sensor_value"]:
            window.append(val)
            arr = np.array(window)
            means.append(np.mean(arr))
            medians.append(np.median(arr))
            maxs.append(np.max(arr))
            mins.append(np.min(arr))
            stds.append(np.std(arr) if len(arr) > 1 else 0)
            vars.append(np.var(arr) if len(arr) > 1 else 0)
            ranges.append(np.max(arr) - np.min(arr) if len(arr) > 1 else 0)
            skew_val = skew(arr) if len(arr) > 2 and np.var(arr) > 1e-6 else 0
            skews.append(skew_val)
            diffs.append(val - prev_val if prev_val is not None else 0)
            prev_val = val
        df.loc[group.index, "sensor_value_rolling_mean"] = means
        df.loc[group.index, "sensor_value_rolling_median"] = medians
        df.loc[group.index, "sensor_value_rolling_max"] = maxs
        df.loc[group.index, "sensor_value_rolling_min"] = mins
        df.loc[group.index, "sensor_value_rolling_std"] = stds
        df.loc[group.index, "sensor_value_rolling_variance"] = vars
        df.loc[group.index, "sensor_value_range"] = ranges
        df.loc[group.index, "sensor_value_skew"] = skews
        df.loc[group.index, "sensor_value_diff"] = diffs
        df["is_gas_range"] = df["sensor_value"].apply(lambda x: 1 if 0.001 <= abs(x) <= 0.05 else 0)
        df["is_temp_range"] = df["sensor_value"].apply(lambda x: 1 if 0 <= abs(x) <= 31 else 0)
        df["value_ratio"] = (df["sensor_value"] - df["sensor_value_rolling_mean"]) / (df["sensor_value_range"] + 1e-6)
        df["value_relative_to_max"] = df["sensor_value"] / (df["sensor_value_rolling_max"] + 1e-6)
    return df

train_df = streaming_feature_engineering(train_df)

# Encode labels
encoder_sensor = LabelEncoder()
train_df["sensor_label"] = encoder_sensor.fit_transform(train_df["sensor_type"])
joblib.dump(encoder_sensor, "encoder_sensor.joblib")

# Define features
feature_columns = [
    "sensor_value", "sensor_value_rolling_mean", "sensor_value_rolling_median", "sensor_value_rolling_max",
    "sensor_value_rolling_min", "sensor_value_rolling_std", "sensor_value_rolling_variance", "sensor_value_range",
    "sensor_value_skew", "sensor_value_diff", "value_ratio", "value_relative_to_max", "hour", "day_of_week",
    "elapsed_time", "is_boolean", "value_magnitude", "is_gas_range", "is_temp_range"
]

# Scaling per sensor type
scalers, transformers = {}, {}
ranges = train_df.groupby("sensor_type")["sensor_value"].agg(['min', 'max']).to_dict('index')
print("Ranges after definition:", ranges)

for sensor in train_df["sensor_type"].unique():
    mask = train_df["sensor_type"] == sensor
    sensor_values = train_df.loc[mask, "sensor_value"]
    if sensor_values.nunique() == 2:
        scalers[sensor] = None
    elif sensor_values.skew() > 1.0:
        transformers[sensor] = FunctionTransformer(np.log1p, validate=True)
        train_df.loc[mask, "sensor_value"] = transformers[sensor].fit_transform(sensor_values.values.reshape(-1, 1))
    else:
        scalers[sensor] = MinMaxScaler()
        train_df.loc[mask, "sensor_value"] = scalers[sensor].fit_transform(sensor_values.values.reshape(-1, 1))
joblib.dump(scalers, "sensor_scalers.joblib")
joblib.dump(transformers, "sensor_transformers.joblib")

# Split data
X = train_df[feature_columns]
y = train_df["sensor_label"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute
imputer = SimpleImputer(strategy="mean")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_columns)
X_val = pd.DataFrame(imputer.transform(X_val), columns=feature_columns)
joblib.dump(imputer, "imputer.joblib")

# Compute sample weights
class_counts = train_df["sensor_label"].value_counts()
max_count = max(class_counts)
gas_boost = 2.0
weights = np.array([max_count / class_counts[label] * (gas_boost if encoder_sensor.inverse_transform([label])[0] in ['co', 'lpg', 'smoke'] else 1) for label in y_train])

# Train LightGBM
lgbm_params = {
    "max_depth": 5, "learning_rate": 0.03, "num_leaves": 20, "lambda_l1": 7.0, "lambda_l2": 7.0,
    "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 10, "objective": "multiclass",
    "num_class": len(encoder_sensor.classes_), "random_state": 42, "verbose": -1
}
train_data = lgb.Dataset(X_train, label=y_train, weight=weights)
valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
model = lgb.train(lgbm_params, train_data, valid_sets=[valid_data], num_boost_round=2000,
                  callbacks=[lgb.early_stopping(stopping_rounds=50)])
joblib.dump(model, "sensor_lgbm_classifier.joblib")

print("Model trained and saved successfully!")