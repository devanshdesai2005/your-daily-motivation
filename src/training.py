"""Training orchestration for congestion forecasting models."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit

from src.config.settings import DATA_DIR, MODEL_DIR, TrainingConfig
from src.feature_engineering import add_lag_features, add_rolling_features, build_feature_table
from src.preprocessing import preprocess_features
from src.models.ml_models import RandomForestModel, TreeModelConfig


def time_series_split(df: pd.DataFrame, config: TrainingConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("timestamp")
    total_hours = df["timestamp"].nunique()
    test_hours = config.test_weeks * 7 * 24
    val_hours = config.validation_weeks * 7 * 24
    train_hours = total_hours - test_hours - val_hours

    unique_times = sorted(df["timestamp"].unique())
    train_cut = unique_times[train_hours - 1]
    val_cut = unique_times[train_hours + val_hours - 1]

    train_df = df[df["timestamp"] <= train_cut]
    val_df = df[(df["timestamp"] > train_cut) & (df["timestamp"] <= val_cut)]
    test_df = df[df["timestamp"] > val_cut]
    return train_df, val_df, test_df


def prepare_dataset(features_df: pd.DataFrame, config: TrainingConfig) -> tuple[pd.DataFrame, list[str]]:
    features_df = add_lag_features(features_df, config.target_col, lags=list(range(1, config.lag_hours + 1)))
    features_df = add_rolling_features(features_df, config.target_col, windows=[3, 6, 12, 24])
    features_df = features_df.dropna()

    feature_cols = [
        col
        for col in features_df.columns
        if col not in {"timestamp", "zone_id", config.target_col, "event_type"}
    ]
    return features_df, feature_cols


def train_models(features_df: pd.DataFrame, config: TrainingConfig) -> dict[str, Path]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    features_df, feature_cols = prepare_dataset(features_df, config)

    numeric_cols = [col for col in feature_cols if features_df[col].dtype != object]
    processed_df, artifacts = preprocess_features(features_df, numeric_cols)

    train_df, val_df, test_df = time_series_split(processed_df, config)
    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df[config.target_col].to_numpy()

    rf_model = RandomForestModel(TreeModelConfig())
    rf_model.fit(X_train, y_train)

    model_paths = {
        "random_forest": MODEL_DIR / "random_forest.joblib",
        "preprocessing": MODEL_DIR / "preprocessing.joblib",
        "feature_columns": MODEL_DIR / "feature_columns.json",
        "training_config": MODEL_DIR / "training_config.json",
    }

    joblib.dump(rf_model, model_paths["random_forest"])
    joblib.dump(artifacts, model_paths["preprocessing"])
    Path(model_paths["feature_columns"]).write_text(pd.Series(feature_cols).to_json())
    Path(model_paths["training_config"]).write_text(pd.Series(asdict(config)).to_json())

    return model_paths


def run_training_pipeline() -> dict[str, Path]:
    gps_df = pd.read_csv(DATA_DIR / "gps_traces.csv")
    sensors_df = pd.read_csv(DATA_DIR / "traffic_sensors.csv")
    weather_df = pd.read_csv(DATA_DIR / "weather.csv")
    events_df = pd.read_csv(DATA_DIR / "events.csv")

    features_df = build_feature_table(gps_df, sensors_df, weather_df, events_df)
    config = TrainingConfig()
    return train_models(features_df, config)


if __name__ == "__main__":
    paths = run_training_pipeline()
    print("Saved artifacts:")
    for name, path in paths.items():
        print(f"{name}: {path}")
