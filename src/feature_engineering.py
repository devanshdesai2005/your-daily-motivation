"""Feature engineering for congestion forecasting."""
from __future__ import annotations

import numpy as np
import pandas as pd


CYCLIC_FEATURES = ["hour", "dayofweek", "month"]


def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col])
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)
    for col in CYCLIC_FEATURES:
        max_val = 24 if col == "hour" else 7 if col == "dayofweek" else 12
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / max_val)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / max_val)
    return df


def add_lag_features(df: pd.DataFrame, target_col: str, lags: list[int]) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("timestamp")
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df.groupby("zone_id")[target_col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target_col: str, windows: list[int]) -> pd.DataFrame:
    df = df.copy()
    for window in windows:
        df[f"{target_col}_roll_mean_{window}"] = (
            df.groupby("zone_id")[target_col].transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        df[f"{target_col}_roll_std_{window}"] = (
            df.groupby("zone_id")[target_col].transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        )
    return df


def aggregate_gps(gps_df: pd.DataFrame) -> pd.DataFrame:
    gps_df = gps_df.copy()
    gps_df["timestamp"] = pd.to_datetime(gps_df["timestamp"]).dt.floor("H")
    agg = gps_df.groupby(["timestamp", "zone_id"]).agg(
        gps_speed_mean=("speed_kph", "mean"),
        gps_speed_std=("speed_kph", "std"),
        gps_vehicle_count=("vehicle_id", "nunique"),
    )
    agg = agg.reset_index()
    agg["gps_speed_std"] = agg["gps_speed_std"].fillna(0)
    return agg


def aggregate_sensors(sensor_df: pd.DataFrame) -> pd.DataFrame:
    sensor_df = sensor_df.copy()
    sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])
    agg = sensor_df.groupby(["timestamp", "zone_id"]).agg(
        sensor_count=("sensor_id", "nunique"),
        vehicle_count=("vehicle_count", "sum"),
        occupancy_mean=("occupancy", "mean"),
        speed_mean=("avg_speed_kph", "mean"),
    )
    agg = agg.reset_index()
    return agg


def enrich_with_events(feature_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    events_df = events_df.copy()
    events_df["start_time"] = pd.to_datetime(events_df["start_time"])
    events_df["end_time"] = pd.to_datetime(events_df["end_time"])
    feature_df = feature_df.copy()
    feature_df["event_active"] = 0
    feature_df["event_attendance"] = 0
    feature_df["event_type"] = "none"

    for _, event in events_df.iterrows():
        mask = (
            (feature_df["zone_id"] == event["zone_id"])
            & (feature_df["timestamp"] >= event["start_time"])
            & (feature_df["timestamp"] <= event["end_time"])
        )
        feature_df.loc[mask, "event_active"] = 1
        feature_df.loc[mask, "event_attendance"] = event["expected_attendance"]
        feature_df.loc[mask, "event_type"] = event["event_type"]
    return feature_df


def build_feature_table(
    gps_df: pd.DataFrame,
    sensors_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> pd.DataFrame:
    gps_agg = aggregate_gps(gps_df)
    sensors_agg = aggregate_sensors(sensors_df)
    weather_df = weather_df.copy()
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])

    merged = sensors_agg.merge(gps_agg, on=["timestamp", "zone_id"], how="left")
    merged = merged.merge(weather_df, on="timestamp", how="left")
    merged = enrich_with_events(merged, events_df)
    merged = add_time_features(merged)

    merged["gps_speed_mean"] = merged["gps_speed_mean"].fillna(merged["speed_mean"])
    merged["gps_vehicle_count"] = merged["gps_vehicle_count"].fillna(merged["vehicle_count"])
    merged["gps_speed_std"] = merged["gps_speed_std"].fillna(0)

    merged["congestion_index"] = (
        0.45 * (merged["vehicle_count"] / (merged["sensor_count"] + 1))
        + 0.35 * merged["occupancy_mean"]
        + 0.2 * (1 - merged["speed_mean"] / 80)
    )

    return merged
