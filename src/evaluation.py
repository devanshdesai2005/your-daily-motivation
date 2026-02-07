"""Evaluation utilities for congestion forecasting."""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-6
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-6
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def segment_metrics(df: pd.DataFrame, y_true_col: str, y_pred_col: str) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    df["segment"] = np.where(df["hour"].between(7, 10) | df["hour"].between(16, 19), "peak", "off_peak")
    metrics = []
    for segment, seg_df in df.groupby("segment"):
        metrics.append(
            {
                "segment": segment,
                "mae": mae(seg_df[y_true_col].to_numpy(), seg_df[y_pred_col].to_numpy()),
                "rmse": rmse(seg_df[y_true_col].to_numpy(), seg_df[y_pred_col].to_numpy()),
                "mape": mape(seg_df[y_true_col].to_numpy(), seg_df[y_pred_col].to_numpy()),
                "smape": smape(seg_df[y_true_col].to_numpy(), seg_df[y_pred_col].to_numpy()),
            }
        )
    return pd.DataFrame(metrics)
