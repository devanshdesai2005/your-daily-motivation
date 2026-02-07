"""Data preprocessing pipeline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler


@dataclass
class PreprocessingArtifacts:
    imputer: KNNImputer
    scaler: RobustScaler


def handle_missing_values(df: pd.DataFrame, numeric_cols: list[str]) -> tuple[pd.DataFrame, KNNImputer]:
    imputer = KNNImputer(n_neighbors=5, weights="distance")
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df, imputer


def detect_outliers(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Flag outliers using IQR and robust z-score."""
    df = df.copy()
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        median = df[col].median()
        mad = np.median(np.abs(df[col] - median)) + 1e-6
        robust_z = 0.6745 * (df[col] - median) / mad
        df[f"{col}_outlier"] = ((df[col] < lower) | (df[col] > upper) | (np.abs(robust_z) > 3.5)).astype(int)
    return df


def scale_features(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, RobustScaler]:
    scaler = RobustScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df, scaler


def preprocess_features(df: pd.DataFrame, numeric_cols: list[str]) -> tuple[pd.DataFrame, PreprocessingArtifacts]:
    df = df.copy()
    df, imputer = handle_missing_values(df, numeric_cols)
    df = detect_outliers(df, numeric_cols)
    df, scaler = scale_features(df, numeric_cols)
    artifacts = PreprocessingArtifacts(imputer=imputer, scaler=scaler)
    return df, artifacts
