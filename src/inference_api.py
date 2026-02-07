"""FastAPI-based inference service."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI

from src.config.settings import MODEL_DIR
from src.feature_engineering import add_time_features

app = FastAPI(title="Traffic Congestion Forecasting API")

MODEL_PATH = MODEL_DIR / "random_forest.joblib"
PREPROCESS_PATH = MODEL_DIR / "preprocessing.joblib"
FEATURE_COLS_PATH = MODEL_DIR / "feature_columns.json"


class ModelBundle:
    def __init__(self) -> None:
        if MODEL_PATH.exists():
            self.model = joblib.load(MODEL_PATH)
            self.preprocess = joblib.load(PREPROCESS_PATH)
            self.feature_cols = pd.read_json(FEATURE_COLS_PATH, typ="series").tolist()
        else:
            self.model = None
            self.preprocess = None
            self.feature_cols = []

    def predict(self, payload: dict[str, Any]) -> float:
        if self.model is None:
            raise RuntimeError("Model artifacts not found. Train the model first.")
        df = pd.DataFrame([payload])
        df = add_time_features(df)
        numeric_cols = [col for col in self.feature_cols if col in df.columns and df[col].dtype != object]
        df[numeric_cols] = self.preprocess.imputer.transform(df[numeric_cols])
        df[numeric_cols] = self.preprocess.scaler.transform(df[numeric_cols])
        X = df[self.feature_cols].to_numpy()
        return float(self.model.predict(X)[0])


bundle = ModelBundle()


@app.post("/predict")
async def predict(payload: dict[str, Any]) -> dict[str, float]:
    prediction = bundle.predict(payload)
    return {"prediction": prediction}
