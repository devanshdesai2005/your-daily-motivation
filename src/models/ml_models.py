"""Machine learning models for tabular time-series features."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None

from sklearn.ensemble import RandomForestRegressor


@dataclass
class TreeModelConfig:
    n_estimators: int = 300
    max_depth: int = 10
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8


class RandomForestModel:
    def __init__(self, config: TreeModelConfig) -> None:
        self.model = RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class LightGBMModel:
    def __init__(self, config: TreeModelConfig) -> None:
        if lgb is None:
            raise ImportError("lightgbm is required for LightGBMModel")
        self.model = lgb.LGBMRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            objective="regression",
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class XGBoostModel:
    def __init__(self, config: TreeModelConfig) -> None:
        if xgb is None:
            raise ImportError("xgboost is required for XGBoostModel")
        self.model = xgb.XGBRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            objective="reg:squarederror",
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
