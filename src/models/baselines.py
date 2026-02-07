"""Baseline statistical models."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:  # pragma: no cover
    SARIMAX = None

try:
    from prophet import Prophet
except ImportError:  # pragma: no cover
    Prophet = None


@dataclass
class SarimaxConfig:
    order: tuple[int, int, int] = (1, 1, 1)
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 24)


class SarimaxModel:
    def __init__(self, config: SarimaxConfig) -> None:
        if SARIMAX is None:
            raise ImportError("statsmodels is required for SARIMAX")
        self.config = config
        self.model = None

    def fit(self, series: pd.Series) -> None:
        self.model = SARIMAX(series, order=self.config.order, seasonal_order=self.config.seasonal_order).fit(disp=False)

    def predict(self, steps: int) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.forecast(steps=steps).to_numpy()


class ProphetModel:
    def __init__(self) -> None:
        if Prophet is None:
            raise ImportError("prophet is required for ProphetModel")
        self.model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.1,
        )

    def fit(self, df: pd.DataFrame) -> None:
        self.model.fit(df.rename(columns={"timestamp": "ds", "y": "y"}))

    def predict(self, future_df: pd.DataFrame) -> np.ndarray:
        forecast = self.model.predict(future_df.rename(columns={"timestamp": "ds"}))
        return forecast["yhat"].to_numpy()
