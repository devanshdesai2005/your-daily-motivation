"""Configuration for traffic congestion forecasting system."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "artifacts"
LOG_DIR = PROJECT_ROOT / "logs"


@dataclass
class SimulationConfig:
    """Configuration for synthetic data generation."""

    start_timestamp: str = "2024-01-01 00:00:00"
    end_timestamp: str = "2024-03-31 23:00:00"
    freq: str = "H"
    num_zones: int = 12
    num_sensors_per_zone: int = 6
    num_road_segments: int = 30
    gps_points_per_hour: int = 450
    seed: int = 42


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    target_col: str = "congestion_index"
    horizon_hours: int = 6
    daily_horizon: int = 24
    validation_weeks: int = 2
    test_weeks: int = 2
    rolling_window_hours: int = 24
    lag_hours: int = 24
    random_state: int = 42


@dataclass
class InferenceConfig:
    """Configuration for inference pipeline."""

    sliding_window_hours: int = 48
    refresh_minutes: int = 15
    max_latency_ms: int = 250
