"""Synthetic data generation for traffic congestion forecasting."""
from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import DATA_DIR, SimulationConfig


def _seasonal_component(hour: int) -> float:
    morning_peak = math.exp(-((hour - 8) ** 2) / 12)
    evening_peak = math.exp(-((hour - 17) ** 2) / 10)
    return 0.6 * morning_peak + 0.8 * evening_peak


def _weather_intensity(timestamp: pd.Timestamp, rng: np.random.Generator) -> tuple[float, float, float, float]:
    day_factor = 0.2 + 0.8 * math.sin(2 * math.pi * timestamp.dayofyear / 365)
    rain = max(0.0, rng.normal(0.3 * day_factor, 0.2))
    temp = rng.normal(20 + 10 * math.sin(2 * math.pi * timestamp.dayofyear / 365), 5)
    visibility = max(0.5, rng.normal(10 - 4 * rain, 1.5))
    wind = max(0.0, rng.normal(8 + 6 * rain, 2))
    return rain, temp, visibility, wind


def generate_datasets(config: SimulationConfig, output_dir: Path | None = None) -> dict[str, Path]:
    rng = np.random.default_rng(config.seed)
    output_dir = output_dir or DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamps = pd.date_range(config.start_timestamp, config.end_timestamp, freq=config.freq)
    zones = [f"zone_{idx:02d}" for idx in range(1, config.num_zones + 1)]
    road_segments = [f"seg_{idx:03d}" for idx in range(1, config.num_road_segments + 1)]

    weather_records = []
    for ts in timestamps:
        rain, temp, visibility, wind = _weather_intensity(ts, rng)
        weather_records.append(
            {
                "timestamp": ts,
                "rain_mm": rain,
                "temperature_c": temp,
                "visibility_km": visibility,
                "wind_kph": wind,
            }
        )
    weather_df = pd.DataFrame(weather_records)

    event_records = []
    for zone in zones:
        for month in range(1, 4):
            event_start = pd.Timestamp(f"2024-{month:02d}-{rng.integers(1, 25):02d} 18:00:00")
            event_end = event_start + pd.Timedelta(hours=rng.integers(3, 8))
            event_records.append(
                {
                    "event_id": f"EVT-{zone}-{month}",
                    "zone_id": zone,
                    "event_type": rng.choice(["concert", "sports", "festival", "conference"]),
                    "start_time": event_start,
                    "end_time": event_end,
                    "expected_attendance": rng.integers(4000, 25000),
                }
            )
    events_df = pd.DataFrame(event_records)

    sensor_records = []
    for ts in timestamps:
        hour = ts.hour
        seasonal = _seasonal_component(hour)
        for zone in zones:
            for sensor_idx in range(config.num_sensors_per_zone):
                baseline = 40 + 60 * seasonal + rng.normal(0, 5)
                occupancy = min(1.0, max(0.05, baseline / 120 + rng.normal(0, 0.05)))
                avg_speed = max(10, 60 - baseline + rng.normal(0, 5))
                sensor_records.append(
                    {
                        "timestamp": ts,
                        "zone_id": zone,
                        "sensor_id": f"{zone}_s{sensor_idx:02d}",
                        "vehicle_count": max(0, baseline + rng.normal(0, 10)),
                        "occupancy": occupancy,
                        "avg_speed_kph": avg_speed,
                    }
                )
    sensors_df = pd.DataFrame(sensor_records)

    gps_records = []
    for ts in timestamps:
        for _ in range(config.gps_points_per_hour):
            zone = rng.choice(zones)
            base_lat = 40.7 + 0.03 * zones.index(zone)
            base_lon = -74.0 + 0.02 * zones.index(zone)
            gps_records.append(
                {
                    "timestamp": ts + pd.Timedelta(minutes=int(rng.integers(0, 60))),
                    "vehicle_id": f"veh_{rng.integers(1, 2000):05d}",
                    "zone_id": zone,
                    "latitude": base_lat + rng.normal(0, 0.01),
                    "longitude": base_lon + rng.normal(0, 0.01),
                    "speed_kph": max(5, rng.normal(45, 12)),
                    "heading": rng.uniform(0, 360),
                }
            )
    gps_df = pd.DataFrame(gps_records)

    road_records = []
    for segment in road_segments:
        zone = rng.choice(zones)
        road_records.append(
            {
                "segment_id": segment,
                "zone_id": zone,
                "num_lanes": rng.integers(1, 5),
                "speed_limit_kph": rng.choice([40, 50, 60, 70]),
            }
        )
    roads_df = pd.DataFrame(road_records)

    output_paths = {
        "weather": output_dir / "weather.csv",
        "events": output_dir / "events.csv",
        "sensors": output_dir / "traffic_sensors.csv",
        "gps": output_dir / "gps_traces.csv",
        "roads": output_dir / "roads.csv",
        "simulation_config": output_dir / "simulation_config.json",
    }

    weather_df.to_csv(output_paths["weather"], index=False)
    events_df.to_csv(output_paths["events"], index=False)
    sensors_df.to_csv(output_paths["sensors"], index=False)
    gps_df.to_csv(output_paths["gps"], index=False)
    roads_df.to_csv(output_paths["roads"], index=False)
    Path(output_paths["simulation_config"]).write_text(pd.Series(asdict(config)).to_json())

    return output_paths


if __name__ == "__main__":
    config = SimulationConfig()
    paths = generate_datasets(config)
    print("Generated datasets:")
    for name, path in paths.items():
        print(f"{name}: {path}")
