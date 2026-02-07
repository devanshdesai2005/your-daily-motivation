# Urban Traffic Congestion Forecasting System

A production-grade, hackathon-ready ML system for multivariate traffic congestion forecasting. See the detailed architecture and methodology in [`docs/solution.md`](docs/solution.md).

## Quickstart (VS Code)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1) Generate synthetic data
```bash
python src/data_generation.py
```

### 2) Train models
```bash
python src/training.py
```

### 3) Run inference API
```bash
uvicorn src.inference_api:app --reload
```

## Notes
- Baseline and deep learning models are included as optional dependencies.
- This repo is structured to support batch training and streaming inference.
