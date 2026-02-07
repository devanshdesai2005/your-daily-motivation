"""Deep learning models for traffic forecasting."""
from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


@dataclass
class SequenceConfig:
    input_size: int
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2


class LSTMModel(nn.Module):
    def __init__(self, config: SequenceConfig) -> None:
        if nn is None:
            raise ImportError("torch is required for LSTMModel")
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            batch_first=True,
        )
        self.head = nn.Linear(config.hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class GRUModel(nn.Module):
    def __init__(self, config: SequenceConfig) -> None:
        if nn is None:
            raise ImportError("torch is required for GRUModel")
        super().__init__()
        self.gru = nn.GRU(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            batch_first=True,
        )
        self.head = nn.Linear(config.hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class TemporalConvNet(nn.Module):
    def __init__(self, input_size: int, num_channels: list[int]) -> None:
        if nn is None:
            raise ImportError("torch is required for TemporalConvNet")
        super().__init__()
        layers = []
        for idx, out_channels in enumerate(num_channels):
            dilation = 2**idx
            in_channels = input_size if idx == 0 else num_channels[idx - 1]
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.network(x)
        out = out[:, :, -1]
        return self.head(out)


class TemporalTransformer(nn.Module):
    def __init__(self, input_size: int, num_heads: int = 4, num_layers: int = 2) -> None:
        if nn is None:
            raise ImportError("torch is required for TemporalTransformer")
        super().__init__()
        self.embedding = nn.Linear(input_size, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        emb = self.embedding(x)
        out = self.encoder(emb)
        return self.head(out[:, -1, :])


class GraphTrafficModel(nn.Module):
    def __init__(self, input_size: int) -> None:
        if nn is None:
            raise ImportError("torch is required for GraphTrafficModel")
        super().__init__()
        self.spatial_fc = nn.Linear(input_size, 64)
        self.temporal_fc = nn.GRU(64, 64, batch_first=True)
        self.head = nn.Linear(64, 1)

    def forward(self, x, adjacency=None):
        spatial = torch.relu(self.spatial_fc(x))
        out, _ = self.temporal_fc(spatial)
        return self.head(out[:, -1, :])
