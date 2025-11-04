"""PyTorch pipeline for intoxication classification from gaze features.

This module provides an end-to-end workflow inspired by the original
Keras/TensorFlow implementation that ships with the repository.  It
reimplements the 1D CNN + (B)LSTM architecture in PyTorch and exposes
command-line entry points for training, evaluation, and running
inference on new sequences.

Example usage (see `--help` for the full list of options)::

    # Train a model and keep the best checkpoint in ./outputs/intox_model
    python pytorch_end_to_end.py train \
        --data data/intox_samples.csv \
        --output-dir outputs/intox_model \
        --window-size 256 \
        --window-stride 64 \
        --epochs 30 \
        --batch-size 64

    # Evaluate the saved model on a hold-out CSV file
    python pytorch_end_to_end.py evaluate \
        --data data/holdout.csv \
        --checkpoint outputs/intox_model/model.pt \
        --config outputs/intox_model/config.json

The input CSV is expected to contain the following columns:

* a `sequence_id` identifying each one-minute recording (string or int),
* one or more feature columns (e.g. `gaze_velocity`, `gaze_azimuth_velocity`,
  `eye_opening`, ...), and
* an `intoxication_label` column containing the global label for the
  sequence (0, 1, 2 in the example above).

Every row corresponds to one timestamped sample.  During training the
dataset is sliced into overlapping windows whose length and stride can
be configured via command-line options.  Windows inherit the parent
sequence's label and are zero-padded when the sequence is shorter than
`window_size`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


def set_seed(seed: int) -> None:
    """Initialise RNGs for repeatability."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sliding_windows(
    sequence: np.ndarray,
    window_size: int,
    stride: int,
) -> List[np.ndarray]:
    """Slice a sequence into fixed-length windows.

    If the sequence is shorter than ``window_size`` it is zero padded.  The
    final chunk is always included to make sure the tail of the sequence is
    covered even when ``len(sequence) - window_size`` is not divisible by
    ``stride``.
    """

    if window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if stride <= 0:
        raise ValueError("stride must be a positive integer")

    if len(sequence) == 0:
        padded = np.zeros((window_size, sequence.shape[1]), dtype=sequence.dtype)
        return [padded]

    if len(sequence) < window_size:
        pad_len = window_size - len(sequence)
        padded = np.pad(sequence, ((0, pad_len), (0, 0)), mode="constant")
        return [padded]

    windows: List[np.ndarray] = []
    last_start = len(sequence) - window_size
    starts = list(range(0, last_start + 1, stride))
    if starts and starts[-1] != last_start:
        starts.append(last_start)
    elif not starts:
        starts = [0]

    for start in starts:
        window = sequence[start : start + window_size]
        windows.append(window)

    return windows


@dataclass
class DatasetStats:
    feature_mean: np.ndarray
    feature_std: np.ndarray

    def as_dict(self) -> Dict[str, List[float]]:
        return {
            "feature_mean": self.feature_mean.astype(float).tolist(),
            "feature_std": self.feature_std.astype(float).tolist(),
        }

    @staticmethod
    def from_dict(payload: Dict[str, Sequence[float]]) -> "DatasetStats":
        return DatasetStats(
            feature_mean=np.asarray(payload["feature_mean"], dtype=np.float32),
            feature_std=np.asarray(payload["feature_std"], dtype=np.float32),
        )


class SequenceWindowDataset(Dataset):
    """Dataset that groups samples by sequence and produces sliding windows."""

    VALID_LABEL_AGGREGATIONS = {"constant", "majority", "last"}

    def __init__(
        self,
        csv_path: Path,
        sequence_column: Optional[str] = "sequence_id",
        label_column: Optional[str] = "intoxication_label",
        feature_columns: Optional[Sequence[str]] = None,
        window_size: int = 256,
        window_stride: int = 64,
        normalization_stats: Optional[DatasetStats] = None,
        label_encoder: Optional[Dict[str, int]] = None,
        label_aggregation: str = "majority",
        require_labels: bool = True,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if window_stride <= 0:
            raise ValueError("window_stride must be positive")

        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        if label_aggregation not in self.VALID_LABEL_AGGREGATIONS:
            raise ValueError(
                "label_aggregation must be one of "
                + ", ".join(sorted(self.VALID_LABEL_AGGREGATIONS))
            )

        self.sequence_column = sequence_column or "__sequence_id__"
        self.label_column = label_column
        self.window_size = window_size
        self.window_stride = window_stride
        self.label_aggregation = label_aggregation
        self.require_labels = require_labels

        df = pd.read_csv(self.csv_path)
        df = df.copy()

        if self.sequence_column not in df.columns:
            df[self.sequence_column] = "sequence_0"

        self.has_labels = (
            self.label_column is not None
            and self.label_column in df.columns
        )
        if self.require_labels and not self.has_labels:
            raise KeyError(
                f"Column '{self.label_column}' not present in {self.csv_path}"
            )

        if feature_columns is None:
            feature_columns = [
                col
                for col in df.columns
                if col not in {self.sequence_column, self.label_column}
            ]
        missing = [col for col in feature_columns if col not in df.columns]
        if missing:
            raise KeyError(
                "The following feature columns are missing in the CSV: "
                + ", ".join(missing)
            )

        if not feature_columns:
            raise ValueError("No feature columns available for modelling")

        self.feature_columns = list(feature_columns)

        feature_matrix = df[self.feature_columns].to_numpy(dtype=np.float32)
        if normalization_stats is None:
            feature_mean = feature_matrix.mean(axis=0)
            feature_std = feature_matrix.std(axis=0)
        else:
            feature_mean = normalization_stats.feature_mean
            feature_std = normalization_stats.feature_std

        feature_std = np.where(feature_std == 0, 1.0, feature_std)
        normalized_features = (feature_matrix - feature_mean) / feature_std

        df_normalized = df.copy()
        df_normalized[self.feature_columns] = normalized_features

        self.normalization_stats = DatasetStats(
            feature_mean=np.asarray(feature_mean, dtype=np.float32),
            feature_std=np.asarray(feature_std, dtype=np.float32),
        )

        provided_encoder = (
            {str(key): int(value) for key, value in label_encoder.items()}
            if label_encoder is not None
            else None
        )

        self.label_encoder: Dict[str, int] = provided_encoder or {}
        self.class_names: List[str] = []
        if self.has_labels:
            label_series = df[self.label_column].astype(str)
            if provided_encoder is None:
                self.label_encoder = self._build_label_encoder(label_series)
            else:
                missing = sorted(
                    set(label_series.unique()) - set(provided_encoder.keys())
                )
                if missing:
                    raise ValueError(
                        "Dataset contains labels that were not seen during training: "
                        + ", ".join(missing)
                    )
            self.class_names = [
                label
                for label, _ in sorted(
                    self.label_encoder.items(), key=lambda item: item[1]
                )
            ]
        elif provided_encoder is not None:
            self.class_names = [
                label
                for label, _ in sorted(
                    self.label_encoder.items(), key=lambda item: item[1]
                )
            ]

        grouped = df_normalized.groupby(self.sequence_column)

        windows: List[np.ndarray] = []
        labels: List[int] = []

        for sequence_id, sequence_df in grouped:
            sequence_features = sequence_df[self.feature_columns].to_numpy(
                dtype=np.float32
            )
            if self.has_labels:
                sequence_labels = sequence_df[self.label_column].astype(str).to_numpy()
                label = self._aggregate_sequence_labels(sequence_labels, sequence_id)
                label_idx = self.label_encoder[label]
            else:
                label_idx = 0

            for window in sliding_windows(
                sequence_features, window_size=self.window_size, stride=self.window_stride
            ):
                windows.append(window)
                labels.append(label_idx)

        if not windows:
            raise ValueError("No windows were generated from the dataset")

        self.windows = torch.from_numpy(np.stack(windows))
        self.labels = torch.tensor(labels, dtype=torch.long)

    def _aggregate_sequence_labels(
        self, sequence_labels: np.ndarray, sequence_id: str
    ) -> str:
        if self.label_aggregation == "constant":
            unique = np.unique(sequence_labels)
            if len(unique) != 1:
                raise ValueError(
                    "Each sequence must map to a single label when "
                    "label_aggregation='constant'. "
                    f"Sequence '{sequence_id}' has labels {unique.tolist()}."
                )
            return str(unique[0])

        if self.label_aggregation == "majority":
            values, counts = np.unique(sequence_labels, return_counts=True)
            max_count = counts.max()
            candidates = sorted(values[counts == max_count])
            return str(candidates[0])

        if self.label_aggregation == "last":
            if len(sequence_labels) == 0:
                raise ValueError(
                    f"Sequence '{sequence_id}' does not contain any labels"
                )
            return str(sequence_labels[-1])

        raise RuntimeError(f"Unsupported label aggregation: {self.label_aggregation}")

    @staticmethod
    def _build_label_encoder(labels: pd.Series) -> Dict[str, int]:
        unique = sorted(labels.astype(str).unique())
        return {label: idx for idx, label in enumerate(unique)}

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.labels[idx]


class TimeDistributedLinear(nn.Module):
    """Apply a linear layer to each time step independently."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, time, features)
        batch, time, features = inputs.shape
        flattened = inputs.reshape(batch * time, features)
        outputs = self.linear(flattened)
        return outputs.view(batch, time, -1)


class ConvBLSTMClassifier(nn.Module):
    """1D CNN followed by (bi-)LSTM layers for sequence classification."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        conv_channels: Sequence[int] = (64, 128, 128),
        conv_kernel_size: int = 5,
        conv_dropout: float = 0.1,
        dense_units: int = 128,
        lstm_hidden_size: int = 128,
        lstm_layers: int = 1,
        lstm_dropout: float = 0.3,
        final_dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if not conv_channels:
            raise ValueError("conv_channels must contain at least one entry")

        conv_layers: List[nn.Module] = []
        in_channels = num_features
        padding = conv_kernel_size // 2
        for out_channels in conv_channels:
            conv_layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=conv_kernel_size,
                        padding=padding,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(conv_dropout),
                ]
            )
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)
        self.time_dense = TimeDistributedLinear(in_channels, dense_units)
        self.time_dense_activation = nn.ReLU(inplace=True)
        self.lstm = nn.LSTM(
            input_size=dense_units,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )

        classifier_input = lstm_hidden_size * 2
        self.classifier = nn.Sequential(
            nn.Dropout(final_dropout),
            nn.Linear(classifier_input, classifier_input // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(final_dropout),
            nn.Linear(classifier_input // 2, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, time, features)
        x = inputs.transpose(1, 2)  # -> (batch, features, time)
        x = self.conv(x)
        x = x.transpose(1, 2)  # -> (batch, time, channels)
        x = self.time_dense_activation(self.time_dense(x))
        lstm_out, _ = self.lstm(x)
        pooled = lstm_out.mean(dim=1)
        logits = self.classifier(pooled)
        return logits


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    correct = (predictions == targets).sum().item()
    return correct / max(1, targets.numel())


def macro_f1_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    num_classes = logits.shape[1]
    f1_scores: List[float] = []

    for class_idx in range(num_classes):
        tp = ((predictions == class_idx) & (targets == class_idx)).sum().item()
        fp = ((predictions == class_idx) & (targets != class_idx)).sum().item()
        fn = ((predictions != class_idx) & (targets == class_idx)).sum().item()

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_f1 = 0.0
    total_batches = 0

    for windows, labels in dataloader:
        windows = windows.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(windows)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits, labels)
        total_f1 += macro_f1_from_logits(logits, labels)
        total_batches += 1

    return (
        total_loss / max(1, total_batches),
        total_acc / max(1, total_batches),
        total_f1 / max(1, total_batches),
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_f1 = 0.0
    total_batches = 0

    for windows, labels in dataloader:
        windows = windows.to(device)
        labels = labels.to(device)

        logits = model(windows)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits, labels)
        total_f1 += macro_f1_from_logits(logits, labels)
        total_batches += 1

    return (
        total_loss / max(1, total_batches),
        total_acc / max(1, total_batches),
        total_f1 / max(1, total_batches),
    )


def build_dataloaders(
    dataset: SequenceWindowDataset,
    batch_size: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0, 1)")
    if not (0.0 <= test_ratio < 1.0):
        raise ValueError("test_ratio must be in [0, 1)")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    total_samples = len(dataset)
    indices = list(range(total_samples))
    generator = torch.Generator().manual_seed(seed)
    train_size = total_samples - int(total_samples * val_ratio) - int(
        total_samples * test_ratio
    )
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size

    splits = torch.utils.data.random_split(
        dataset,
        lengths=[train_size, val_size, test_size],
        generator=generator,
    )

    train_subset: Subset = splits[0]
    val_subset: Subset = splits[1]
    test_subset: Subset = splits[2]

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_loader = (
        DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        if len(test_subset) > 0
        else None
    )

    return train_loader, val_loader, test_loader


def load_checkpoint(
    checkpoint_path: Path,
    config_path: Path,
    device: torch.device,
) -> Tuple[
    ConvBLSTMClassifier,
    Dict[str, int],
    DatasetStats,
    Dict[str, object],
]:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    class_names = config.get("class_names")
    if class_names is None:
        legacy_encoder = config.get("label_encoder")
        if legacy_encoder is None:
            raise KeyError("Configuration must contain 'class_names' or 'label_encoder'")
        class_names = [
            key
            for key, _ in sorted(legacy_encoder.items(), key=lambda item: item[1])
        ]

    label_encoder = {str(name): idx for idx, name in enumerate(class_names)}
    normalization_stats = DatasetStats.from_dict(config["normalization"])

    model = ConvBLSTMClassifier(
        num_features=len(config["feature_columns"]),
        num_classes=len(class_names),
        conv_channels=config["model"]["conv_channels"],
        conv_kernel_size=config["model"]["conv_kernel_size"],
        conv_dropout=config["model"]["conv_dropout"],
        dense_units=config["model"]["dense_units"],
        lstm_hidden_size=config["model"]["lstm_hidden_size"],
        lstm_layers=config["model"]["lstm_layers"],
        lstm_dropout=config["model"]["lstm_dropout"],
        final_dropout=config["model"]["final_dropout"],
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, label_encoder, normalization_stats, config


def train_command(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    window_size = args.window_size or 256
    window_stride = args.window_stride or 64
    label_aggregation = args.label_aggregation or "majority"
    dataset = SequenceWindowDataset(
        csv_path=Path(args.data),
        sequence_column=args.sequence_column,
        label_column=args.label_column,
        feature_columns=args.feature_columns,
        window_size=window_size,
        window_stride=window_stride,
        normalization_stats=None,
        label_aggregation=label_aggregation,
        require_labels=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_loader, val_loader, test_loader = build_dataloaders(
        dataset,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    model = ConvBLSTMClassifier(
        num_features=len(dataset.feature_columns),
        num_classes=len(dataset.class_names),
        conv_channels=args.conv_channels,
        conv_kernel_size=args.conv_kernel_size,
        conv_dropout=args.conv_dropout,
        dense_units=args.dense_units,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_layers=args.lstm_layers,
        lstm_dropout=args.lstm_dropout,
        final_dropout=args.final_dropout,
    )
    model.to(device)

    if args.class_weights:
        if len(args.class_weights) != len(dataset.class_names):
            raise ValueError(
                "Number of class weights must equal the number of labels in the dataset"
            )
        weights = torch.tensor(args.class_weights, dtype=torch.float32, device=device)
    else:
        weights = None

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        verbose=True,
    )

    best_val_loss = math.inf
    best_epoch = -1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1 = evaluate_model(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:03d}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} train_f1={train_f1:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_f1={val_f1:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    if best_epoch == -1:
        raise RuntimeError("Training failed to produce a valid checkpoint")

    print(f"Best validation loss {best_val_loss:.4f} achieved at epoch {best_epoch}")

    model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
    model.to(device)

    if test_loader is not None:
        test_loss, test_acc, test_f1 = evaluate_model(
            model, test_loader, criterion, device
        )
        print(
            "Test metrics: "
            f"loss={test_loss:.4f} acc={test_acc:.3f} macro_f1={test_f1:.3f}"
        )

    config = {
        "data_path": str(Path(args.data).resolve()),
        "sequence_column": args.sequence_column,
        "label_column": args.label_column,
        "feature_columns": dataset.feature_columns,
        "label_encoder": dataset.label_encoder,
        "class_names": dataset.class_names,
        "label_aggregation": label_aggregation,
        "window_size": window_size,
        "window_stride": window_stride,
        "normalization": dataset.normalization_stats.as_dict(),
        "model": {
            "conv_channels": list(args.conv_channels),
            "conv_kernel_size": args.conv_kernel_size,
            "conv_dropout": args.conv_dropout,
            "dense_units": args.dense_units,
            "lstm_hidden_size": args.lstm_hidden_size,
            "lstm_layers": args.lstm_layers,
            "lstm_dropout": args.lstm_dropout,
            "final_dropout": args.final_dropout,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "class_weights": args.class_weights,
        },
    }

    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved configuration to {output_dir / 'config.json'}")


def evaluate_command(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    (
        model,
        label_encoder,
        normalization_stats,
        config,
    ) = load_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        config_path=Path(args.config),
        device=device,
    )

    feature_columns = args.feature_columns or config["feature_columns"]
    sequence_column = args.sequence_column or config.get("sequence_column")
    label_column = args.label_column or config.get("label_column")
    window_size = args.window_size or config.get("window_size", 256)
    window_stride = args.window_stride or config.get("window_stride", 64)
    label_aggregation = args.label_aggregation or config.get("label_aggregation", "majority")

    dataset = SequenceWindowDataset(
        csv_path=Path(args.data),
        sequence_column=sequence_column,
        label_column=label_column,
        feature_columns=feature_columns,
        window_size=window_size,
        window_stride=window_stride,
        normalization_stats=normalization_stats,
        label_encoder=label_encoder,
        label_aggregation=label_aggregation,
        require_labels=True,
    )

    inverse_label_encoder = {v: k for k, v in label_encoder.items()}
    criterion = nn.CrossEntropyLoss()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    loss, acc, f1 = evaluate_model(model, dataloader, criterion, device)
    print(
        f"Evaluation metrics: loss={loss:.4f} acc={acc:.3f} macro_f1={f1:.3f}"
    )

    if args.output_predictions:
        predictions: List[Dict[str, float]] = []
        with torch.no_grad():
            for windows, _ in dataloader:
                logits = model(windows.to(device))
                probs = torch.softmax(logits, dim=1)
                pred = probs.argmax(dim=1).cpu().numpy()
                for prob_row, pred_label in zip(probs.cpu().numpy(), pred):
                    predictions.append(
                        {
                            "predicted_label": inverse_label_encoder[int(pred_label)],
                            **{
                                f"prob_class_{inverse_label_encoder[idx]}": float(prob)
                                for idx, prob in enumerate(prob_row)
                            },
                        }
                    )

        pd.DataFrame(predictions).to_csv(args.output_predictions, index=False)
        print(f"Saved per-window predictions to {args.output_predictions}")


def predict_command(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    (
        model,
        label_encoder,
        normalization_stats,
        config,
    ) = load_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        config_path=Path(args.config),
        device=device,
    )

    feature_columns = args.feature_columns or config["feature_columns"]
    sequence_column = args.sequence_column or config.get("sequence_column")
    label_column = args.label_column or config.get("label_column")
    window_size = args.window_size or config.get("window_size", 256)
    window_stride = args.window_stride or config.get("window_stride", 64)
    label_aggregation = args.label_aggregation or config.get("label_aggregation", "majority")

    dataset = SequenceWindowDataset(
        csv_path=Path(args.data),
        sequence_column=sequence_column,
        label_column=label_column,
        feature_columns=feature_columns,
        window_size=window_size,
        window_stride=window_stride,
        normalization_stats=normalization_stats,
        label_encoder=label_encoder,
        label_aggregation=label_aggregation,
        require_labels=False,
    )

    inverse_label_encoder = {v: k for k, v in label_encoder.items()}
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    predictions: List[Dict[str, float]] = []

    with torch.no_grad():
        for windows, labels in dataloader:
            logits = model(windows.to(device))
            probs = torch.softmax(logits, dim=1)
            for prob_row in probs.cpu().numpy():
                predictions.append(
                    {
                        "predicted_label": inverse_label_encoder[int(prob_row.argmax())],
                        **{
                            f"prob_class_{inverse_label_encoder[idx]}": float(prob)
                            for idx, prob in enumerate(prob_row)
                        },
                    }
                )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PyTorch implementation of the intoxication classifier"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--data", type=str, required=True, help="Path to CSV data")
    base_parser.add_argument(
        "--sequence-column",
        type=str,
        default="sequence_id",
        help="Name of the column that identifies each sequence",
    )
    base_parser.add_argument(
        "--label-column",
        type=str,
        default="groundtruth+phase+",
        help="Name of the column containing labels (per sequence or sample)",
    )
    base_parser.add_argument(
        "--label-aggregation",
        type=str,
        choices=sorted(SequenceWindowDataset.VALID_LABEL_AGGREGATIONS),
        default=None,
        help=(
            "How to derive one label per sequence when multiple labels appear. "
            "Defaults to 'majority' during training and to the training-time "
            "setting when evaluating/predicting."
        ),
    )
    base_parser.add_argument(
        "--feature-columns",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of feature columns. Defaults to all non-label columns.",
    )
    base_parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help=(
            "Number of time steps per window. Defaults to 256 during training or "
            "the value stored in the checkpoint configuration."
        ),
    )
    base_parser.add_argument(
        "--window-stride",
        type=int,
        default=None,
        help=(
            "Step between successive windows. Defaults to 64 during training or "
            "the value stored in the checkpoint configuration."
        ),
    )
    base_parser.add_argument(
        "--batch-size", type=int, default=64, help="Mini-batch size"
    )
    base_parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force execution on CPU even if CUDA is available",
    )

    train_parser = subparsers.add_parser("train", parents=[base_parser])
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory where checkpoints and configs will be stored",
    )
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--val-ratio", type=float, default=0.1)
    train_parser.add_argument("--test-ratio", type=float, default=0.1)
    train_parser.add_argument(
        "--seed", type=int, default=17, help="Random seed for reproducibility"
    )
    train_parser.add_argument(
        "--class-weights",
        type=float,
        nargs="*",
        default=None,
        help="Optional weighting for each class (ordered by label id)",
    )
    train_parser.add_argument(
        "--conv-channels",
        type=int,
        nargs="*",
        default=(64, 128, 128),
        help="Number of filters in each convolutional layer",
    )
    train_parser.add_argument(
        "--conv-kernel-size", type=int, default=5, help="Kernel width for Conv1d layers"
    )
    train_parser.add_argument(
        "--conv-dropout", type=float, default=0.1, help="Dropout between convolutional layers"
    )
    train_parser.add_argument(
        "--dense-units", type=int, default=128, help="Units in the time-distributed dense layer"
    )
    train_parser.add_argument(
        "--lstm-hidden-size", type=int, default=128, help="Hidden size of the LSTM"
    )
    train_parser.add_argument(
        "--lstm-layers", type=int, default=1, help="Number of stacked LSTM layers"
    )
    train_parser.add_argument(
        "--lstm-dropout", type=float, default=0.3, help="Dropout between LSTM layers"
    )
    train_parser.add_argument(
        "--final-dropout", type=float, default=0.3, help="Dropout before the classifier"
    )

    evaluate_parser = subparsers.add_parser("evaluate", parents=[base_parser])
    evaluate_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the saved model checkpoint"
    )
    evaluate_parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration JSON"
    )
    evaluate_parser.add_argument(
        "--output-predictions",
        type=str,
        default=None,
        help="Optional path for storing per-window predictions as CSV",
    )

    predict_parser = subparsers.add_parser("predict", parents=[base_parser])
    predict_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the saved model checkpoint"
    )
    predict_parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration JSON"
    )
    predict_parser.add_argument(
        "--output", type=str, required=True, help="Destination CSV for predictions"
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        train_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "predict":
        predict_command(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

