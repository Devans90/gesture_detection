from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gesture_detection.training.dataset import DatasetIndex


@dataclass(slots=True)
class TrainingRunConfig:
    dataset_dir: Path
    run_name: str = "baseline"
    holdout_sessions: int = 1


class GestureTrainingPipeline:
    """Fill in the training, validation, and export steps for your learning exercise."""

    def __init__(self, config: TrainingRunConfig) -> None:
        self.config = config

    def prepare_features(self, dataset: DatasetIndex) -> None:
        raise NotImplementedError(
            "Turn each captured distance series into model-ready tensors or arrays here."
        )

    def train(self, dataset: DatasetIndex) -> None:
        raise NotImplementedError(
            "Train a small PyTorch model here and log metrics/artifacts to MLflow."
        )

    def validate(self, dataset: DatasetIndex) -> None:
        raise NotImplementedError(
            "Validate by holding out sessions, not random samples, and add idle-motion rejection checks here."
        )

    def export_numpy_runtime(self) -> None:
        raise NotImplementedError(
            "Export your trained weights into a format that UserImplementedGestureClassifier can load."
        )
