from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    config: Path

    model: Path | None = None
    dataset: Path | None = None
