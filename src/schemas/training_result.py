from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingResult:
    best_model: Path
    last_model: Path

    save_dir: Path

    epochs: int

    precision: float
    recall: float

    map50: float
    map5095: float
