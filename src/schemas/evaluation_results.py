from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvaluationResult:
    precision: float
    recall: float
    map50: float
    map5095: float

    save_dir: Path

    per_class: dict[str, dict]
