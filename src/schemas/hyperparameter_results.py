from dataclasses import dataclass
from pathlib import Path


@dataclass
class HyperparameterResult:
    best_params: dict

    best_model: Path

    save_dir: Path
