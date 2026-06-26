from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    config: Path

    model: Path | None = None
    dataset: Path | None = None

    overrides: dict[str, Any] = field(default_factory=dict)
