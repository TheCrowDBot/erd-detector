from dataclasses import dataclass


@dataclass
class HyperparameterConfig:
    iterations: int = 50

    space: dict[str, tuple] | None = None
