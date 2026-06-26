from dataclasses import dataclass

from src.schemas.experiment_result import ExperimentResult


@dataclass
class SweepResult:
    parameters: dict[str, object]
    experiment_result: ExperimentResult
