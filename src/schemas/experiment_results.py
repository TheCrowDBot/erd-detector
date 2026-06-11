from src.schemas.evaluation_results import EvaluationResult
from src.schemas.training_result import TrainingResult
from dataclasses import dataclass


@dataclass
class ExperimentResult:
    train_result: TrainingResult
    transfer_result: TrainingResult
    evaluation_result: EvaluationResult
