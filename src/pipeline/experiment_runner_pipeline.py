from copy import deepcopy

from src.utils.experiment_config_loader import ExperimentConfig
from src.pipeline.train_pipeline import TrainPipeline
from src.pipeline.transfer_pipeline import TransferPipeline
from src.pipeline.evaluate_pipeline import EvaluatePipeline
from src.schemas.experiment_result import ExperimentResult


class ExperimentRunnerPipeline:

    def run(
        self,
        cfg: ExperimentConfig,
    ) -> ExperimentResult:

        cfg = deepcopy(cfg)

        train_result = TrainPipeline().run(
            download_cfg=cfg.train_download,
            split_cfg=cfg.train_dataset,
            train_cfg=cfg.train_model,
            
        )

        cfg.transfer_model.model = train_result.best_model

        transfer_result = TransferPipeline().run(
            download_cfg=cfg.transfer_download,
            split_cfg=cfg.transfer_dataset,
            train_cfg=cfg.transfer_model,
        )

        cfg.evaluation_model.model = transfer_result.best_model

        evaluation_result = EvaluatePipeline().run(
            download_cfg=cfg.evaluation_download,
            split_cfg=cfg.evaluation_dataset,
            train_cfg=cfg.evaluation_model,
        )

        return ExperimentResult(
            train_result=train_result,
            transfer_result=transfer_result,
            evaluation_result=evaluation_result,
        )
