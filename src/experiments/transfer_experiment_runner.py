from src.utils.experiment_config_loader import ExperimentConfigLoader
from src.schemas.experiment_result import ExperimentResult
from copy import deepcopy


from src.pipeline.transfer_pipeline import TransferPipeline
from src.pipeline.evaluate_pipeline import EvaluatePipeline


class TransferExperimentRunner:

    def run(
        self,
        cfg: ExperimentConfigLoader,
        train_result,
    ) -> ExperimentResult:

        cfg = deepcopy(cfg)

        cfg.transfer_model.model = train_result.best_model
        cfg.transfer_model.dataset = train_result.dataset_yaml

        transfer_result = TransferPipeline().run(
            download_cfg=cfg.transfer_download,
            split_cfg=cfg.transfer_dataset,
            train_cfg=cfg.transfer_model,
            should_delete=False,
        )

        cfg.evaluation_model.model = transfer_result.best_model

        evaluation_result = EvaluatePipeline().run(
            download_cfg=cfg.evaluation_download,
            split_cfg=cfg.evaluation_dataset,
            model_cfg=cfg.evaluation_model,
            should_delete=False,
        )

        return ExperimentResult(
            train_result=train_result,
            transfer_result=transfer_result,
            evaluation_result=evaluation_result,
        )
