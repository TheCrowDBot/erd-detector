from pathlib import Path

import yaml

from src.config.downloader import DownloaderConfig
from src.config.dataset import DatasetConfig
from src.config.model import ModelConfig
from src.config.experiment_config import ExperimentConfig


class ExperimentConfigLoader:

    @staticmethod
    def load(
        path: str | Path,
        api_key: str,
        url: str,
    ) -> ExperimentConfig:

        with open(path) as f:
            data = yaml.safe_load(f)

        return ExperimentConfig(
            train_download=DownloaderConfig(
                **data["train"]["download"],
                api_key=api_key,
                url=url,
            ),
            train_dataset=DatasetConfig(**data["train"]["dataset"]),
            train_model=ModelConfig(**data["train"]["model"]),
            transfer_download=DownloaderConfig(
                **data["transfer"]["download"],
                api_key=api_key,
                url=url,
            ),
            transfer_dataset=DatasetConfig(**data["transfer"]["dataset"]),
            # model and dataset will be overwritten by ExperimentRunner
            transfer_model=ModelConfig(
                model=None,
                dataset=None,
                config=data["transfer"]["model"]["config"],
                overrides=data["transfer"]["model"].get(
                    "overrides",
                    {},
                ),
            ),
            evaluation_download=DownloaderConfig(
                **data["evaluation"]["download"],
                api_key=api_key,
                url=url,
            ),
            evaluation_dataset=DatasetConfig(**data["evaluation"]["dataset"]),
            # model and dataset will be overwritten by ExperimentRunner
            evaluation_model=ModelConfig(
                model=None,
                dataset=Path(data["evaluation"]["model"]["dataset"]),
                config=data["evaluation"]["model"]["config"],
                overrides=data["evaluation"]["model"].get(
                    "overrides",
                    {},
                ),
            ),
        )
