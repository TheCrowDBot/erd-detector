from pathlib import Path
import yaml
from src.config.experiment_config import ExperimentConfig
from src.config.model import ModelConfig
from src.config.dataset import DatasetConfig
from src.config.downloader import DownloaderConfig


class ExperimentConfigLoader:

    @staticmethod
    def load(
        path: Path,
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
            transfer_model=ModelConfig(**data["transfer"]["model"]),
            evaluation_download=DownloaderConfig(
                **data["evaluation"]["download"],
                api_key=api_key,
                url=url,
            ),
            evaluation_dataset=DatasetConfig(**data["evaluation"]["dataset"]),
            evaluation_model=ModelConfig(**data["evaluation"]["model"]),
        )
