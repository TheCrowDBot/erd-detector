from dataclasses import dataclass
from src.config.model import ModelConfig
from src.config.dataset import DatasetConfig
from src.config.downloader import DownloaderConfig
from pathlib import Path
import yaml
import os


@dataclass
class ExperimentConfig:
    train_download: DownloaderConfig
    train_dataset: DatasetConfig
    train_model: ModelConfig

    transfer_download: DownloaderConfig
    transfer_dataset: DatasetConfig
    transfer_model: ModelConfig

    evaluation_download: DownloaderConfig
    evaluation_dataset: DatasetConfig
    evaluation_model: ModelConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(
            train_download=DownloaderConfig(
                **data["train"]["download"],
                api_key=os.getenv("LABEL_STUDIO_API_KEY"),
                url=os.getenv("LABEL_STUDIO_URL")
            ),
            train_dataset=DatasetConfig(**data["train"]["dataset"]),
            train_model=ModelConfig(**data["train"]["model"]),
            transfer_download=DownloaderConfig(
                **data["transfer"]["download"],
                api_key=os.getenv("LABEL_STUDIO_API_KEY"),
                url=os.getenv("LABEL_STUDIO_URL")
            ),
            transfer_dataset=DatasetConfig(**data["transfer"]["dataset"]),
            transfer_model=ModelConfig(**data["transfer"]["model"]),
            evaluation_download=DownloaderConfig(
                **data["evaluation"]["download"],
                api_key=os.getenv("LABEL_STUDIO_API_KEY"),
                url=os.getenv("LABEL_STUDIO_URL")
            ),
            evaluation_dataset=DatasetConfig(**data["evaluation"]["dataset"]),
            evaluation_model=ModelConfig(**data["evaluation"]["model"]),
        )
