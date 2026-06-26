from dataclasses import dataclass

from src.config.downloader import DownloaderConfig
from src.config.dataset import DatasetConfig
from src.config.model import ModelConfig


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
