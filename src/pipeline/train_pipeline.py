from src.config.model import ModelConfig
from src.config.dataset import DatasetConfig
from src.config.downloader import DownloaderConfig
from src.schemas.training_result import TrainingResult
from src.utils.downloader import DatasetDownloader
from src.utils.dataset import Dataset
from src.utils.trainer import Trainer
from pathlib import Path


class TrainPipeline:
    def run(
        self,
        download_cfg: DownloaderConfig,
        split_cfg: DatasetConfig,
        train_cfg: ModelConfig,
    ) -> TrainingResult:
        DatasetDownloader(
            api_key=download_cfg.api_key,
            project_id=download_cfg.project_id,
            url=download_cfg.url,
        ).download(zip_path=download_cfg.zip_path, unzip_path=download_cfg.unzip_path)

        Dataset().split(
            data_dir=split_cfg.data_dir,
            seed=split_cfg.seed,
            train_ratio=split_cfg.train_ratio,
            val_ratio=split_cfg.val_ratio,
        )
        train_result = Trainer(
            config_path=train_cfg.config, data=train_cfg.dataset, model=train_cfg.model
        ).train()
        Dataset().cleanup(
            data_path=download_cfg.unzip_path, zip_file=download_cfg.zip_path
        )
        return train_result


if __name__ == "__main__":
    download_cfg = DownloaderConfig(
        project_id=4, zip_path="dataset/source/source.zip", unzip_path="dataset/source"
    )
    split_cfg = DatasetConfig(
        data_dir="dataset/source", seed=39031290, train_ratio=0.8, val_ratio=0.2
    )
    train_cfg = ModelConfig(
        model=Path("models/yolo26l-obb.pt"),
        dataset=Path("dataset/source/data.yaml"),
        config=Path("configs/train/train.yaml"),
    )

    train = TrainPipeline()
    train.run(download_cfg, split_cfg, train_cfg)
