from src.config.model import ModelConfig
from src.config.dataset import DatasetConfig
from src.config.downloader import DownloaderConfig
from src.schemas.training_result import TrainingResult
from src.utils.downloader import DatasetDownloader
from src.utils.dataset import Dataset
from src.utils.trainer import Trainer
from pathlib import Path
import os


class TransferPipeline:
    def run(
        self,
        download_cfg: DownloaderConfig,
        split_cfg: DatasetConfig,
        train_cfg: ModelConfig,
        should_delete: bool = True,
    ) -> TrainingResult:
        dataset_dir = Path(download_cfg.unzip_path)

        if not dataset_dir.exists():
            DatasetDownloader(
                api_key=download_cfg.api_key,
                project_id=download_cfg.project_id,
                url=download_cfg.url,
            ).download(
                zip_path=download_cfg.zip_path,
                unzip_path=download_cfg.unzip_path,
            )

        if not (dataset_dir / "train").exists():
            Dataset().split(
                data_dir=split_cfg.data_dir,
                seed=split_cfg.seed,
                train_ratio=split_cfg.train_ratio,
                val_ratio=split_cfg.val_ratio,
            )
        metrics = Trainer(
            config_path=train_cfg.config,
            data=train_cfg.dataset,
            model=train_cfg.model,
            overrides=train_cfg.overrides,
        ).train()
        if should_delete:
            Dataset().cleanup(
                data_path=download_cfg.unzip_path, zip_file=download_cfg.zip_path
            )

        return TrainingResult(
            best_model=Path(metrics.save_dir) / "weights" / "best.pt",
            last_model=Path(metrics.save_dir) / "weights" / "last.pt",
            save_dir=Path(metrics.save_dir),
            dataset_yaml=Path(train_cfg.dataset),
            precision=metrics.box.mp,
            recall=metrics.box.mr,
            map50=metrics.box.map50,
            map5095=metrics.box.map,
        )


if __name__ == "__main__":
    download_cfg = DownloaderConfig(
        api_key=os.getenv("LABEL_STUDIO_API_KEY"),
        url=os.getenv("LABEL_STUDIO_URL"),
        project_id=13,
        zip_path="dataset/source1/source.zip",
        unzip_path="dataset/source1",
    )
    split_cfg = DatasetConfig(
        data_dir="dataset/source1", seed=4242323423424, train_ratio=0.8, val_ratio=0.2
    )
    train_cfg = ModelConfig(
        model=Path("/home/crowdbot/YOLO/erd-detector/runs/obb/train-4/weights/best.pt"),
        dataset=Path("dataset/source1/data.yaml"),
        config=Path("configs/train/train.yaml"),
    )

    train = TransferPipeline()
    train.run(download_cfg, split_cfg, train_cfg)
