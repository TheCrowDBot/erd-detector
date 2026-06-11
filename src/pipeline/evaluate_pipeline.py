from src.config.model import ModelConfig
from src.config.dataset import DatasetConfig
from src.config.downloader import DownloaderConfig
from src.schemas.evaluation_results import EvaluationResult
from src.utils.downloader import DatasetDownloader
from src.utils.dataset import Dataset
from src.utils.trainer import Trainer
import os
from pathlib import Path


class EvaluatePipeline:
    def run(
        self,
        download_cfg: DownloaderConfig,
        split_cfg: DatasetConfig,
        model_cfg: ModelConfig,
    ) -> EvaluationResult:
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

        eval_result = Trainer(
            config_path=model_cfg.config, data=model_cfg.dataset, model=model_cfg.model
        ).evaluate()
        Dataset().cleanup(
            data_path=download_cfg.unzip_path, zip_file=download_cfg.zip_path
        )

        return eval_result


if __name__ == "__main__":
    download_cfg = DownloaderConfig(
        api_key=os.getenv("LABEL_STUDIO_API_KEY"),
        url=os.getenv("LABEL_STUDIO_URL"),
        project_id=13,
        zip_path="dataset/eval/source.zip",
        unzip_path="dataset/eval",
    )
    split_cfg = DatasetConfig(
        data_dir="dataset/eval", seed=23231, train_ratio=0.0, val_ratio=0.0
    )
    train_cfg = ModelConfig(
        model=Path("/home/crowdbot/YOLO/erd-detector/runs/obb/train-5/weights/best.pt"),
        dataset=Path("dataset/eval/data.yaml"),
        config=Path("configs/val/val.yaml"),
    )

    evaluate = EvaluatePipeline()
    evaluate.run(download_cfg, split_cfg, train_cfg)
