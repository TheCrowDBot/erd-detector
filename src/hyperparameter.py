from src.pipeline.hyperparameter_pipeline import HyperparameterPipeline
from src.config.dataset import DatasetConfig
from src.config.downloader import DownloaderConfig
from src.config.model import ModelConfig
from src.config.hyperparameter import HyperparameterConfig
import os
from dotenv import load_dotenv


def run():
    load_dotenv()
    downloader_config = DownloaderConfig(
        api_key=os.getenv("LABEL_STUDIO_API_KEY"),
        url=os.getenv("LABEL_STUDIO_URL"),
        project_id=13,
        unzip_path="dataset/hyper",
        zip_path="dataset/hyper/data.zip",
    )

    dataset_config = DatasetConfig(
        train_ratio=0.8, val_ratio=0.2, seed=9349329, data_dir="dataset/hyper"
    )

    model_config = ModelConfig(
        config="/home/crowdbot/YOLO/erd-detector/configs/train/transfer-learning.yaml",
        model="/home/crowdbot/YOLO/erd-detector/runs/obb/YOLO_HAND_WRITING_STAGE2/train-2/weights/best.pt",
        dataset="dataset/hyper/data.yaml",
    )

    hyper_config = HyperparameterConfig(
        iterations=10,
        space={"close_mosaic": (0, 10), "mosaic": (0.4, 0.8), "copy_paste": (0.1, 0.5)},
    )

    HyperparameterPipeline().run(
        download_cfg=downloader_config,
        split_cfg=dataset_config,
        model_cfg=model_config,
        hyper_cfg=hyper_config,
    )


if __name__ == "__main__":
    run()
