import os
from dotenv import load_dotenv

os.environ["YOLO_VERBOSE"] = "True"

load_dotenv()
from pathlib import Path
from src.pipeline.train_pipeline import TrainPipeline
from src.pipeline.transfer_pipeline import TransferPipeline
from src.pipeline.evaluate_pipeline import EvaluatePipeline
from src.config.model import ModelConfig
from src.config.dataset import DatasetConfig
from src.config.downloader import DownloaderConfig


def main():
    download_cfg = DownloaderConfig(
        project_id=4,
        zip_path="dataset/source/source.zip",
        unzip_path="dataset/source",
        api_key=os.getenv("LABEL_STUDIO_API_KEY"),
        url=os.getenv("LABEL_STUDIO_URL"),
    )
    split_cfg = DatasetConfig(
        data_dir="dataset/source",
        seed=23789234567912445,
        train_ratio=0.8,
        val_ratio=0.2,
    )
    train_cfg = ModelConfig(
        model=Path("models/yolo26l-obb.pt"),
        dataset=Path("dataset/source/data.yaml"),
        config=Path("configs/train/train.yaml"),
    )

    print("starting first train")

    model = TrainPipeline().run(
        download_cfg=download_cfg, split_cfg=split_cfg, train_cfg=train_cfg
    )
    model_path = Path(model.save_dir / "weights" / "best.pt")

    # Transfer learning

    download_cfg = DownloaderConfig(
        project_id=13,
        zip_path="dataset/source/source.zip",
        unzip_path="dataset/source",
        api_key=os.getenv("LABEL_STUDIO_API_KEY"),
        url=os.getenv("LABEL_STUDIO_URL"),
    )
    split_cfg = DatasetConfig(
        data_dir="dataset/source", seed=4292345798412525, train_ratio=0.8, val_ratio=0.2
    )
    train_cfg = ModelConfig(
        model=model_path,
        dataset=Path("dataset/source/data.yaml"),
        config=Path("configs/train/transfer-learning.yaml"),
    )

    print("Start transfer learning")

    transfer_learn_model = TransferPipeline().run(
        download_cfg=download_cfg, split_cfg=split_cfg, train_cfg=train_cfg
    )

    transfer_learn_model_path = Path(
        transfer_learn_model.save_dir / "weights" / "best.pt"
    )

    download_cfg = DownloaderConfig(
        api_key=os.getenv("LABEL_STUDIO_API_KEY"),
        url=os.getenv("LABEL_STUDIO_URL"),
        project_id=13,
        zip_path="dataset/eval/source.zip",
        unzip_path="dataset/eval",
    )
    split_cfg = DatasetConfig(
        data_dir="dataset/eval", seed=94867326, train_ratio=0.0, val_ratio=0.0
    )
    eval_cfg = ModelConfig(
        model=transfer_learn_model_path,
        dataset=Path("dataset/eval/data.yaml"),
        config=Path("configs/val/val.yaml"),
    )

    EvaluatePipeline().run(
        download_cfg=download_cfg, split_cfg=split_cfg, model_cfg=eval_cfg
    )


if __name__ == "__main__":
    main()
