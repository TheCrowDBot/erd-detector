from src.utils.trainer import Trainer
from src.config.model import ModelConfig
from pathlib import Path


def run(cfg: ModelConfig):
    trainer = Trainer(model=cfg.model, config_path=cfg.config, data=cfg.dataset)

    results = trainer.train()
    path = Path(results.save_dir) / "weights" / "best.pt"
    return path


if __name__ == "__main__":
    cfg = ModelConfig(
        model=Path("/home/crowdbot/YOLO/erd-detector/runs/obb/train/weights/best.pt"),
        dataset=Path("/home/crowdbot/YOLO/erd-detector/source_2"),
        config=Path(
            "/home/crowdbot/YOLO/erd-detector/configs/train/transfer-learning.yaml"
        ),
    )
    run(cfg)
