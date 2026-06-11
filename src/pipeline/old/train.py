from src.utils.trainer import Trainer
from src.config.model import ModelConfig
from pathlib import Path


def run(cfg: ModelConfig):
    trainer = Trainer(model=cfg.model, data=cfg.dataset, config_path=cfg.config)

    results = trainer.train()
    path = Path(results.save_dir) / "weights" / "best.pt"
    return path


# if __name__ == "__main__":
#     run()
