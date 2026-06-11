from src.utils.dataset import Dataset
from src.config.dataset import DatasetConfig


def run(cfg: DatasetConfig):
    dataset = Dataset()
    dataset.split(
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        data_dir=cfg.data_dir,
        seed=cfg.seed,
    )


if __name__ == "__main__":
    cfg = DatasetConfig(
        train_ratio=0.8, val_ratio=0.2, data_dir="source", seed=239892929912
    )
    run(cfg)
