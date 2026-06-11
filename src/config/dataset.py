from dataclasses import dataclass


@dataclass
class DatasetConfig:
    train_ratio: float
    val_ratio: float
    seed: int
    data_dir: str
