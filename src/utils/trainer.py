from ultralytics.utils import DEFAULT_CFG_DICT
from pathlib import Path
from ultralytics import YOLO
import yaml
import torch
import os


class Trainer:

    def __init__(
        self,
        config_path: Path,
        data: Path,
        model: Path = Path("models/yolo26l-obb.pt"),
        overrides: dict | None = None,
    ):
        self.config_path = config_path
        self.model: YOLO = YOLO(model, verbose=True)
        self.data = data
        self.overrides = overrides or {}

    def _validate_yaml(self, params):
        valid_keys = DEFAULT_CFG_DICT.keys()
        unknown = set(params.keys()) - valid_keys

        if unknown:
            raise ValueError(f"Unknown Ultralytics config keys: {sorted(unknown)}")

    def _flatten_yaml(self, config: dict) -> dict:
        flat = {}
        for section in config.values():
            if isinstance(section, dict):
                flat.update(section)
        return flat

    def _load_yaml(self, path: str | Path):
        print(f"Loading YAML from: {path}")
        with open(path, "r") as stream:
            try:
                params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                params = {}
        print("YAML Loaded")
        return params

    def _clean_vram(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.empty_cache()

    def evaluate(self):
        self._clean_vram()
        config = self._flatten_yaml(self._load_yaml(self.config_path))
        if not config:
            raise Exception("Validation Parameters not loaded")
        self._validate_yaml(config)
        self.results = self.model.val(data=str(self.data), **config)
        return self.results

    def train(self):
        self._clean_vram()
        config = self._flatten_yaml(self._load_yaml(self.config_path))
        if not config:
            raise Exception("Training Parameters not loaded")

        # Apply runtime overrides
        config.update(self.overrides)
        self._validate_yaml(config)
        print(config)
        self.results = self.model.train(data=str(self.data), **config)
        return self.results
