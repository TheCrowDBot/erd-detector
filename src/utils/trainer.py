from ultralytics.utils import DEFAULT_CFG_DICT
from pathlib import Path
from ultralytics import YOLO
import yaml



class Trainer:
    def __init__(
            self,
            config_path: Path,
            data: Path,
            model_path: str = "models",
            model_name: str = "yolo26l-obb.pt",
        ):
            self.config_path = config_path
            self.model_path = Path(model_path)
            self.model_name = model_name
            self.model: YOLO = YOLO(self.model_path / self.model_name)
            self.data = data

            

    def _validate_yaml(self, params):
        print(params)
        valid_keys = DEFAULT_CFG_DICT.keys()
        unknown = set(params.keys()) - valid_keys

        if unknown:
            raise ValueError(
                f"Unknown Ultralytics config keys: {sorted(unknown)}"
            )

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
    
    def train(self): 
        config = self._flatten_yaml(self._load_yaml(self.config_path))
        if not config:
            raise Exception("Training Parameters not loaded")
        self._validate_yaml(config)

        self.results = self.model.train(data=str(self.data), **config)
        return self.results