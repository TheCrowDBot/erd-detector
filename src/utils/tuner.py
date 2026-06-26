from pathlib import Path
from ultralytics.models import YOLO


class Tuner:

    def tune(
        self,
        model_path: Path,
        dataset: Path,
        config: Path,
        iterations: int,
        space: dict,
    ):
        model = YOLO(model_path, verbose=True)

        return model.tune(
            data=dataset,
            iterations=iterations,
            space=space,
        )
