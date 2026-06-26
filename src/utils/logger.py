import json
from pathlib import Path


class SweepLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, sweep_result):
        record = {
            "parameters": sweep_result.parameters,
            "map5095": sweep_result.experiment_result.evaluation_result.map5095,
        }

        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")
