from dotenv import load_dotenv
from pathlib import Path
import os
from src.experiments.experiment_loader import ExperimentConfigLoader
from src.experiments.experiment_runner import ExperimentRunner


def main():
    load_dotenv()

    cfg = ExperimentConfigLoader.load(
        Path("experiments/test.yaml"),
        api_key=os.getenv("LABEL_STUDIO_API_KEY"),
        url=os.getenv("LABEL_STUDIO_URL"),
    )

    ExperimentRunner().run(cfg)


if __name__ == "__main__":
    main()
