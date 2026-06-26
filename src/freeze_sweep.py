from dotenv import load_dotenv
import os

# os.environ["YOLO_VERBOSE"] = "False"
from pathlib import Path
from src.utils.experiment_config_loader import ExperimentConfigLoader
from src.experiments.transfer_sweep_runner import TransferSweepRunner
from src.pipeline.train_pipeline import TrainPipeline
from src.utils.logger import SweepLogger


def main():
    load_dotenv()

    cfg = ExperimentConfigLoader.load(
        path=Path("experiments/experiment.yaml"),
        api_key=os.getenv("LABEL_STUDIO_API_KEY"),
        url=os.getenv("LABEL_STUDIO_URL"),
    )

    #
    # Source train ONCE
    #
    print()
    print("=" * 60)
    print("SOURCE TRAIN")
    print("=" * 60)

    train_result = TrainPipeline().run(
        download_cfg=cfg.train_download,
        split_cfg=cfg.train_dataset,
        train_cfg=cfg.train_model,
        should_delete=False,
    )

    #
    # Sweep transfer stage
    #
    results = TransferSweepRunner().run(
        cfg=cfg,
        train_result=train_result,
        parameters={
            "freeze": [0, 5, 10, 15, 20],
            "lr0": [1e-3, 3e-4, 1e-4],
            "optimizer": ["SGD", "AdamW"],
        },
        logger=SweepLogger(Path("runs/sweeps.jsonl")),
    )

    print()
    print("====== LEADERBOARD ======")

    leaderboard = sorted(
        results,
        key=lambda x: x.experiment_result.evaluation_result.map5095,
        reverse=True,
    )

    for row in leaderboard:

        param_str = "  ".join(f"{k}={v}" for k, v in row.parameters.items())

        print(
            f"{param_str}  "
            f"mAP50-95={row.experiment_result.evaluation_result.map5095:.4f}"
        )


if __name__ == "__main__":
    main()
