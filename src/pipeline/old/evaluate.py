from pathlib import Path
from src.utils.evaluator import Evaluator
from src.config.model import ModelConfig


def run(cfg: ModelConfig):
    evaluator = Evaluator(
        model=cfg.model,
        dataset=cfg.dataset,
        config=cfg.config,
    )

    result = evaluator.evaluate()

    evaluator.save_metrics(
        result,
        "metrics.json",
    )

    evaluator.save_report(
        result,
        "evaluation_report.txt",
    )

    print("\nEvaluation Complete")
    print(f"Precision : {result.precision:.4f}")
    print(f"Recall    : {result.recall:.4f}")
    print(f"mAP50     : {result.map50:.4f}")
    print(f"mAP50-95  : {result.map5095:.4f}")
    print(f"Results saved to: {result.save_dir}")


if __name__ == "__main__":
    eval_cfg = ModelConfig(
        model=Path(
            "/home/crowdbot/YOLO/erd-detector/runs/obb/YOLO_HAND_WRITING_STAGE2/train-4/weights/best.pt"
        ),
        dataset=Path("source_3/data.yaml"),
        config=Path("configs/val/val.yaml"),
    )
    run(eval_cfg)
