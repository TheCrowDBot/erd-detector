from pathlib import Path
from typing import Dict

from ultralytics import YOLO

from src.schemas.evaluation_results import EvaluationResult
import yaml


class Evaluator:
    def __init__(
        self,
        model: str | Path,
        dataset: str | Path,
        config: str | Path,
    ):
        self.model = YOLO(str(model), verbose=True)
        self.dataset = Path(dataset)
        self.config = Path(config)

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

    def evaluate(self) -> EvaluationResult:
        """
        Run YOLO validation.
        """

        config = self._flatten_yaml(self._load_yaml(self.config))
        if not config:
            raise Exception("Training Parameters not loaded")

        results = self.model.val(data=str(self.dataset), **config)

        per_class = self.class_metrics(results)

        return EvaluationResult(
            precision=float(results.box.mp),
            recall=float(results.box.mr),
            map50=float(results.box.map50),
            map5095=float(results.box.map),
            per_class=per_class,
            save_dir=Path(results.save_dir),
        )

    def save_metrics(
        self,
        result: EvaluationResult,
        output_file: str | Path,
    ) -> None:
        """
        Save metrics to JSON.
        """

        import json

        metrics = {
            "precision": result.precision,
            "recall": result.recall,
            "mAP50": result.map50,
            "mAP50-95": result.map5095,
            "per_class": result.per_class,
        }

        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=4)

    def save_report(
        self,
        result: EvaluationResult,
        output_file: str | Path,
    ) -> None:
        """
        Save human-readable report.
        """

        report = f"""
            YOLO Evaluation Report
            ======================

            Precision : {result.precision:.4f}
            Recall    : {result.recall:.4f}
            mAP@50    : {result.map50:.4f}
            mAP@50:95 : {result.map5095:.4f}

            Per-Class Metrics
            -----------------
            """

        for cls, metrics in result.per_class.items():
            report += (
                f"\n{cls}\n"
                f"  Precision: {metrics['precision']:.4f}\n"
                f"  Recall   : {metrics['recall']:.4f}\n"
                f"  mAP50    : {metrics['map50']:.4f}\n"
            )

        with open(output_file, "w") as f:
            f.write(report)

    def confusion_matrix(self, results) -> Dict:
        """
        Return confusion matrix if available.
        """

        try:
            return results.confusion_matrix.matrix.tolist()
        except Exception:
            return {}

    def class_metrics(self, results) -> Dict:
        """
        Extract per-class metrics.
        """

        metrics = {}

        try:
            names = results.names

            for idx, cls_name in names.items():
                metrics[cls_name] = {
                    "precision": float(results.box.p[idx]),
                    "recall": float(results.box.r[idx]),
                    "map50": float(results.box.ap50[idx]),
                    "map5095": float(results.box.ap[idx]),
                }

        except Exception:
            pass

        return metrics
