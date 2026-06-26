from src.schemas.hyperparameter_results import HyperparameterResult
from src.utils.dataset import Dataset
from src.utils.downloader import DatasetDownloader
from src.utils.tuner import Tuner
from src.config.dataset import DatasetConfig
from src.config.downloader import DownloaderConfig
from src.config.model import ModelConfig
from src.config.hyperparameter import HyperparameterConfig
from pathlib import Path


class HyperparameterPipeline:

    def run(
        self,
        download_cfg: DownloaderConfig,
        split_cfg: DatasetConfig,
        model_cfg: ModelConfig,
        hyper_cfg: HyperparameterConfig,
    ) -> HyperparameterResult:
        dataset_dir = Path(download_cfg.unzip_path)

        if not dataset_dir.exists():
            DatasetDownloader(
                api_key=download_cfg.api_key,
                project_id=download_cfg.project_id,
                url=download_cfg.url,
            ).download(
                zip_path=download_cfg.zip_path,
                unzip_path=download_cfg.unzip_path,
            )

        if not (dataset_dir / "train").exists():
            Dataset().split(
                data_dir=split_cfg.data_dir,
                seed=split_cfg.seed,
                train_ratio=split_cfg.train_ratio,
                val_ratio=split_cfg.val_ratio,
            )

        results = Tuner().tune(
            model_path=model_cfg.model,
            dataset=model_cfg.dataset,
            config=model_cfg.config,
            iterations=hyper_cfg.iterations,
            space=hyper_cfg.space,
        )

        return HyperparameterResult(best_params=results, best_model=results)
