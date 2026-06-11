import os
from pathlib import Path
from src.utils.downloader import DatasetDownloader
from src.config.downloader import DownloaderConfig
from dotenv import load_dotenv


load_dotenv()


def run(cfg: DownloaderConfig) -> str:

    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    url = os.getenv("LABEL_STUDIO_URL")
    if not api_key:
        raise ValueError("LABEL_STUDIO_API_KEY is not set")

    if not url:
        raise ValueError("LABEL_STUDIO_URL is not set")

    dataset = DatasetDownloader(api_key=api_key, url=url, project_id=cfg.project_id)

    dataset_path = dataset.download(zip_path=cfg.zip_path, unzip_path=cfg.unzip_path)

    return dataset_path


if __name__ == "__main__":
    download_cfg = DownloaderConfig(
        project_id=4, zip_path=Path("source_1/dataset.zip"), unzip_path=Path("source_1")
    )
    run(download_cfg)
