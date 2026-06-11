from dataclasses import dataclass
from pathlib import Path


@dataclass
class DownloaderConfig:
    api_key: str
    url: str
    project_id: int
    zip_path: Path
    unzip_path: Path
