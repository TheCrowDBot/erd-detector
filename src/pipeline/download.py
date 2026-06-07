from src.utils.downloader import DatasetDownloader
from dotenv import load_dotenv
import os

load_dotenv()

def run(project_id: int) -> str:

    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    url = os.getenv("LABEL_STUDIO_URL")
    if not api_key:
        raise ValueError("LABEL_STUDIO_API_KEY is not set")

    if not url:
        raise ValueError("LABEL_STUDIO_URL is not set")

    dataset = DatasetDownloader(
        api_key=api_key,
        url=url,
        project_id=int(project_id)
    )

    dataset_path = dataset.download(
        zip_path='source/dataset.zip',
        unzip_path='source'
    )
    
    return dataset_path

if __name__ == "__main__": 
    run(project_id=4)