from label_studio_sdk import Client
import os
import zipfile
from tqdm import tqdm

class DatasetDownloader:
    EXPORT_TYPE: str = "YOLO_OBB_WITH_IMAGES"

    def __init__(self, url: str, project_id: int, api_key: str):
        self.project_id = project_id
        self.LsClient = Client(
            url=url,
            api_key=api_key
        )

    
    def _verify_label_studio(self):
        try:
            health = self.LsClient.check_connection()
            print("Connection Successful")
            if health.get("status") != "UP":
                raise RuntimeError(
                    f"Label Studio is unhealthy: {health}"
                )

        except Exception as e:
            raise ConnectionError(
                f"Unable to connect to Label Studio: {e}"
            ) from e

    def download(self, unzip_path: str, zip_path: str, download_resources: bool = True, export_type: str | None = None, extract: bool = True, remove_zip: bool = True) -> str: 
        self._verify_label_studio()
        project = self.LsClient.get_project(self.project_id)
        self.unzip_path = unzip_path
        self.zip_file = zip_path

        if os.path.exists(zip_path):
            os.unlink(zip_path)

        project.export_tasks(
            export_type=export_type or self.EXPORT_TYPE,
            download_resources=download_resources,
            export_location=zip_path,
        )

        if extract:
            os.makedirs(unzip_path, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                members = zip_ref.namelist()

                for member in tqdm(
                    members,
                    desc="Extracting",
                    unit="file"
                ):
                    zip_ref.extract(member, unzip_path)

        if remove_zip: 
            os.unlink(zip_path)
        
        return unzip_path