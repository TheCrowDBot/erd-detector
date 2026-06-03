from label_studio_sdk import Client
import os
import zipfile

class DatasetPipeline:

    EXPORT_TYPE: str = "YOLO_OBB_WITH_IMAGES"
    LsClient: Client = None

    
    def __init__(self, url: str, project_id: int, api_key: str):
        if self.LsClient is not None: 
            return self.LsClient
        
        self.project_id = project_id

        self.LsClient = Client(
            url = url,
            api_key = api_key,
            project_id = self.project_id
        )
        self.LsClient.check_connection()

    def download(sef, unzip_path: str, zip_path: str, should_unzip: bool = True, download_resources: bool = True, export_type: str = EXPORT_TYPE): 
        project = self.LsClient.get_project(self.project_id)
        self.unzip_path = unzip_path
        self.zip_file = zip_path

        if os.path.exists(zip_path):
            os.unlink(zip_path)
        project.export_tasks(
            export_type=export_type,
            download_resources=download_resources,
            export_location=zip_path,
        )

        if should_unzip:
            os.makedirs(unzip_path, exist_ok=True)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(unzip_path)


    def split(self):
        pass

    def cleanup(self):

        def remove_files(): 
            pass
            
        def remove_zip(): 
            pass

        pass

    def run(self):
        self.download()
        self.split()

if __name__ == "__main__": 
    print("Dataset")
