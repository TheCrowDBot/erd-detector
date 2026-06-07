from src.pipeline.download import run as download_dataset
from src.pipeline.split import run as split_dataset
from src.pipeline.train import run as train_model

def main():
    print("Downloading dataset...")
    download_dataset(project_id=4)

    print("Splitting dataset...")
    split_dataset()

    print("Training Model")
    train_model()

    print("Pipeline complete.")


if __name__ == "__main__":
    main()