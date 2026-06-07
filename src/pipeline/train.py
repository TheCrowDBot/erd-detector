from src.utils.trainer import Trainer


def run():
    trainer = Trainer(
        config_path='configs/t1.yml',
        model_path='models',
        model_name="yolo26l-obb.pt",
        data="source"
    )

    trainer.train()

if __name__ == "__main__": 
    run()