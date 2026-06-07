from src.utils.dataset import Dataset


def run():
    dataset = Dataset()
    dataset.split(train_ratio=0.8, val_ratio=.1, seed=49939, data_dir='source')




if __name__ == "__main__": 
    run()