import os
import zipfile
import random
from pathlib import Path
import shutil
import yaml
from tqdm import tqdm

class Dataset:
    
    def _move_pairs(self, pairs, split, images_dir, labels_dir):
        for img_path, label_path in tqdm(
            pairs,
            desc=f"Moving {split}",
            unit="file"
        ):
            shutil.move(
                str(img_path),
                str(images_dir / split / img_path.name)
            )
            shutil.move(
                str(label_path),
                str(labels_dir / split / label_path.name)
            )

    def _get_dataset_classes(self, classes_file): 
        classes: list[str]
        with open(classes_file, "r") as f:
            classes = [line.strip() for line in f if line.strip()]
        
        return classes
    
    def _create_yaml(self, base_path, classes): 
        config = {
            "path": str(base_path.absolute()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "channels": 1, # this is for black and white images, we are training only on B&W images
            "nc": len(classes),
            "names": classes
        }
        with open(base_path / "data.yaml", "w") as f:
            try:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            except Exception as e: 
                print(f"Error creating YAML: {e}")

    
    def _get_image_label_pairs(
        self,
        images_dir: Path,
        labels_dir: Path,
    ) -> list[tuple[Path, Path]]: 
        pairs = []
        for img_path in images_dir.iterdir():
            if img_path.is_file():
                label_path = labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    pairs.append((img_path, label_path))
        return pairs

    def split(self, train_ratio: float, val_ratio: float, seed: int, data_dir: str):
        """
        Split the dataset into train, validation, and test subsets.

        Args:
            train_ratio: Fraction of samples assigned to training.
            val_ratio: Fraction of samples assigned to validation.
            seed: Random seed used for reproducible shuffling.
            data_dir: Dataset directory containing images and labels.

        The test ratio is computed as:
            1 - (train_ratio + val_ratio)
        """

        if train_ratio + val_ratio > 1:
            raise ValueError("train_ratio + val_ratio must be less than 1")
        base_path = Path(data_dir or self.unzip_path)
        images_dir: Path = base_path / "images"
        labels_dir: Path = base_path / "labels"
        classes_file = base_path / "classes.txt"
        if not classes_file.exists():
            raise FileNotFoundError(
                f"Missing classes file: {classes_file}"
            )

        classes = self._get_dataset_classes(classes_file)
        pairs = self._get_image_label_pairs(images_dir, labels_dir)

        random.seed(seed) # For reproducible splits
        random.shuffle(pairs)

        total = len(pairs)
        
        # Compute split boundaries
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_pairs = pairs[:train_end]
        val_pairs = pairs[train_end:val_end]
        
        # Remaining samples are assigned to the test set.
        test_pairs = pairs[val_end:]

        for split in ["train", "val", "test"]: 
            (images_dir / split).mkdir(parents=True, exist_ok=True)
            (labels_dir / split).mkdir(parents=True, exist_ok=True)
        
        self._move_pairs(train_pairs, "train", images_dir, labels_dir)
        self._move_pairs(val_pairs, "val", images_dir, labels_dir)
        self._move_pairs(test_pairs, "test", images_dir, labels_dir)

        self._create_yaml(base_path, classes)
        print(f"✓ Train: {len(train_pairs)} images")
        print(f"✓ Val: {len(val_pairs)} images")
        print(f"✓ Test: {len(test_pairs)} images")


    def _remove_files(self, path): 
        if os.path.exists(path): 
            try: 
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.unlink(path)
                print(f"Successfully removed {path}")
            except Exception as e:
                print(f"Error removing {path}: {e}")

    def cleanup(self, should_remove_zip: bool = True, should_remove_dataset: bool = True):
        if should_remove_dataset: 
            self._remove_files(self.unzip_path)
        
        if should_remove_zip:
            self._remove_files(self.zip_file)

