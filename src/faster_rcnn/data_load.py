# data_load.py
from pathlib import Path

def data_paths(data_root: str | None = None):
    root = Path(__file__).resolve().parent

    train_img = (root / "train")
    train_ann = (root / "train"/"_annotations.coco.json")
    val_img = (root / "valid")
    val_ann = (root / "valid" / "_annotations.coco.json")
    test_img = (root / "test")

    return train_img, train_ann, val_img, val_ann, test_img


