# datasets.py


from typing import Dict, Any, List, Tuple
import torch, os,json
from torchvision.datasets import CocoDetection
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as TF

def xywh_to_xyxy(box: List[float]) -> List[float]:
    x, y, w, h = box
    return [x, y, x + w, y + h]

def resize_target(target: Dict ,original_size, new_size):
    ow, oh = original_size
    nw, nh = new_size
    scale_w, scale_h = nw / ow, nh / oh

    scaled_x1, scaled_y1, scaled_x2, scaled_y2 = 0, 0, 0, 0
    scaled_boxes = []
    for k, v in target.items():
        if k == 'boxes':
            for box in v:
                x1, y1, x2, y2 = box
                scaled_x1, scaled_y1, scaled_x2, scaled_y2 = x1 * scale_w, y1 * scale_h, x2 * scale_w, y2 * scale_h
                scaled_boxes.append([scaled_x1, scaled_y1, scaled_x2, scaled_y2])

            target[k] = torch.tensor(scaled_boxes)
        if k == 'area':
            scaled_a = (scaled_x2 - scaled_x1) * (scaled_y2 - scaled_y1)
            target[k] = scaled_a
    return target

class CocoDetWrapped(CocoDetection):
    def __init__(self, img_folder: str, ann_file: str, transforms=None):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img, anno = super().__getitem__(idx)
        # bbox 없거나 crowd인 항목 제거
        anno = [o for o in anno if ('bbox' in o) and (o.get('iscrowd', 0) == 0)]

        boxes, labels, area, iscrowd = [], [], [], []
        for o in anno:
            boxes.append(xywh_to_xyxy(o["bbox"]))
            labels.append(int(o["category_id"]))
            area.append(float(o.get("area", o["bbox"][2] * o["bbox"][3])))
            iscrowd.append(int(o.get("iscrowd", 0)))

        if len(boxes) == 0:
            boxes_t = torch.zeros((0,4), dtype=torch.float32)
            labels_t= torch.zeros((0,),  dtype=torch.int64)
            area_t = torch.zeros((0,),  dtype=torch.float32)
            iscrowd_t = torch.zeros((0,),  dtype=torch.uint8)
        else:
            boxes_t = torch.as_tensor(boxes,  dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            area_t = torch.as_tensor(area,   dtype=torch.float32)
            iscrowd_t = torch.as_tensor(iscrowd,dtype=torch.uint8)

        image_id = torch.tensor([self.ids[idx]], dtype=torch.int64)
        target: Dict[str, Any] = {
            "boxes": boxes_t, "labels": labels_t, "image_id": image_id,
            "area": area_t, "iscrowd": iscrowd_t
        }
        if self._transforms is not None:
            img = self._transforms(img)
            _, orig_h, orig_w = img.shape
            target = resize_target(target, (orig_h, orig_w), (640, 640))
        return img, target





def build_datasets_from_paths(train_img: str, train_ann: str, val_img: str, val_ann: str):

    train_tfms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor()
    ])
    val_tfms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor()
    ])

    return (
        CocoDetWrapped(train_img, train_ann, transforms=train_tfms),
        CocoDetWrapped(val_img,  val_ann,  transforms=val_tfms),
    )
# Test Dataset
class TestDataset(Dataset):
    def __init__(self, img_dir="./test"):
        self.img_dir = Path(img_dir)
        self.images = sorted(list(self.img_dir.glob("*.png")))
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = TF.to_tensor(img)
        target = {"image_id": idx, "file_name": os.path.basename(img_path)}
        return img_tensor, target


def build_test_dataset(img_dir="./test"):
    return TestDataset(img_dir)

