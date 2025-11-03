import json, os
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

class SSD_Dataset(Dataset):
    def __init__(self, img_dir, annotation, size, transforms=None):
        self.img_dir = img_dir
        self.annotation = annotation
        self.transforms = transforms
        self.output_size = size

        with open(self.annotation, "r", encoding="utf-8") as f:
            annot = json.load(f)

        self.images = annot["images"]
        self.img_name = {img["id"]: img["file_name"] for img in self.images}
        self.img_size = {img["id"]: (img["width"], img["height"]) for img in self.images}

        # json 파일 내에 있는 categories
        self.categories = annot["categories"]
        self.num_classes = len(self.categories)

        # 라벨이 연속으로 됐는지 안전용 확인
        self.id_to_label = {cat["id"]: i for i, cat in enumerate(self.categories)}

        # json 파일 내에 있는 annotations
        self.annotations = annot["annotations"]

        # annotaions를 image로 그룹화
        self.img_to_anns = defaultdict(list)
        for ann in self.annotations:
            self.img_to_anns[ann["image_id"]].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 이미지 불러오기
        img_info = self.images[idx]
        img_id = img_info["id"]
        img_path = os.path.join(self.img_dir, self.img_name[img_id])
        img = Image.open(img_path).convert("RGB")

        anns = self.img_to_anns[img_id]
        bboxes_coco = [ann["bbox"] for ann in anns] # [x,y,w,h]
        # 라벨 연속화
        labels = [self.id_to_label[ann["category_id"]] for ann in anns]

        # albumentations에서 구동하기 위해 numpy로 변환
        img_np = np.array(img)

        transformed = self.transforms(image=img_np, bboxes=bboxes_coco, labels=labels)
        img_t = transformed["image"] # Tensor CxHxW
        bboxes_coco_t = transformed["bboxes"] # 변환된 [x,y,w,h]
        labels_t = transformed["labels"]

        boxes_xyxy = []
        for (x, y, w, h) in bboxes_coco_t:
            boxes_xyxy.append([x, y, x + w, y + h])

        boxes = torch.as_tensor(boxes_xyxy, dtype=torch.float32)
        labels = torch.as_tensor(labels_t, dtype=torch.int64)

        target = {
          "boxes" : boxes,
          "labels" : labels,
          "image_id" : torch.tensor([img_id]),
          "image_name" : self.img_name[img_id],
          "image_size": torch.tensor([self.output_size, self.output_size], dtype=torch.int64)
        }

        return img_t, target


class Test_Dataset(Dataset):
    def __init__(self, image_dir, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_files = sorted([f for f in os.listdir(self.image_dir)
                                  if f.lower().endswith(".png")])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")

        if self.transforms:
            img_t = self.transforms(image=np.array(img), bboxes=[], labels=[])["image"]
        else:
            img_t = ToTensorV2()(image=np.array(img))["image"]

        return img_t

def build_transforms(train=True, size=300): # bbox와 같이 변경해주는 함수
    if train:
        return A.Compose(
            [
                A.Resize(size, size), # 512 혹은 300으로 선택
                A.HorizontalFlip(p=0.5), # 좌우 반전(50% 확률)
                A.RandomBrightnessContrast(p=0.3), # 밝기, 대비 조정
                A.ColorJitter(p=0.2), # 명도 조절
                A.ShiftScaleRotate(
                  shift_limit=0.02, # 이미지를 범위 내에서 이동
                  scale_limit=0.1, # 범위 내에서 확대/축소
                  rotate_limit=5, # 범위 내에서 회전
                  p=0.3, # 확률
                  border_mode=0), # 빈 공간은 검은색으로 채움
                A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="coco", # [x, y, w, h]
                label_fields=["labels"],
                min_visibility=0.2, # 너무 작은 박스는 드랍
            ),
        )
    else:
        return A.Compose(
            [
                A.Resize(size, size),
                A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["labels"],
            ),
        )

def detection_collate(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)

def get_trainloader(DATA_DIR, batch_size=8, size=300, split=0.8):
    train_set = build_transforms(train=True, size=size)
    val_set = build_transforms(train=False, size=size)

    # 우선 dataset 설정
    dataset = SSD_Dataset(
      img_dir = os.path.join(DATA_DIR, "train_images"),
      annotation = os.path.join(DATA_DIR, "output.json"),
      size = size,
      transforms = train_set # 일단 train으로 로드
    )

    num_classes = dataset.num_classes

    # 8:2로 split
    train_size = int(split * len(dataset))
    val_size   = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    # 다시 dataset 설정
    train_dataset.dataset.transforms = train_set
    val_dataset.dataset.transforms = val_set

    # dataloader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=detection_collate, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=detection_collate, num_workers=4)

    return train_loader, val_loader, num_classes

def get_testloader(DATA_DIR, batch_size=8, size=300):
    test_img_dir = str(DATA_DIR / "test_images")
    test_transforms = build_transforms(train=False, size=size)

    test_dataset = Test_Dataset(image_dir=test_img_dir, transforms=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return test_loader