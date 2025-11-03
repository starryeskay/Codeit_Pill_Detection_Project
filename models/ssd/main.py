import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead
from dotenv import load_dotenv
from pathlib import Path
import os

from train_and_eval import train_each_epoch, evaluate
from dataloader import get_trainloader
from submit import test_csv


def main(DATA_DIR, device, num_epochs=10, size=300):
    load_dotenv("config.env")
    os.makedirs("weights", exist_ok=True)

    train_loader, val_loader, num_classes = get_trainloader(DATA_DIR, batch_size=8, size=300)

    model = ssd300_vgg16(weights="COCO_V1")
    total_classes = num_classes + 1  # 배경 포함

    # head에 있는 num_classes 교체
    try:
        model.replace_head(total_classes)
    except AttributeError:
        try:
            # 직접 교체
            model.head.classification_head.num_classes = total_classes
        except Exception:
            # 구버전 완전 수동 교체
            num_anchors = model.anchor_generator.num_anchors_per_location()
            in_channels = [512, 1024, 512, 256, 256, 256]  # SSD300 기본 구조 유지
            model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, total_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=0.002, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # 초기값 설정
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"{epoch+1}/{num_epochs}")

        train_loss = train_each_epoch(model, train_loader, optimizer, device)
        val_loss, metrics_75 = evaluate(model, val_loader, device)
        val_loss = float(val_loss)

        # mAP, recall 출력
        map_all = float(metrics_75['map'])
        map_75 = float(metrics_75['map_75'])
        mar_100 = float(metrics_75['mar_100'])
        print(f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")
        print(f"mAP@[.75:.95]: {map_all:.4f} | mAP@0.75: {map_75:.4f} | recall: {mar_100:.4f}")

        lr_scheduler.step()

        # 모델 저장
        torch.save(model.state_dict(), f"weights/ssd_epoch{epoch+1}.pth")


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "weights/ssd_best.pth")

    print("Training Finished")

if __name__ == "__main__":
    DATA_DIR = Path('/content/drive/MyDrive/Codeit/project1')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(DATA_DIR, device, num_epochs=20)

    # best 모델 로드
    model = ssd300_vgg16(weights=None)
    model.load_state_dict(torch.load("weights/ssd_best.pth", map_location=device))
    model = model.to(device)

    df = test_csv(model, DATA_DIR, device, output_csv="test.csv")