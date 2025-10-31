from ultralytics import YOLO
import torch
import os
from pathlib import Path

def train_yolo(
        model:YOLO, data_path,
        epochs=50, img_size=640, batch_size=16, device='cpu', seed=42,
        lr=0.001, optimizer="adam", momentum=0.9, dropout=0.0, box=7.5, cls=0.5,
        save=True, save_period=10,
               ):
    """
    욜로 모델을 훈련시킵니다.  
    파라미터:
    - 모델 및 데이터셋 설정
        - model: YOLO 모델 객체
        - data_path: 데이터셋 yaml 파일 경로
    - 훈련 설정
        - epochs: 훈련 에폭 수
        - img_size: 입력 이미지 크기
        - batch_size: 배치 크기
        - device: 'cpu' 또는 'cuda'
        - seed: 랜덤 시드 값
        - lr: 학습률
        - optimizer: 옵티마이저 종류 ('adam', 'sgd' 등)
        - momentum: adam beta값 / sgd 모멘텀 값
        - dropout: 드롭아웃 비율
        - box: 박스 손실 가중치
        - cls: 클래스 손실 가중치
    - 저장 설정
        - save: 훈련된 모델 저장 여부
        - save_period: 모델 저장 주기 (에폭 단위)
    """
    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        lr0=lr,
        device=device,
        seed=seed,
        optimizer=optimizer,
        momentum=momentum,
        dropout=dropout,
        box=box,
        cls=cls,
        save=save,
        save_period=save_period
    )
# 장치 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 하이퍼파라미터 설정
opt = "adam"
LR = 0.001
BATCH_SIZE = 16

# 모델 불러오기
model = YOLO("yolov10n.yaml",
            task="detect")  # load a pretrained YOLOv10n model

# 데이터셋 경로 설정
data_path = str(Path(os.getcwd()) / "data" / "Pill_Detection_yolov8" / "data.yaml")

train_yolo(model=model, data_path=data_path, lr=LR, optimizer=opt, batch_size=BATCH_SIZE, device=device)

