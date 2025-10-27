"""
YOLOv8 (v8.3.221) - 학습
"""

from ultralytics import YOLO
import torch
import os

# 경로 설정
BASE_DIR = "/Users/apple/Documents/AI_TEAM_PROJECT_1/Codeit_Pill_Detection_Project"
DATA_PATH = os.path.join(BASE_DIR, "models/yolo8/data.yaml")

# Device 자동 설정 (CUDA -> MPS -> CPU 순으로 선택)
if torch.cuda.is_available():
    device = "cuda"       # NVIDIA GPU (Colab, PC)
elif torch.backends.mps.is_available():
    device = "mps"        # Apple M1/M2
else:
    device = "cpu"        # CPU fallback

# 모델 불러오기
model = YOLO("yolov8n.pt")

# 실제 학습 시작
model.train(
    data=DATA_PATH,
    epochs=100,        # 학습 epoch 수
    imgsz=640,         # 입력 이미지 크기
    batch=8,           # 배치 사이즈 (기본값: 16, Colab이나 M1처럼 GPU 메모리가 제한된 환경에서는 4~8 권장)
    device=device,      # Mac M1 GPU 가속 (속도 3~4배)
    project=os.path.join(BASE_DIR, "runs", "detect"),
    name="pill_yolo8_train"
)