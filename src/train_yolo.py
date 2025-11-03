from ultralytics import YOLO
import torch
import os
from pathlib import Path
import json

def yolo_train(model_ver:str, data_path:Path, config_file_name:str):
    """
    욜로 모델을 훈련시킵니다.  
    - 입력:
        - model_ver: YOLO 모델 버전 ('yolov8', 'yolov10')
        - data_path: 데이터셋 yaml 파일 경로
        - config_file_name: 하이퍼파라미터 설정 파일 이름
    - 출력:
        - weights_path: 학습한 가중치 저장 경로
    """
    # 모델 불러오기
    if model_ver == "yolov8":
        model = YOLO("yolov8n.pt")  # yolov8n 사전학습 모델 불러오기
    elif model_ver == "yolov10":
        model = YOLO("yolov10n.yaml")  # yolov10n 사전학습 모델 불러오기
 
    # 장치 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 하이퍼파라미터 설정 파일 로드
    config_dir = os.getenv("CONFIG_DIR")
    config_path = Path(config_dir) / config_file_name
    
    # 하이퍼파라미터 설정
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        train_config = config["train_config"]
        opt = train_config.get("OPTIM", "adam")
        lr = train_config.get("LR", 0.001)
        batch_size = train_config.get("BATCH_SIZE", 16)
        momentum = train_config.get("MOMENTUM", 0.9)
        dropout = train_config.get("DROPOUT", 0.0)
        epochs = train_config.get("EPOCHS", 50)
        save_period = train_config.get("SAVE_P", 10)
    
    # 학습 결과 저장 디렉토리
    train_run_dir = Path(os.getenv("RUN_DIR", "yolo_runs")) / "train"

    # 모델 훈련
    model.train(
        data=data_path,
        epochs=epochs,
        batch=batch_size,
        lr0=lr,
        device=device,
        optimizer=opt,
        momentum=momentum,
        dropout=dropout,
        save=True,
        save_period=save_period,
        project=str(train_run_dir),
        name=f"{model_ver}_pill_detection_{config_file_name.split('.')[0]}",
    )

    # 가중치 저장 경로
    weights_path = train_run_dir / f"{model_ver}_pill_detection_{config_file_name.split('.')[0]}" / "weights" / "best.pt"
    return weights_path

def yolo_test_data_prediction(model_ver:str, weights_path:Path, config_file_name:str):
    """
    훈련된 모델로 테스트 이미지를 예측합니다.  
    - 입력:
        - model_ver: YOLO 모델 버전 ('yolov8', 'yolov10')
        - weights_path: 훈련된 모델 가중치 파일 경로
        - config_file_name: 하이퍼파라미터 설정 파일 이름  
    - 출력:
        - results: 모델의 종합 예측 결과(지표, 예측 결과 등)   
    """

    # 모델 불러오기
    model = YOLO(weights_path)

    # 장치 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 테스트 이미지 경로
    img_path = os.getenv("TEST_DIR", "data/test/images")
    
    # 결과 저장 디렉토리
    pred_run_dir = Path(os.getenv("RUN_DIR", "yolo_runs")) / "predict"

    # 하이퍼파라미터 설정 파일 로드
    config_dir = os.getenv("CONFIG_DIR")
    config_path = Path(config_dir) / config_file_name
    
    # 하이퍼파라미터 설정
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        pred_config = config["prediction_config"]
        conf = pred_config.get("CONFIDENCE_THRESHOLD", 0.25)
        max_det = pred_config.get("MAX_DET", 4)

    # 예측 수행
    results = model.predict(
        source=img_path,
        conf=conf,
        max_det=max_det,
        save_txt=True,    # labels/*.txt 저장
        save_conf=True,   # conf 점수 포함
        project=str(pred_run_dir),
        name=f"{model_ver}_predict_{config_file_name.split('.')[0]}",
        device=device
        )

    return results



