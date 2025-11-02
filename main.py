import os
from pathlib import Path
import argparse
from dotenv import load_dotenv
from src.train_yolo import yolo_train, yolo_test_data_prediction
from src.kaggle_submission import make_class_map, yolo_make_submission

def main(args):

    load_dotenv()

    # 모델 설정 (욜로/SSD/Faster RCNN)
    model = args.model

    # 욜로 모델
    if model == "yolov8" or model == "yolov10":
        print(f"Using {model} model and loading yolo format dataset")
        yolo_config_file_name = args.yolo_config
        yolo_data_dir = os.getenv("YOLO_DATA_DIR")
        print(yolo_data_dir)
        yolo_data_path = Path(yolo_data_dir) / "data.yaml"

        # 데이터셋 경로 설정 오류
        if not yolo_data_path.exists():
            raise FileNotFoundError(f"YOLO data.yaml file not found at {yolo_data_path}")
        
        # 학습 설정 중복
        train_path = Path(os.getenv("RUN_DIR", "yolo_runs")) / "train" / f"{model}_pill_detection_{yolo_config_file_name.split('.')[0]}"
        if train_path.exists():
            print(f"Training path {train_path} already exists. Please remove or change the configuration file.")
            return
        
        # 모델 훈련
        weights_path = yolo_train(model_ver=model, data_path=yolo_data_path, config_file_name=yolo_config_file_name)
        print("train finished.")

        # 테스트 이미지 예측
        results = yolo_test_data_prediction(weights_path=weights_path, config_file_name=yolo_config_file_name)
        print("prediction finished.")

        class_dict = results[0].names
        # 모델 예측 id -> 실제 카테고리 id 매핑 사전 생성
        class_map = make_class_map(class_dict)
        yolo_make_submission(results, model_ver=model, class_map=class_map)

        
    # Faster-RCNN / SSD 모델
    elif model == "faster-rcnn" or model == "ssd":
        print(f"Using {model} model and loading COCO format dataset")
        



if __name__ == "__main__":
    print("This is the main module for the Codeit Pill Detection Project.")
    # 커맨드 인자 파싱
    parser = argparse.ArgumentParser(description="Codeit Pill Detection Project")
    parser.add_argument('--model', type=str, required=True, help='which model to use: yolov8, yolov10, faster-rcnn, ssd')
    parser.add_argument('--yolo_config', type=str, help='select model training / prediction configuration file in yolo_train_configs folder.', default="default.json")
    args = parser.parse_args()
    main(args)