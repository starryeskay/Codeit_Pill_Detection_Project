import argparse
from dotenv import load_dotenv
from src.ssd.main import start_ssd
from src.yolo.main import start_yolo

def main(args):
    # .env 파일에 저장된 폴더 경로 환경변수에 추가
    load_dotenv()

    # 모델 설정 (욜로/SSD/Faster RCNN)
    model = args.model

    # 욜로 모델
    if model == "yolov8" or model == "yolov10":
        print(f"Using {model} model and loading yolo format dataset")
        yolo_config_file_name = args.yolo_config
        start_yolo(model, yolo_config_file_name)
        
    # Faster-RCNN 모델
    elif model == "faster-rcnn":
        print(f"Using {model} model and loading COCO format dataset")

    # SSD 모델
    elif model == "ssd":
        print(f"Using {model} model and loading COCO format dataset")
        start_ssd()    

if __name__ == "__main__":
    print("This is the main module for the Codeit Pill Detection Project.")
    # 커맨드 인자 파싱
    parser = argparse.ArgumentParser(description="Codeit Pill Detection Project")
    parser.add_argument('--model', type=str, required=True, help='which model to use: yolov8, yolov10, faster-rcnn, ssd')
    parser.add_argument('--yolo_config', type=str, help='select model training / prediction configuration file in yolo_train_configs folder.', default="default.json")
    args = parser.parse_args()
    main(args)