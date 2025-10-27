"""
YOLOv8 (v8.3.221) - 팀 공용 학습 스크립트

-------------------------------------------
YOLOv8 사전학습(pretrained) 모델 목록
-------------------------------------------
Detection (객체 탐지용)
yolov8n.pt : Nano (가장 가벼움, 빠르지만 정확도 낮음)
yolov8s.pt : Small (속도 빠름, 정확도 보통)
yolov8m.pt : Medium (속도/정확도 균형)
yolov8l.pt : Large (정확도 높음, 속도 느림)
yolov8x.pt : X-Large (가장 정확, 연산량 큼)
"""

from ultralytics import YOLO
from dotenv import load_dotenv
import os, yaml, torch, datetime

# -----------------------------
# 0️⃣ 파라미터 설정
# -----------------------------
EPOCHS = 100
BATCH_SIZE = 8
IMAGE_SIZE = 640
MODEL_NAME = "yolov8n.pt"
RUNS_SUBDIR = "yolo8"   # 결과 저장 폴더명 (ex: runs/yolo8)
VERSION_NAME = f"pill_v1_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"


# -----------------------------
# 1️⃣ config.env 로드 및 데이터 경로 설정
# -----------------------------
def get_data_dir():
    # ✅ config.env를 항상 프로젝트 루트에서 찾도록 절대경로 계산
    env_path = os.path.join(os.path.dirname(__file__), "../../config.env")
    env_path = os.path.abspath(env_path)
    print("🔍 .env 경로:", env_path)  # 경로 확인용 로그

    load_dotenv(env_path)
    data_dir = os.getenv("DATA_DIR")

    if not data_dir:
        raise ValueError("❌ DATA_DIR 환경 변수가 없습니다. config.env 파일을 확인하세요.")
    print("✅ DATA_DIR =", data_dir)
    return data_dir


# -----------------------------
# 2️⃣ data.yaml 자동 생성
# -----------------------------
def generate_data_yaml(data_dir):
    data_yaml_path = os.path.join(data_dir, "models/yolo8/data.yaml")   # data.yaml 저장 경로
    os.makedirs(os.path.dirname(data_yaml_path), exist_ok=True)

    data_yaml = {
        "train": os.path.join(data_dir, "dataset/train/images"),
        "val": os.path.join(data_dir, "dataset/val/images"),
        "test": os.path.join(data_dir, "dataset/test/images"),
        "nc": 74,
        "names": [
            "Abilify Tab-10mg", "Airtal Tab-", "Aldrin Tab-", "Amosartan Tab 5-100mg", "Atorva Tab- 10mg",
            "Atozet Tab- 10-40mg", "Azilect Tab-", "Beecom-CF Tab-", "Brintellix Tab- 20mg", "Buspar Tab- 5mg Boryung",
            "Cholinate Soft Cap-", "Chongkundang Gliatirin Soft Cap-", "Crestor Tab- 20mg", "Dabotamin Q Tab-",
            "Dried Aluminium Hydroxide Gel Tab- Samnam", "Ebixa Tab-", "Eswonamp Tab- 20mg", "Exforge Tab- 5-160mg",
            "Gabapentin Tab- 800mg Dong-A", "Gabatopa Tab- 100mg", "Ginexin-F Tab-", "Gliatamin Soft Cap.",
            "Glitin Tab-", "Hytrin Tab- 2mg Ilyang", "JANUMET TAB 50-850", "Janumet XR Tab- 100-1000mg",
            "Januvia Tab- 50mg", "Joins Tab-", "K-CAB Tab-", "Kabalin Cap- 25mg", "Kanarb Tab- 60mg", "LAYLA TAB-",
            "Lanston LFDT Tab- 30mg", "Lexapro Tab- 15mg", "Lipilow Tab- 20mg", "Lipitor Tab- 20mg", "Lirexpen Tab-",
            "Livalo Tab- 4mg", "Lyrica Cap- 150mg", "Madopar Tab-", "Maxibupen Er Tab-", "Mega Power Tab-",
            "Mucosta Tab-", "Muteran Cap- 100mg", "Naxozole Tab- 500-20mg", "Neuromed Tab- 800mg", "Nexium Tab- 40mg",
            "Noltec Tab-", "Norvasc Tab- 5mg", "Omacor Soft Cap-", "Pelubi Tab-", "Plavix Tab- 75mg", "Q-Cid Tab-",
            "Quetapin Tab- 25mg", "Rabiet Tab-20mg", "Rosuvamibe Tab- 10-20mg", "Rosuzet Tab- 10-5mg",
            "SEROQUEL Tab- 100mg", "Sevikar Tab- 10-40mg", "Shinbaro Tab-", "Stogar Tab- 10mg",
            "Suspen 8 hours ER Tab-", "Trajenta Tab-", "Trajenta-duo Tab- 2-5-850mg", "Truvita Tab-",
            "Twynsta Tab- 40-5mg", "Tylenol 8 hours ER Tab-", "Tylenol Tab- 500mg", "Ultracet ER Tab-",
            "Vimovo Tab- 500-20mg", "Vita B 100 Tab-", "Zemimet SR Tab- 50-1000mg", "Zoloft Tab- 100mg", "Zyprexa Tab- 2-5mg"
        ]
    }

    with open(data_yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    print(f"✅ data.yaml 자동 생성 완료 → {data_yaml_path}")
    return data_yaml_path



# -----------------------------
# 3️⃣ 디바이스 자동 감지
# -----------------------------
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


# -----------------------------
# 4️⃣ YOLO 학습 실행
# -----------------------------
def train_yolo(data_yaml_path, data_dir, epochs=EPOCHS, batch=BATCH_SIZE, imgsz=IMAGE_SIZE):
    device = get_device()
    print(f"🚀 학습 시작: device={device}, epochs={epochs}, batch={batch}")

    model = YOLO(MODEL_NAME)
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=os.path.join(data_dir, "runs", RUNS_SUBDIR),
        name=VERSION_NAME
    )

    print("✅ YOLOv8 학습 완료!")


# -----------------------------
# 5️⃣ 실행 엔트리포인트 (import 시 자동 실행 방지)
# -----------------------------
if __name__ == "__main__":
    data_dir = get_data_dir()
    data_yaml_path = generate_data_yaml(data_dir)
    train_yolo(data_yaml_path, data_dir)
