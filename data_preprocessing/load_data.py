import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# .env 파일 읽기
load_dotenv("config.env")

# 데이터 경로 가져오기
DATA_DIR = os.getenv("DATA_DIR")

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"⚠️ DATA_DIR 경로가 존재하지 않습니다: {DATA_DIR}")

print(f"✅ 데이터 경로: {DATA_DIR}")

img_path = os.path.join(DATA_DIR, "test_images", "1.png")

if not os.path.exists(img_path):
    raise FileNotFoundError(f"❌ 이미지 파일이 없습니다: {img_path}")

# === 이미지 로드 및 시각화 ===
img = mpimg.imread(img_path)

# 한글 폰트 설정 -> 깨짐 방지
plt.rc('font', family='NanumGothic')
plt.figure(figsize=(6,6))
plt.imshow(img)
plt.axis("off")
plt.title("1.png 미리보기", fontsize=14)
plt.show()