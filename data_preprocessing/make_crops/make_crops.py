import json, re, os
from pathlib import Path
from PIL import Image

DATA_DIR = Path(os.getenv("SSD_DIR")) 
DATA_SET = DATA_DIR / 'datasets_balanced/rare/original'
TRAIN_IMAGES_DIR = DATA_SET / "images" # 크롭할 이미지
CROPS_DIR = DATA_SET / "crops"
CROPS_DIR.mkdir(parents=True, exist_ok=True)

COCO_JSON = DATA_SET / "annotations.json" # COCO 형태 annotation 파일

# 안전성을 위해 넣기
def safe_name(name: str) -> str:
    
    name = (name or "").strip()
    name = re.sub(r"\s+", "_", name)
    return re.sub(r"[^\w\.-]", "_", name) or "unnamed"

# COCO 데이터셋 구조 : [x, y, w, h]
def clamp_bbox_to_image(x, y, w, h, width, height):
    
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(width,  int(round(x + w)))
    y2 = min(height, int(round(y + h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

# annotation 불러오기
with open(COCO_JSON, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = coco.get("images", [])
anns   = coco.get("annotations", [])
cats   = coco.get("categories", [])

# 빠른 매핑
id2img = {img["id"]: img for img in images}
id2cat = {cat["id"]: cat for cat in cats}

total_images = len(images)
total_annotations = len(anns)
total_crops = 0

for idx, ann in enumerate(anns):
    image_id = ann.get("image_id")
    category_id = ann.get("category_id")
    bbox = ann.get("bbox")

    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        continue

    imginfo = id2img.get(image_id)
    catinfo = id2cat.get(category_id)

    if imginfo is None or catinfo is None:
        continue

    file_name = imginfo["file_name"]
    w_img = int(imginfo.get("width", 0))
    h_img = int(imginfo.get("height", 0))

    # 원본 이미지 경로
    img_path = TRAIN_IMAGES_DIR / file_name
    if not img_path.exists():
        continue

    # 카테고리별 하위 폴더
    out_dir = CROPS_DIR / safe_name(catinfo.get("name", f"cat_{category_id}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        with Image.open(img_path) as im:
            x, y, w, h = bbox[:4]
            box = clamp_bbox_to_image(x, y, w, h, w_img or im.size[0], h_img or im.size[1])
            if box is None:
                continue

            crop = im.crop(box)  # (left, top, right, bottom)
            x1, y1, x2, y2 = box

            # 파일명: 원본스테믹스_annoID_idx_x1_y1_x2_y2.png
            stem = Path(file_name).stem
            ann_id = ann.get("id", idx)
            out_name = f"{stem}_{ann_id:06d}_{idx:05d}_{x1}_{y1}_{x2}_{y2}.png"
            out_path = out_dir / out_name

            crop.save(out_path, format="PNG", optimize=True)
            total_crops += 1
    except Exception:
        continue

print(f"images: {total_images}개, annotations: {total_annotations}개")
print(f"저장된 크롭: {total_crops}개")