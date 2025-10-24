import json, re, os
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

load_dotenv("config.env")
DATA_DIR = Path(os.getenv("DATA_DIR"))
TRAIN_IMAGES_DIR = DATA_DIR / "train_images"
ANNOT_DIR = DATA_DIR / "train_annotations_pills"
CROPS_DIR = DATA_DIR / "crops"
CROPS_DIR.mkdir(parents=True, exist_ok=True)


def safe_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"\s+", "_", name)
    return re.sub(r"[^\w\.-]", "_", name) or "unnamed"

def to_png_name(imgfile: str) -> str:
    p = Path(imgfile)
    return f"{p.stem}.png"

def clamp_bbox_to_image(x, y, w, h, width, height):
    # COCO bbox: [x, y, w, h]
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(width,  int(round(x + w)))
    y2 = min(height, int(round(y + h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

total_json = total_locs = total_crops = 0
skipped_missing_img = skipped_bad_bbox = 0

for cat_json in sorted(ANNOT_DIR.glob("*.json")):
    with open(cat_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    cat_name = (payload.get("file_data") or {}).get("name") or cat_json.stem
    out_dir = CROPS_DIR / safe_name(cat_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    locations = payload.get("location") or []
    total_json += 1

    for idx, loc in enumerate(locations):
        total_locs += 1
        imgfile = to_png_name(str(loc.get("imgfile", "")))
        bbox    = loc.get("bbox")

        # bbox 기본 검증
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            skipped_bad_bbox += 1
            continue

        # PNG만 사용
        img_path = TRAIN_IMAGES_DIR / imgfile
        if not img_path.exists():
            skipped_missing_img += 1
            continue

        try:
            with Image.open(img_path) as im:
                w_img, h_img = im.size
                x, y, w, h = bbox[:4]
                box = clamp_bbox_to_image(x, y, w, h, w_img, h_img)
                if box is None:
                    skipped_bad_bbox += 1
                    continue

                crop = im.crop(box)  # (left, top, right, bottom)
                x1, y1, x2, y2 = box
                out_name = f"{Path(imgfile).stem}_{idx:05d}_{x1}_{y1}_{x2}_{y2}.png"
                out_path = out_dir / out_name

                # PNG로 저장
                crop.save(out_path, format="PNG", optimize=True)
                total_crops += 1
        except Exception:
            skipped_bad_bbox += 1
            continue

print(f"[완료] JSON: {total_json}개, location: {total_locs}개")
print(f"[결과] 저장된 크롭: {total_crops}개")
print(f"[스킵] 이미지 없음: {skipped_missing_img}개, bbox 문제: {skipped_bad_bbox}개")
print(f"[출력 경로] {CROPS_DIR.resolve()}")
