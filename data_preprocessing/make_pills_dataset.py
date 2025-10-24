import os, json, re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv("config.env")
DATA_DIR = Path(os.getenv("DATA_DIR"))
SRC_JSON = DATA_DIR / "coco_annotations.json"
OUT_DIR = DATA_DIR / "train_annotations_pills"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def safe_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"\s+", "_", name)
    return re.sub(r"[^\w\.-]", "_", name) or "unnamed"

with open(SRC_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

images_by_id     = {img["id"]: img for img in data.get("images", [])}
categories_by_id = {c["id"]: c for c in data.get("categories", [])}

cat_locations = {}
seen = set() # (cat_id, imgfile, *bbox) 로 중복 체크

for ann in data.get("annotations", []):
    cat_id = ann.get("category_id")
    img_id = ann.get("image_id")
    bbox   = ann.get("bbox") or []
    if cat_id is None or img_id is None:
        continue

    img = images_by_id.get(img_id, {})
    imgfile = (img.get("extra", {}) or {}).get("name") or img.get("file_name") or f"image_{img_id}"
    key = (cat_id, imgfile, *bbox) 

    if key in seen:
        continue
    seen.add(key)

    cat_locations.setdefault(cat_id, []).append({"imgfile": imgfile, "bbox": list(bbox)})

for cat_id, locations in cat_locations.items():
    cat_name = (categories_by_id.get(cat_id, {}) or {}).get("name") or f"category_{cat_id}"
    out_path = OUT_DIR / f"{safe_name(cat_name)}.json"
    payload = {"file_data": {"name": cat_name, "category_id": cat_id}, "location": locations}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("Saved:", out_path)

print("Done.")