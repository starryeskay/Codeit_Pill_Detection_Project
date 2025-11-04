import json
from pathlib import Path

DATA_DIR = Path("")
BASE = DATA_DIR / "output.json" # 기존 파일
ADD  = DATA_DIR / "coco_annotations.json" # 새로 넣을 파일
OUT  = DATA_DIR / "coco_merged.json" # 합친 파일

base = json.loads(BASE.read_text(encoding="utf-8"))
add  = json.loads(ADD.read_text(encoding="utf-8"))

merged = {
    "categories": base["categories"], # 카테고리 그대로 유지
    "images": base["images"] + add["images"],
    "annotations": base["annotations"] + add["annotations"]
}

OUT.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")