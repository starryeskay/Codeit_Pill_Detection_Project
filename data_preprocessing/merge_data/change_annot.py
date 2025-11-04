import json, re
from pathlib import Path

DATA_DIR  = Path("")
coco_path = DATA_DIR / "annotations_coco.json"
map_path  = DATA_DIR / "pill_map.json"
out_path  = DATA_DIR / "coco_annotations.json"

IMAGE_OFFSET = 1488 # image id - 0 고려
ID_OFFSET = 5662 # bbox id

def norm(s):
    return re.sub(r'[^0-9a-zA-Z]+', '', s).lower()

coco = json.loads(coco_path.read_text(encoding="utf-8"))
name_to_new = json.loads(map_path.read_text(encoding="utf-8"))

categories = coco["categories"]
annotations = coco["annotations"]
images = coco["images"]

# Normalize map
map_norm = {norm(k): int(v) for k, v in name_to_new.items()}

# 카테고리 id 변경
old_to_new = {}
for c in categories:
    nk = norm(c["name"])
    if nk in map_norm:
        old_to_new[c["id"]] = map_norm[nk]

for c in categories:
    nk = norm(c["name"])
    if nk in map_norm:
        c["id"] = map_norm[nk]

# 카테고리 ID 변경
for a in annotations:
    oc = a["category_id"]
    if oc in old_to_new:
        a["category_id"] = old_to_new[oc]

for img in images:
    img["id"] += IMAGE_OFFSET

for ann in annotations:
    ann["image_id"] += IMAGE_OFFSET
    ann["id"] += ID_OFFSET

# 저장
out_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")