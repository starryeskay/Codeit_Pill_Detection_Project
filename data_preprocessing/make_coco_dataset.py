import os, json, orjson
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv("config.env")
DATA_DIR = Path(os.getenv("DATA_DIR"))
PILLS_DIR = DATA_DIR / "train_annotations"
ANNOT_DIR = DATA_DIR / "train_annotations_coco"
ANNOT_DIR.mkdir(parents=True, exist_ok=True)

# 'YYYYMMDD' -> 'YYYY-MM-DDT00:00:00+00:00'
def to_iso_ymd_zeros(s: str) -> str:
    if not s:
        return ""
    for fmt in ("%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%dT00:00:00+00:00")
        except Exception:
            pass
    return ""

# 년도만 추출
def extract_year(s: str) -> str:
    if not s: return ""
    return s[:4] if (len(s) == 8 and s.isdigit()) else (s if len(s) == 4 and s.isdigit() else s)

# id를 하나로 통일
def norm_id(v):
    try:
        return int(v)
    except Exception:
        return v

try:
    def jload(path: Path):
        with open(path, "rb") as f:
            return orjson.loads(f.read())
except ImportError:
    def jload(path: Path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

by_file = {}

for src in PILLS_DIR.rglob("*.json"):
    try:
        data = jload(src)
    except Exception as e:
        print(f"load failed: {src} ({e})")
        continue

    images      = data.get("images") or []
    annotations = data.get("annotations") or []
    categories  = data.get("categories") or []

    # 카테고리 맵
    cats_map = {c.get("id"): {"id": c.get("id"), "name": c.get("name"), "supercategory": c.get("supercategory")}
                for c in categories if c.get("id") is not None}

    # 이미지 별 json 데이터 추가
    for img in images:
        fname = Path(img.get("file_name") or "").name
        if not fname:
            continue

        # 엔트리 생성
        if fname not in by_file:
            by_file[fname] = {
                "img": {
                    "id": norm_id(img.get("id")),
                    "license": img.get("item_seq"),
                    "file_name": fname,
                    "height": img.get("height", 0),
                    "width": img.get("width", 0),
                    "date_captured": to_iso_ymd_zeros(img.get("img_regist_ts", "")) or "2020-07-20T00:00:00+00:00",
                },
                "info": {
                    "year": extract_year(img.get("di_item_permit_date", "")),
                    "version": "1",
                    "description": img.get("chart", ""),
                    "contributor": img.get("dl_company", ""),
                    "url": "",
                    "date_created": to_iso_ymd_zeros(img.get("change_date", "")) or "2000-01-01T00:00:00+00:00",
                },
                "license": {
                    "id": img.get("item_seq"),
                    "url": img.get("img_key"),
                    "name": "Public Domain",
                },
                "anns": [],
                "cats": {}
            }

        entry = by_file[fname]
        entry["cats"].update({k: v for k, v in cats_map.items() if k is not None})

        # 이 이미지와 매칭되는 어노테이션
        img_id_norm = entry["img"]["id"]  # 첫 등장 이미지의 id를 대표로 사용
        for ann in annotations:
            ann_img_id = norm_id(ann.get("image_id", ann.get("images_id")))
            if ann_img_id != img_id_norm: continue
            bbox = ann.get("bbox")
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                continue
            area = ann["area"] if ann.get("area") is not None else bbox[2] * bbox[3]

            entry["anns"].append({
                "category_id": ann.get("category_id"),
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                "area": float(area),
                "segmentation": ann.get("segmentation", []),
                "iscrowd": 0 if ann.get("iscrowd") in (None, 0, False) else 1,
            })

cat_ann_mismatch = 0
for fname, entry in by_file.items():
    img_meta = entry["img"]
    anns     = entry["anns"]
    cats_all = entry["cats"]

    # 카테고리 id가 있을 경우
    used_cids = {a["category_id"] for a in anns if a.get("category_id") in cats_all}
    categories = [cats_all[cid] for cid in sorted(used_cids)] if used_cids else sorted(cats_all.values(), key=lambda c: c["id"])

    # 어노테이션 id 재부여 + 이미지를 id로 확인 후 통일
    anns_coco = []
    for i, a in enumerate(anns):
        cid = a.get("category_id")
        if cid not in cats_all:
            continue
        anns_coco.append({
            "id": i,
            "image_id": img_meta["id"],
            "category_id": cid,
            "bbox": a["bbox"],
            "area": a["area"],
            "segmentation": a["segmentation"],
            "iscrowd": a["iscrowd"],
        })

    coco = {
        "info": entry["info"],
        "licenses": [entry["license"]],
        "categories": categories,
        "images": [img_meta],
        "annotations": anns_coco,
    }

    out_path = ANNOT_DIR / Path(fname).with_suffix(".json").name
    json.dump(coco, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"{out_path.name} (anns={len(anns_coco)}, cats={len(categories)})")
    if len(anns_coco) != len(categories):
        cat_ann_mismatch += 1

print(f"{cat_ann_mismatch}개가 불일치")
print("Done")