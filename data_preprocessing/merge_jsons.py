import os, json, glob, sys
from pathlib import Path
from collections import OrderedDict
from PIL import Image

# 각자 경로에 맞게 수정
ROOT      = r"C:\Users\yth91\OneDrive\바탕 화면\코드잇 데이터\초급\ai05-level1-project"
IMG_DIR   = rf"{ROOT}\train_images"
ANN_ROOT  = rf"{ROOT}\train_annotations"  # <- 여기를 루트로 재귀 탐색
OUT_JSON  = rf"{ROOT}\train_annotations_merged.json"

# 
def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_image_size(file_name, img_dir):
    """
    (w,h) 반환. json에 width/height 없으면 실제 이미지 열어 채움.
    file_name이 절대/상대/폴더 포함/파일명만 들어와도 최대한 찾음.
    찾지 못하면 None.
    """
    p = Path(file_name)

    #  절대 경로로 보이는 경우
    if p.is_file():
        try:
            with Image.open(p) as im:
                return im.size  # (w,h)
        except Exception:
            pass

    # IMG_DIR 기준으로 상대/파일명
    cand = Path(img_dir) / file_name
    if cand.is_file():
        try:
            with Image.open(cand) as im:
                return im.size
        except Exception:
            pass

    # 3) 파일명만 떼서 IMG_DIR에서 찾기
    base = Path(file_name).name
    cand2 = Path(img_dir) / base
    if cand2.is_file():
        try:
            with Image.open(cand2) as im:
                return im.size
        except Exception:
            pass

    return None

# 머지 컨테이너 
merged = {"images": [], "annotations": [], "categories": []}

# 카테고리는 "이름" 기준으로 통합(이름이 없으면 id를 키로)
cat_name_to_newid = OrderedDict()
next_cat_id = 1

# image / ann id는 전역 카운터
next_img_id = 1
next_ann_id = 1

# (file_name basename, w, h) -> new image_id
seen_images = {}

# 재귀적으로 모든 .json 수집
json_list = glob.glob(os.path.join(ANN_ROOT, "**", "*.json"), recursive=True)
json_list.sort()
print(f"[INFO] Found json files: {len(json_list)}")
if len(json_list) == 0:
    print("[ERROR] No JSON found under ANN_ROOT. 경로를 확인해줘.")
    sys.exit(1)

# 본격 머지 
for idx, jp in enumerate(json_list, 1):
    if idx % 50 == 0:
        print(f"[INFO] Processing {idx}/{len(json_list)}: {jp}")

    try:
        coco = load_json(jp)
    except Exception as e:
        print(f"[WARN] Cannot read {jp}: {e}")
        continue

    # categories: 이름 기준 전역 통합 
    local_old2new = {}
    for cat in coco.get("categories", []):
        name = cat.get("name")
        key = name if name else f"id:{cat.get('id')}"
        if key not in cat_name_to_newid:
            cat_name_to_newid[key] = next_cat_id
            merged["categories"].append({
                "id": next_cat_id,
                "name": name if name else f"cls{next_cat_id}",
                "supercategory": cat.get("supercategory", "pill")
            })
            next_cat_id += 1
        local_old2new[cat["id"]] = cat_name_to_newid[key]

    # images: 파일명/크기 기준으로 통합 
    oldimg2new = {}
    for im in coco.get("images", []):
        file_name = im.get("file_name")
        if not file_name:
            # COCO 표준이면 거의 없지만, 혹시 없으면 스킵
            continue

        w = im.get("width")
        h = im.get("height")
        if (w is None) or (h is None):
            wh = ensure_image_size(file_name, IMG_DIR)
            if wh is None:
                # 치수 확인 불가하면 스킵
                print(f"[WARN] size unknown & image not found: {file_name} (json: {jp})")
                continue
            w, h = wh

        # 중복 방지: (basename, w, h)
        key = (Path(file_name).name, int(w), int(h))
        if key in seen_images:
            new_id = seen_images[key]
            oldimg2new[im["id"]] = new_id
            continue

        new_id = next_img_id
        merged["images"].append({
            "id": new_id,
            "file_name": Path(file_name).name,  # 경로 제거, 파일명만 저장
            "width": int(w),
            "height": int(h)
        })
        seen_images[key] = new_id
        oldimg2new[im["id"]] = new_id
        next_img_id += 1

    #  annotations: 유효 bbox만, 카테고리/이미지 맵핑 적용 
    for ann in coco.get("annotations", []):
        bbox = ann.get("bbox")
        if not bbox or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        if bbox[2] <= 0 or bbox[3] <= 0:
            continue

        old_img_id = ann.get("image_id")
        if old_img_id not in oldimg2new:
            continue

        old_cat_id = ann.get("category_id")
        if old_cat_id not in local_old2new:
            continue

        merged["annotations"].append({
            "id": next_ann_id,
            "image_id": oldimg2new[old_img_id],
            "category_id": local_old2new[old_cat_id],
            "bbox": [float(b) for b in bbox],  # [x,y,w,h]
            "area": float(ann.get("area", bbox[2]*bbox[3])),
            "iscrowd": int(ann.get("iscrowd", 0)),
            "segmentation": ann.get("segmentation", [])
        })
        next_ann_id += 1

# 저장 
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False)
print(f"[OK] Saved merged COCO: {OUT_JSON}")
print(f"  #images={len(merged['images'])}  #anns={len(merged['annotations'])}  #cats={len(merged['categories'])}")
