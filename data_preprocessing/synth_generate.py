import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import numpy as np
from datetime import datetime

# -----------------------------
# 1) 유틸: YOLO 정규화 변환
# -----------------------------
def xywh_pixels_to_yolo_norm(x: float, y: float, w: float, h: float, W: int, H: int) -> Tuple[float, float, float, float]:
    # (x,y,w,h)는 좌상단 기준 픽셀, YOLO는 중심 기준 [0,1]
    xc = (x + w/2) / W
    yc = (y + h/2) / H
    ww = w / W
    hh = h / H
    # 경계 안전
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    ww = min(max(ww, 0.0), 1.0)
    hh = min(max(hh, 0.0), 1.0)
    return xc, yc, ww, hh

# -----------------------------
# 2) 매핑 로더: 약 이름 -> category_id
# -----------------------------
def load_name_to_id(mapping_json: Path) -> Dict[str, int]:
    with open(mapping_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2가지 포맷 지원
    if "name_to_id" in data and isinstance(data["name_to_id"], dict):
        # keys: name, values: id
        # 키/값의 타입 보정
        return {str(k): int(v) for k, v in data["name_to_id"].items()}

    if "categories" in data and isinstance(data["categories"], list):
        # COCO 스타일
        out = {}
        for cat in data["categories"]:
            # 일부 json은 {"id":..,"name":..} 구조
            cid = int(cat["id"])
            name = str(cat["name"]).replace(' ', '_')
            out[name] = cid
        return out

    raise ValueError("annotations.json에서 이름→id 매핑을 찾을 수 없습니다. 'name_to_id' 또는 'categories' 키가 필요합니다.")

# -----------------------------
# 3) 오브젝트 뱅크 스캔: 폴더명=약 이름
# -----------------------------
def scan_object_bank(bank_dir: Path) -> Dict[str, List[Path]]:
    """
    bank_dir/
      ├─ 아세트아미노펜/   ← 약 이름(폴더명)
      │   ├─ *.png (alpha 포함 crop/seg 결과)
      ├─ 이부프로펜/
      ...
    """
    name_to_files = {}
    for cls_dir in sorted(p for p in bank_dir.iterdir() if p.is_dir()):
        pngs = sorted(list(cls_dir.glob("*.png")))
        if pngs:
            name_to_files[cls_dir.name] = pngs
    if not name_to_files:
        raise RuntimeError(f"object_bank가 비어있습니다: {bank_dir}")
    return name_to_files

# -----------------------------
# 4) 합성 한 장 만들기
# -----------------------------
def compose_one_image(
    canvas_size: Tuple[int, int],
    positions_px: List[Tuple[int, int]],
    bank: Dict[str, List[Path]],
    name_to_id: Dict[str, int],
    scale_range: Tuple[float, float] = (0.9, 1.1),
    bg_color: Tuple[int, int, int] = (230, 230, 230),
    rng: random.Random = random
):
    """
    2x2 고정 위치에 4개 배치 (positions_px 길이가 4라고 가정)
    각 객체는 오브젝트 뱅크에서 (약 이름 선택 -> PNG 선택) 순으로 고름.
    """
    W, H = canvas_size
    canvas = Image.new("RGB", (W, H), bg_color)

    # 결과(라벨) 누적
    yolo_labels = []  # (class_index_for_yolo, xc, yc, w, h)
    coco_anns = []    # dicts
    used = []         # for debug

    # 폴더명(=약 이름) 리스트
    class_names = list(bank.keys())

    # 4개 샘플
    for i, (px, py) in enumerate(positions_px):
        # 1) 약 이름 선택
        name = rng.choice(class_names)
        files = bank[name]
        # 2) 약 이미지 선택
        png_path = rng.choice(files)
        im = Image.open(png_path).convert("RGBA")

        # 3) 크기 스케일 & 붙여넣기
        s = rng.uniform(*scale_range)
        new_w = max(1, int(im.width * s))
        new_h = max(1, int(im.height * s))
        im = im.resize((new_w, new_h), Image.BICUBIC)

        # 캔버스 넘어가지 않게 간단 보정
        px_clamped = min(max(px, 0), W - new_w)
        py_clamped = min(max(py, 0), H - new_h)

        # 합성
        canvas.alpha_composite(im, dest=(px_clamped, py_clamped)) if canvas.mode == "RGBA" else canvas.paste(im, (px_clamped, py_clamped), im)

        # 4) 알파 채널 기준 실제 bbox 계산
        alpha = np.array(im.split()[-1])  # (new_h, new_w)
        nz = np.argwhere(alpha > 0)
        if nz.size == 0:
            # 전부 투명이라면 스킵
            continue
        (min_y, min_x), (max_y, max_x) = nz.min(axis=0), nz.max(axis=0)
        bw = int(max_x - min_x + 1)
        bh = int(max_y - min_y + 1)
        bx = int(px_clamped + min_x)
        by = int(py_clamped + min_y)

        # 5) 카테고리 id 매핑 (필수)
        if name not in name_to_id:
            raise KeyError(f"'{name}'에 해당하는 category_id가 annotations.json에 없습니다.")
        category_id = int(name_to_id[name])

        # 6) YOLO 정규화 좌표
        xc, yc, ww, hh = xywh_pixels_to_yolo_norm(bx, by, bw, bh, W, H)
        # YOLO의 class index는 학습용 names[]의 순서에 맞춰야 하는데,
        # 여기서는 'category_id' 제출을 위해 COCO용 id만 저장하고,
        # YOLO txt에는 '학습 시 사용한 class index'가 필요하다면 외부에서 매핑 사용.
        # (지금은 submission 목적이라 class index=category_id로 일시 저장 예시)
        yolo_labels.append((category_id, xc, yc, ww, hh))

        # 7) COCO bbox(x,y,w,h)
        coco_anns.append({
            "category_id": category_id,
            "bbox": [bx, by, bw, bh],
            "area": float(bw * bh)
        })
        used.append((name, png_path.name))

    return canvas.convert("RGB"), yolo_labels, coco_anns

# -----------------------------
# 5) 합성 파이프라인
# -----------------------------
def synthesize_dataset(
    object_bank_dir: Path,
    mapping_json: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    out_coco_json: Path,
    n_images: int = 200,
    canvas_size: Tuple[int, int] = (1024, 1024),
    fixed_positions: List[Tuple[int, int]] | None = None,
    seed: int = 42
):
    rng = random.Random(seed)
    name_to_id = load_name_to_id(mapping_json)
    bank = scan_object_bank(object_bank_dir)

    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    # 2x2 고정 위치(겹침 방지 위해 충분한 마진)
    if fixed_positions is None:
        W, H = canvas_size
        # 좌상 / 우상 / 좌하 / 우하
        # 각 셀 (W/2, H/2) 안에서 좌상단 오프셋
        margin = 40
        cell_w, cell_h = W // 2, H // 2
        fixed_positions = [
            (margin, margin),
            (cell_w + margin, margin),
            (margin, cell_h + margin),
            (cell_w + margin, cell_h + margin),
        ]

    images_info = []
    annotations = []
    categories = [{"id": cid, "name": name} for name, cid in sorted(name_to_id.items(), key=lambda x: x[1])]

    ann_id = 1
    for i in range(1, n_images + 1):
        img, yolo_labels, coco_anns = compose_one_image(
            canvas_size=canvas_size,
            positions_px=fixed_positions,
            bank=bank,
            name_to_id=name_to_id,
            rng=rng
        )
        img_name = f"syn_{i:06d}.jpg"
        lbl_name = f"syn_{i:06d}.txt"

        img_path = out_images_dir / img_name
        lbl_path = out_labels_dir / lbl_name
        img.save(img_path, quality=95)

        # YOLO txt 저장 (여기서는 class=category_id를 임시로 사용)
        with open(lbl_path, "w", encoding="utf-8") as f:
            for cls, xc, yc, ww, hh in yolo_labels:
                f.write(f"{cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")

        # COCO image/annotations 누적
        W, H = img.size
        images_info.append({
            "id": i,
            "file_name": img_name,
            "width": W,
            "height": H
        })
        for a in coco_anns:
            a_out = {
                "id": ann_id,
                "image_id": i,
                "category_id": a["category_id"],
                "bbox": a["bbox"],     # [x,y,w,h]
                "area": a["area"],
                "iscrowd": 0
            }
            annotations.append(a_out)
            ann_id += 1

    coco = {
        "info": {
            "year": datetime.now().year,
            "description": "Synthetic pill dataset (fixed 2x2 layout)"
        },
        "licenses": [],
        "images": images_info,
        "annotations": annotations,
        "categories": categories
    }
    out_coco_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_coco_json, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Images: {len(images_info)}  Annotations: {len(annotations)}")
    print(f"       → {out_images_dir}")
    print(f"       → {out_labels_dir}")
    print(f"       → {out_coco_json}")

# -----------------------------
# 6) 실행 예시
# -----------------------------
if __name__ == "__main__":
    object_bank_dir = Path("/content/drive/MyDrive/코드잇 AI 엔지니어 부트캠프/초급 프로젝트/project1/datasets_balanced/rare/original/crops_rgba")   # 폴더명=약 이름
    mapping_json    = Path("/content/drive/MyDrive/코드잇 AI 엔지니어 부트캠프/초급 프로젝트/project1/datasets_balanced/rare/original/annotations.json")  # 이름→id 매핑
    out_root        = Path("/content/drive/MyDrive/코드잇 AI 엔지니어 부트캠프/초급 프로젝트/project1/datasets_balanced/rare/aug")
    out_images_dir  = out_root / "images"
    out_labels_dir  = out_root / "labels"
    out_coco_json   = out_root / "annotations_coco.json"

    synthesize_dataset(
        object_bank_dir=object_bank_dir,
        mapping_json=mapping_json,
        out_images_dir=out_images_dir,
        out_labels_dir=out_labels_dir,
        out_coco_json=out_coco_json,
        n_images=200,
        canvas_size=(1024, 1024),
        seed=2025
    )

