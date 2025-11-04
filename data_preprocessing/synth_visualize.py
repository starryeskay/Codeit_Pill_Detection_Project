import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# ---------- 공통 유틸 ----------
def _rand_color(seed: int) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    return (rng.random(), rng.random(), rng.random())

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ---------- COCO 시각화 ----------
def visualize_coco(
    images_dir: Path,
    coco_json: Path,
    out_dir: Path,
    sample_n: int = 16,
    random_pick: bool = True,
    show_label: bool = True,
    alpha: float = 0.35,
):
    """
    COCO 포맷 (x,y,w,h) 기준 시각화 후 파일 저장.
    """
    _ensure_dir(out_dir)

    with open(coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    id_to_img = {im["id"]: im for im in coco["images"]}
    cat_id_to_name = {c["id"]: c.get("name", str(c["id"])) for c in coco.get("categories", [])}

    # 이미지별 어노테이션 묶기
    img_to_anns: Dict[int, List[dict]] = {}
    for ann in coco["annotations"]:
        img_to_anns.setdefault(ann["image_id"], []).append(ann)

    img_ids = list(id_to_img.keys())
    if random_pick:
        random.shuffle(img_ids)
    img_ids = img_ids[:sample_n]

    saved_paths = []
    for img_id in img_ids:
        im_meta = id_to_img[img_id]
        img_path = images_dir / im_meta["file_name"]
        if not img_path.exists():
            print(f"[WARN] not found: {img_path}")
            continue

        im = np.array(Image.open(img_path).convert("RGB"))
        H, W = im.shape[:2]

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(im)
        ax.axis("off")

        anns = img_to_anns.get(img_id, [])
        for ann in anns:
            x, y, w, h = ann["bbox"]
            cid = ann["category_id"]
            name = cat_id_to_name.get(cid, str(cid))
            color = _rand_color(cid)

            rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                     edgecolor=color, facecolor=color, alpha=alpha)
            ax.add_patch(rect)

            if show_label:
                ax.text(x, y - 2,
                        f"{name} ({cid})",
                        fontsize=10, color="white",
                        bbox=dict(facecolor=color, alpha=0.9, edgecolor="none", pad=1.5))

        out_path = out_dir / f"viz_coco_{img_path.stem}.jpg"
        fig.tight_layout(pad=0)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)
        saved_paths.append(out_path)

    print(f"[COCO] saved {len(saved_paths)} visualizations → {out_dir}")
    return saved_paths

# ---------- YOLO 시각화 ----------
def _read_yolo_txt(txt_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    YOLO txt 한 줄: cls xc yc w h  (정규화)
    conf가 뒤에 있을 수 있지만 여기선 무시.
    """
    items = []
    for line in txt_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])
        items.append((cls, xc, yc, w, h))
    return items

def load_id_to_name(mapping_json: Path) -> Dict[str, int]:
    with open(mapping_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "categories" in data and isinstance(data["categories"], list):
        # COCO 스타일
        out = {}
        for cat in data["categories"]:
            # 일부 json은 {"id":..,"name":..} 구조
            cid = int(cat["id"])
            name = str(cat["name"]).replace(' ', '_')
            out[cid] = name
        return out

    raise ValueError("annotations.json에서 이름→id 매핑을 찾을 수 없습니다. 'name_to_id' 또는 'categories' 키가 필요합니다.")

def visualize_yolo(
    images_dir: Path,
    labels_dir: Path,
    out_dir: Path,
    sample_n: int = 16,
    random_pick: bool = True,
    class_map_to_name: Optional[Dict[int, str]] = None,  # {yolo_cls: "약 이름"} 옵션
    alpha: float = 0.35,
):
    """
    YOLO 포맷 (정규화된 cx,cy,w,h) 기준 시각화 후 파일 저장.
    """
    _ensure_dir(out_dir)

    # 이미지 목록 스캔
    imgs = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if random_pick:
        random.shuffle(imgs)
    imgs = imgs[:sample_n]

    saved_paths = []
    for img_path in imgs:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"[WARN] label not found: {label_path}")
            continue

        im_pil = Image.open(img_path).convert("RGB")
        W, H = im_pil.size
        im = np.array(im_pil)

        yolo_items = _read_yolo_txt(label_path)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(im)
        ax.axis("off")

        for (cls, xc, yc, w, h) in yolo_items:
            # 정규화 → 픽셀 xywh
            bw = w * W
            bh = h * H
            bx = (xc * W) - bw / 2
            by = (yc * H) - bh / 2

            color = _rand_color(cls)
            rect = patches.Rectangle((bx, by), bw, bh, linewidth=2,
                                     edgecolor=color, facecolor=color, alpha=alpha)
            ax.add_patch(rect)

            name = class_map_to_name.get(cls, str(cls)) if class_map_to_name else str(cls)
            ax.text(bx, by - 2,
                    f"{name} ({cls})",
                    fontsize=10, color="white",
                    bbox=dict(facecolor=color, alpha=0.9, edgecolor="none", pad=1.5))

        out_path = out_dir / f"viz_yolo_{img_path.stem}.jpg"
        fig.tight_layout(pad=0)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)
        saved_paths.append(out_path)

    print(f"[YOLO] saved {len(saved_paths)} visualizations → {out_dir}")
    return saved_paths

# ---------- 예시 실행 ----------
if __name__ == "__main__":
    # 합성 결과 폴더 지정
    ROOT = Path("/content/drive/MyDrive/코드잇 AI 엔지니어 부트캠프/초급 프로젝트/project1/datasets_balanced/rare/aug")
    IMAGES = ROOT / "images"
    LABELS = ROOT / "labels"
    COCO_J = ROOT / "annotations_coco.json"
    OUT   = ROOT / "viz"

    # 1) COCO 시각화 (있으면)
    if COCO_J.exists():
        visualize_coco(
            images_dir=IMAGES,
            coco_json=COCO_J,
            out_dir=OUT / "coco",
            sample_n=16,
            random_pick=True,
            show_label=True
        )

    # 2) YOLO 시각화 (있으면)
    if LABELS.exists():
        # 필요하다면 클래스 이름 맵 제공: {yolo_cls: "약 이름"}
        # 예) class_map_to_name = {0:"Aspirin", 1:"Buspar", ...}
        visualize_yolo(
            images_dir=IMAGES,
            labels_dir=LABELS,
            out_dir=OUT / "yolo",
            sample_n=16,
            random_pick=True,
            class_map_to_name=load_id_to_name('/content/drive/MyDrive/코드잇 AI 엔지니어 부트캠프/초급 프로젝트/project1/datasets_balanced/rare/original/annotations.json')
        )
