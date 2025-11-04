import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import numpy as np
from datetime import datetime

# -----------------------------
# 1) 유틸: YOLO 정규화 변환 (원본과 동일)
# -----------------------------
def xywh_pixels_to_yolo_norm(x: float, y: float, w: float, h: float, W: int, H: int) -> Tuple[float, float, float, float]:
    xc = (x + w/2) / W
    yc = (y + h/2) / H
    ww = w / W
    hh = h / H
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    ww = min(max(ww, 0.0), 1.0)
    hh = min(max(hh, 0.0), 1.0)
    return xc, yc, ww, hh

# -----------------------------
# 2) 매핑 로더 (원본과 동일)
# -----------------------------
def load_name_to_id(mapping_json: Path) -> Dict[str, int]:
    with open(mapping_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "name_to_id" in data and isinstance(data["name_to_id"], dict):
        return {str(k): int(v) for k, v in data["name_to_id"].items()}
    if "categories" in data and isinstance(data["categories"], list):
        out = {}
        for cat in data["categories"]:
            cid = int(cat["id"])
            name = str(cat["name"]).replace(' ', '_')
            out[name] = cid
        return out
    raise ValueError("annotations.json에서 이름→id 매핑을 찾을 수 없습니다. 'name_to_id' 또는 'categories' 키가 필요합니다.")

# -----------------------------
# 3) 오브젝트 뱅크 스캔 (원본과 동일)
# -----------------------------
def scan_object_bank(bank_dir: Path) -> Dict[str, List[Path]]:
    name_to_files = {}
    for cls_dir in sorted(p for p in bank_dir.iterdir() if p.is_dir()):
        pngs = sorted(list(cls_dir.glob("*.png")))
        if pngs:
            name_to_files[cls_dir.name] = pngs
    if not name_to_files:
        raise RuntimeError(f"object_bank가 비어있습니다: {bank_dir}")
    return name_to_files

# -----------------------------
# 4) 합성 한 장 만들기 (이름 강제 지정 가능하도록 확장)
# -----------------------------
def compose_one_image(
    canvas_size: Tuple[int, int],
    positions_px: List[Tuple[int, int]],
    bank: Dict[str, List[Path]],
    name_to_id: Dict[str, int],
    scale_range: Tuple[float, float] = (0.95, 1.0),
    bg_color: Tuple[int, int, int] = (230, 230, 230),
    rng: random.Random = random,
    names_for_slots: Optional[List[str]] = None,   # ★ 추가: 이 순서대로 배치
):
    """
    positions_px 길이만큼 배치.
    names_for_slots가 주어지면 해당 카테고리만 사용.
    """
    W, H = canvas_size
    canvas = Image.new("RGB", (W, H), bg_color)

    yolo_labels = []
    coco_anns = []
    used_names: List[str] = []

    class_names = list(bank.keys())
    
    MIN_GAP = 10               # 최소 간격(px)
    MIN_SCALE = 0.9            # 원본의 최소 90%까지 축소 허용
    SHRINK_FACTOR = 0.92       # 스케일 실패 시 감소 비율(단, MIN_SCALE 아래로는 내려가지 않음)
    
    # 이미 배치된 bbox들(원래 bbox)을 보관
    placed_bboxes: List[Tuple[int, int, int, int]] = []  # (x, y, w, h)
    
    def intersects_with_gap(a, b, gap: int) -> bool:
        """사각형 a,b가 gap만큼 확장했을 때 겹치면 True.
        a,b: (x, y, w, h) in pixels"""
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        # 각각 gap만큼 확장
        ax1i, ay1i, ax2i, ay2i = ax1 - gap, ay1 - gap, ax2 + gap, ay2 + gap
        bx1i, by1i, bx2i, by2i = bx1 - gap, by1 - gap, bx2 + gap, by2 + gap

        # 축 정렬 사각형 겹침 검사
        if ax1i >= bx2i or bx1i >= ax2i:
            return False
        if ay1i >= by2i or by1i >= ay2i:
            return False
        return True
    
    # 배치 개수 = positions_px 길이 (슬롯 일부만 써도 됨)
    for i, (px, py) in enumerate(positions_px):
        # 1) 약 이름 선택(강제 or 랜덤)
        if names_for_slots is not None:
            if i >= len(names_for_slots):
                break  # 더 배치할 이름이 없으면 종료
            name = names_for_slots[i]
            if name not in bank:
                # 없는 이름이면 스킵
                continue
        else:
            name = rng.choice(class_names)

        files = bank[name]
        png_path = rng.choice(files)
        im = Image.open(png_path).convert("RGBA")

        # 3) 크기 스케일 & 붙여넣기
        s = min(max(rng.uniform(*scale_range), MIN_SCALE), max(scale_range))  # 시작 스케일을 범위 내에서, 하한 0.9 보장
        placed = False

        while True:
            # 하한 0.9 보장
            if s < MIN_SCALE: 
                s = MIN_SCALE
        
            new_w = max(1, int(im.width * s))
            new_h = max(1, int(im.height * s))
            im = im.resize((new_w, new_h), Image.BICUBIC)

            px_clamped = min(max(px, 0), W - new_w)
            py_clamped = min(max(py, 0), H - new_h)

            

            # 4) 알파 기준 bbox
            alpha = np.array(im.split()[-1])
            nz = np.argwhere(alpha > 0)
            if nz.size == 0:
                # 투명만 있으면 스케일 한 단계 줄여서 재시도 (단, 0.9 아래로는 안내려감)
                next_s = max(MIN_SCALE, s * SHRINK_FACTOR)
                # 더 줄일 수 없는 경우(이미 0.9)면 실패 처리
                if next_s == s:
                    raise RuntimeError(f"이미지 '{name}'가 완전 투명으로 계산되어 배치할 수 없습니다.")
                s = next_s
                continue
            
            (min_y, min_x), (max_y, max_x) = nz.min(axis=0), nz.max(axis=0)
            bw = int(max_x - min_x + 1)
            bh = int(max_y - min_y + 1)
            bx = int(px_clamped + min_x)
            by = int(py_clamped + min_y)
            
            candidate = (bx, by, bw, bh)
            
            # 기존 배치와 10px 간격 검증
            conflict = any(intersects_with_gap(candidate, prev, MIN_GAP) for prev in placed_bboxes)
            
            if not conflict:
                canvas.paste(im, (px_clamped, py_clamped), im)

                # 5) 카테고리 id
                if name not in name_to_id:
                    raise KeyError(f"'{name}'에 해당하는 category_id가 annotations.json에 없습니다.")
                category_id = int(name_to_id[name])

                # 6) YOLO 정규화 좌표
                xc, yc, ww, hh = xywh_pixels_to_yolo_norm(bx, by, bw, bh, W, H)
                yolo_labels.append((category_id, xc, yc, ww, hh))

                # 7) COCO bbox
                coco_anns.append({
                    "category_id": category_id,
                    "bbox": [bx, by, bw, bh],
                    "area": float(bw * bh)
                })
                used_names.append(name)
                placed_bboxes.append(candidate)
                placed = True
                break
            
            # ★ 충돌: 스케일 더 줄이고 재시도 (단, 0.9 미만은 금지)
            next_s = s * SHRINK_FACTOR
            if next_s < MIN_SCALE:
                # 더 이상 줄일 수 없음 → 요구한 최소 스케일(0.9)에서도 간격 불가
                # 스킵 없이 에러로 알려줌
                raise RuntimeError(
                    f"positions_px 및 오브젝트 크기 조건상 최소 간격 {MIN_GAP}px을 만족할 수 없습니다. "
                    f"(클래스='{name}', 스케일={s:.3f})\n"
                    f"- 해결책: positions 간격을 더 띄우거나, 이미지 원본 자체를 더 작게 준비하세요."
                )
            s = next_s
        
        # placed는 여기서 반드시 True여야 함(위에서 에러를 던지거나 배치됨)
        assert placed, "논리 오류: placed가 False 상태로 루프를 탈출했습니다."

    return canvas.convert("RGB"), yolo_labels, coco_anns, used_names  # ★ used_names 반환

# -----------------------------
# 5’) 카테고리별 타겟 개수 맞춰 합성
# -----------------------------
def synthesize_dataset_per_category(
    object_bank_dir: Path,
    mapping_json: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    out_coco_json: Path,
    per_category_target: int = 100,           # ★ 카테고리별 목표 개수
    canvas_size: Tuple[int, int] = (976, 1280),
    fixed_positions: Optional[List[Tuple[int, int]]] = None,
    seed: int = 42
):
    rng = random.Random(seed)
    name_to_id = load_name_to_id(mapping_json)
    bank = scan_object_bank(object_bank_dir)

    # bank와 mapping 교집합만 사용
    valid_names = sorted(set(bank.keys()) & set(name_to_id.keys()))
    if not valid_names:
        raise RuntimeError("object_bank와 annotations 매핑의 교집합 카테고리가 없습니다.")

    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    # 기본 2x2 그리드
    if fixed_positions is None:
        W, H = canvas_size
        margin = 40
        cell_w, cell_h = W // 2, H // 2
        fixed_positions = [
            (margin, margin),
            (cell_w + margin, margin),
            (margin, cell_h + margin),
            (cell_w + margin, cell_h + margin),
        ]

    # 각 카테고리 목표 개수(“인스턴스 수”) 설정
    remaining: Dict[str, int] = {name: per_category_target for name in valid_names}

    images_info = []
    annotations = []
    categories = [{"id": name_to_id[name], "name": name} for name in sorted(valid_names, key=lambda n: name_to_id[n])]

    ann_id = 1
    img_idx = 1

    # 남은 quota 총합이 0이 될 때까지 이미지 생성
    def total_remaining() -> int:
        return sum(max(0, v) for v in remaining.values())

    while total_remaining() > 0:
        # 이번 이미지에 배치할 이름들 결정(슬롯 수만큼, 남은 quota > 0인 것만)
        names_this_image: List[str] = []

        # 간단한 전략: 남은 수량이 큰 순서대로 채우기 (균형 잡히게)
        candidates = [n for n in valid_names if remaining[n] > 0]
        if not candidates:
            break

        # 한 장에 배치 가능한 최대 개수
        max_slots = len(fixed_positions)
        # 남은 수량이 큰 순으로 소팅
        candidates.sort(key=lambda n: remaining[n], reverse=True)

        # 슬롯 개수만큼 pick (남은 수가 슬롯보다 적으면 그만큼만)
        for n in candidates[:max_slots]:
            names_this_image.append(n)

        # positions도 같은 개수만큼만 사용
        use_positions = fixed_positions[:len(names_this_image)]

        # 합성
        img, yolo_labels, coco_anns, used_names = compose_one_image(
            canvas_size=canvas_size,
            positions_px=use_positions,
            bank=bank,
            name_to_id=name_to_id,
            rng=rng,
            names_for_slots=names_this_image  # 강제 배치
        )

        # 실제로 배치된 이름 기준으로 remaining 차감(알파 0 같은 예외 케이스 대비)
        for u in used_names:
            if remaining.get(u, 0) > 0:
                remaining[u] -= 1

        # 저장
        img_name = f"syn_{img_idx:06d}.png"
        lbl_name = f"syn_{img_idx:06d}.txt"
        img_path = out_images_dir / img_name
        lbl_path = out_labels_dir / lbl_name
        img.save(img_path, optimize=True)

        with open(lbl_path, "w", encoding="utf-8") as f:
            for cls, xc, yc, ww, hh in yolo_labels:
                f.write(f"{cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")

        Wc, Hc = img.size
        images_info.append({
            "id": img_idx,
            "file_name": img_name,
            "width": Wc,
            "height": Hc
        })
        for a in coco_anns:
            a_out = {
                "id": ann_id,
                "image_id": img_idx,
                "category_id": a["category_id"],
                "bbox": a["bbox"],
                "area": a["area"],
                "iscrowd": 0
            }
            annotations.append(a_out)
            ann_id += 1

        img_idx += 1

    coco = {
        "info": {
            "year": datetime.now().year,
            "description": f"Synthetic pill dataset (per-category target={per_category_target})"
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
    print("Per-category counts achieved:")
    for n in sorted(valid_names):
        print(f" - {n}: target={per_category_target}, remaining={remaining[n]} (<=0이면 충족)")

# -----------------------------
# 6) 실행 예시
# -----------------------------
if __name__ == "__main__":
    DATA_DIR = Path("/content/drive/MyDrive/Codeit/project1/datasets_balanced/rare/original")
    object_bank_dir = DATA_DIR / 'crops_rgba'   # 폴더명=약 이름
    mapping_json    = DATA_DIR / "annotations.json"  # 이름→id 매핑
    out_root        = Path("/content/drive/MyDrive/Codeit/project1/Final_data")
    out_images_dir  = out_root / "train_images"
    out_labels_dir  = out_root / "train_labels"
    out_coco_json   = out_root / "annotations_coco.json"

    synthesize_dataset_per_category(
        object_bank_dir=object_bank_dir,
        mapping_json=mapping_json,
        out_images_dir=out_images_dir,
        out_labels_dir=out_labels_dir,
        out_coco_json=out_coco_json,
        per_category_target=100,           # ★ 카테고리별 100개
        canvas_size=(976, 1280),
        seed=2025
    )