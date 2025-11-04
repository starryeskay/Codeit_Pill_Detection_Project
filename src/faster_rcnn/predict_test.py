# predict_test.py
import os, re, json
import torch
import pandas as pd
from torchvision.ops import nms

from datasets import build_test_dataset
from engine import collate_fn
from model import build_model
from label_mapping import label_mapper  # 모델라벨 -> 원본 category_id

# ---- 기본 설정 (필요하면 바꿔서 사용) ----
DATA_ROOT = "./test"
WEIGHTS = "./checkpoints/best.pt"
CAT_JSON = "./cat_anns_coco.json"
OUT_CSV = "./submission_final.csv"

# 탐지 후처리
SCORE_THR = 0.05
NMS_IOU = 0.6
TOP_K = 300  # 이미지당 최대 내보낼 박스 수

ID_RE = re.compile(r"(\d+)")

def xyxy_to_xywh(box):
    x1, y1, x2, y2 = map(float, box)
    return [x1, y1, max(1e-3, x2 - x1), max(1e-3, y2 - y1)]

def extract_image_id(file_name: str) -> int:
    m = ID_RE.search(file_name)
    if not m:
        raise ValueError(f"image_id 추출 실패: {file_name}")
    return int(m.group(1))

def main():
    # 매핑: 모델라벨(0..K) -> 원본 category_id / 이름
    class_map = label_mapper(CAT_JSON)





    # 데이터/모델
    ds = build_test_dataset(DATA_ROOT)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(num_classes=75, weights="default").to(device)
    state  = torch.load(WEIGHTS, map_location=device)
    state  = state.get("model", state)
    model.load_state_dict(state, strict=True)
    model.eval()

    rows = []
    ann_id = 1
    empty_imgs = 0
    bad_cat_count = 0

    with torch.no_grad():
        for images, metas in loader:
            images  = [img.to(device) for img in images]
            outputs = model(images)

            for meta, out in zip(metas, outputs):
                image_id = extract_image_id(meta["file_name"])

                boxes  = out.get("boxes",  torch.empty(0)).detach().cpu()
                labels = out.get("labels", torch.empty(0)).detach().cpu()
                scores = out.get("scores", torch.empty(0)).detach().cpu()

                # 스코어 필터
                keep = scores >= SCORE_THR
                boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

                # NMS
                if boxes.numel() > 0:
                    keep_idx = nms(boxes, scores, NMS_IOU)
                    boxes, labels, scores = boxes[keep_idx], labels[keep_idx], scores[keep_idx]
                else:
                    empty_imgs += 1

                # TOP-K 제한
                if len(scores) > TOP_K:
                    topk = scores.topk(TOP_K).indices
                    boxes, labels, scores = boxes[topk], labels[topk], scores[topk]

                # 행 추가
                for b, l, s in zip(boxes, labels, scores):
                    x1, y1, x2, y2 = map(float, b.tolist())
                    x, y, w, h = xyxy_to_xywh([x1, y1, x2, y2])

                    lab = int(l.item())                # 내부 라벨
                    cat_id = class_map[lab]     # ✅ 원본 category_id로 변환


                    rows.append([
                        ann_id, image_id, cat_id,
                        round(x), round(y), round(w), round(h),
                        float(round(s.item(), 2))
                    ])
                    ann_id += 1

    cols = ["annotation_id","image_id","category_id","bbox_x","bbox_y","bbox_w","bbox_h","score"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(OUT_CSV, index=False)
    print(f"[OK] saved: {OUT_CSV} (rows={len(df)})")
    print(f"[INFO] images with no detections: {empty_imgs}")
    print(f"[INFO] skipped (disallowed category_id): {bad_cat_count}")

if __name__ == "__main__":
    main()
