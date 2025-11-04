import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms
import json

from datasets import build_test_dataset
from model import build_model
from engine import collate_fn
from label_mapping import label_mapper

cate_path = "./cat_anns_coco.json"
class_map, id_to_name = label_mapper(cate_path)


def draw_image(img, boxes, labels, scores, score_thr=0.3):
    draw = ImageDraw.Draw(img)
    keep = scores >= score_thr
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

    if boxes.numel() > 0:
        keep_idx = nms(boxes, scores, iou_threshold=0.5)
        boxes, labels, scores = boxes[keep_idx], labels[keep_idx], scores[keep_idx]
    else:
        return img

    for b, l, s in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(float, b.tolist())
        cls_id = int(l.item())
        prob = float(s.item())

        drug_name = id_to_name[cls_id]
        text = f"{drug_name} {prob*100:.1f}%"

        # Pillow 8.0+ : textbbox 사용
        tb = draw.textbbox((0, 0), text)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]

        # 박스 & 라벨 배경
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        draw.rectangle([x1, y1 - th - 6, x1 + tw + 8, y1], fill=(255, 0, 0))
        draw.text((x1 + 4, y1 - th - 4),text, fill=(255, 255, 255))

    return img

def main():
    test_dir = "./test"
    ds_test = build_test_dataset(test_dir)
    loader = DataLoader(ds_test, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=75, weights=None).to(device)
    ckpt = torch.load("./checkpoints/best.pt", map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(" best.pt 로드 완료")

    out_dir = Path("./outputs/test_vis")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[viz] 저장 폴더: {out_dir.resolve()}")

    with torch.no_grad():
        for images, metas in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            # outputs = model(images) 이후


            # === DEBUG: 이번 배치에서 상위 예측 몇 개 그대로 보기 ===
            out0 = outputs[0]
            lbls = out0["labels"].detach().cpu()
            scrs = out0["scores"].detach().cpu()
            topk = min(5, len(scrs))
            idxs = scrs.topk(topk).indices.tolist() if topk > 0 else []

            for i in idxs:
                li = int(lbls[i].item())
                si = float(scrs[i].item())
                print(f"  pred: cls={li:3d}  orig={class_map[li]:4d}  name={id_to_name[li]}  score={si:.4f}")

            for meta, out in zip(metas, outputs):
                file_name = meta["file_name"]
                img_path = Path(test_dir) / file_name
                img = Image.open(img_path).convert("RGB")

                boxes = out.get("boxes", torch.empty(0)).detach().cpu()
                labels = out.get("labels", torch.empty(0)).detach().cpu()
                scores = out.get("scores", torch.empty(0)).detach().cpu()

                img_drawn = draw_image(img, boxes, labels, scores, score_thr=0.3)
                save_path = out_dir / file_name
                img_drawn.save(save_path)
                print(f"[viz] {file_name} 저장 완료")

    print("모든 시각화 완료!")


if __name__ == "__main__":
    main()