import pandas as pd
import torch, os, json
from pathlib import Path
from tqdm import tqdm

from dataloader import get_testloader

def test_csv(model, DATA_DIR, device, output_csv="test.csv", batch_size=8, size=300):
    model.eval()
    test_loader = get_testloader(DATA_DIR, batch_size=batch_size, size=size)

    with open(os.path.join(DATA_DIR, "output.json"), "r", encoding="utf-8") as f:
        _annot = json.load(f)
    class_ids = [cat["id"] for cat in _annot["categories"]]  # 모델 라벨 인덱스 → 원래 category_id

    file_list = test_loader.dataset.image_files
    img_dir   = test_loader.dataset.image_dir

    rows = []
    ann_id = 1  # annotation_id 순번 카운터

    with torch.no_grad():
        for batch_i, images in enumerate(tqdm(test_loader, desc="Predicting")):
            if isinstance(images, torch.Tensor):
                images = [im.to(device) for im in images]
            else:
                images = [im.to(device) for im in images]

            outputs = model(images)


            for i, output in enumerate(outputs):
                # 배치 내 i번째 샘플의 전역 인덱스
                global_idx = batch_i * test_loader.batch_size + i

                # 파일명에서 png 제거
                img_name  = file_list[global_idx]
                image_id  = Path(img_name).stem
                img_path  = Path(img_dir) / img_name

                with Image.open(img_path) as img:
                    orig_w, orig_h = img.size

                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                # 원본 크기로 복원
                scale_x = orig_w / size
                scale_y = orig_h / size

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    # 원본 크기로 되돌리고 int 값으로 변
                    x1 = int(round(x1 * scale_x))
                    y1 = int(round(y1 * scale_y))
                    x2 = int(round(x2 * scale_x))
                    y2 = int(round(y2 * scale_y))

                    bbox_x = x1
                    bbox_y = y1
                    bbox_w = max(0, x2 - x1)
                    bbox_h = max(0, y2 - y1)

                    rows.append({
                        "annotation_id": ann_id,
                        "image_id": image_id,
                        "category_id": int(class_ids[int(label)]),
                        "bbox_x": bbox_x,
                        "bbox_y": bbox_y,
                        "bbox_w": bbox_w,
                        "bbox_h": bbox_h,
                        "score": float(score)
                    })
                    ann_id += 1  # 순서 증가

    df = pd.DataFrame(rows, columns=[
        "annotation_id", "image_id", "category_id",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"
    ])

    df.to_csv(output_csv, index=False)
    print(f"{output_csv} saved")
    return df