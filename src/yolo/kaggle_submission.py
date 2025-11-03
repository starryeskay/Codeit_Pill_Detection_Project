import json, re, os, csv
from pathlib import Path


def normalize(s: str) -> str:
    """
    문자열 정규화: 소문자, 알파벳·숫자만 남기고 나머지 제거
    """
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def make_class_map(model_class_names:dict):
    '''
    모델이 예측한 id를 캐글 제출용 실제 카테고리 id로 매핑해주는 사전 생성  
    - 입력:
        - model_class_names(dict): 모델이 예측한 id - 약 이름 매핑 사전   
    - 출력:
        - class_map(dict): 모델 예측 id -> 실제 카테고리 id 매핑 사전
    '''

    # 모델 예측 id -> 실제 카테고리 id 매핑 사전
    class_map = {}  

    # 실제 카테고리 id를 담고 있는 json 파일 경로
    id_json_path = Path(os.getenv("ID_DATA_DIR"))

    # 모델의 약 클래스별 id 할당 정보({id:name} 형식)
    names = model_class_names

    # original_id.json에서 category name → id 매핑 읽기
    with open(id_json_path, "r", encoding="utf-8") as f:
        id_dict = json.load(f)
    orig_name_to_orig_id = {normalize(c["name"]): c["id"] for c in id_dict["categories"]}

    # 매칭 안된 id 저장용 리스트
    unmatched = []

    # 모델 예측 id -> 실제 카테고리 id 매핑
    for i, name in names.items():
        key = normalize(name)
        if key in orig_name_to_orig_id:
            class_map[i] = orig_name_to_orig_id[key]
        else:
            unmatched.append(name)

    print(f"[INFO] 매칭된 클래스: {len(class_map)} / {len(names)}")
    if unmatched:
        print("[WARN] 매칭 안된 이름들 예시:", unmatched[:5])

    return class_map


def yolo_make_submission(
        results,
        model_ver:str,
        class_map: dict[int, int] | None = None):
    '''
    yolo 모델의 예측을 캐글 제출용 csv파일에 양식에 맞추어 저장
    - 입력
        - results: YOLO 모델의 예측 결과
        - model_ver(str): YOLO 모델 버전 ('yolov8', 'yolov10')
        - class_map(dict): 모델 예측 id -> 실제 카테고리 id 매핑 사전
    '''
    if not results:
        raise RuntimeError(f"No predictions are made.")
    rows, ann_id = [], 1
    for result in results:
        # 이미지 경로 마지막에 있는 이미지 이름에서 숫자만 추출
        img_id = result.path.split('/')[-1].split('.')[0]
        boxes = result.boxes
        W, H = result.orig_shape
        obj_num = len(boxes)
        for i in range(obj_num):
            cls = boxes.cls[i]
            conf = boxes.conf[i]
            xywh = boxes.xywh[i]
            x, y, ww, hh = xywh[0], xywh[1], xywh[2], xywh[3]
            # 경계 보정
            x = max(0.0, x); y = max(0.0, y)
            ww = max(0.0, min(ww, W - x))
            hh = max(0.0, min(hh, H - y))

            category_id = class_map.get(cls, cls) if class_map else cls
            rows.append([ann_id, img_id, category_id,
                            int(x), int(y), int(ww), int(hh),
                            round(float(conf), 2)])
            ann_id += 1

    out_csv = Path(os.getenv("PROJECT_ROOT")) / f"submission_{model_ver}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wri = csv.writer(f)
        wri.writerow(["annotation_id","image_id","category_id",
                      "bbox_x","bbox_y","bbox_w","bbox_h","score"])
        wri.writerows(rows)

    print(f"[DONE] {len(rows)} detections → {out_csv}")