import json
from typing import Dict, Any
cat_path = "./cat_anns_coco.json"
ann_path = "./train/_annotations.coco.json"

def label_mapper(cat_path):

    with open(cat_path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    if "categories" not in data or not isinstance(data["categories"], list):
        raise ValueError("JSON에 'categories' 리스트가 없습니다.")

    with open(ann_path, "r", encoding="utf-8") as af:
        ann_load = json.load(af)

    cats = data["categories"]


    # 원본 id -> name 사전 (키는 int)
    name_to_id: Dict[str, int] = {}
    for c in cats:
        cid = int(c["id"])
        name_to_id[str(c["name"])] = cid

     # 모델이 예측한 사전 정보
    model_id_to_name: Dict[int, str] = {}
    for ca in ann_load["categories"]:
        mid = int(ca["id"])
        model_id_to_name[mid] = str(ca["name"])

    # 위의 2 반복문 연결
    model_id_to_cat_id: Dict[int, int] = {}
    for mid, name in model_id_to_name.items():
        if name in name_to_id.keys():   #약 이름 없는거 걸러줌
            model_id_to_cat_id[mid] = name_to_id[name]   #

    return model_id_to_cat_id, model_id_to_name






