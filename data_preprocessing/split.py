import json
import os
from pathlib import Path
from collections import defaultdict


# merge_jsons.py 이용해서 만든 jSON 파일 경로 (각자에 맞게)
MERGED_JSON_PATH = r"C:\Users\yth91\OneDrive\바탕 화면\코드잇 데이터\초급\ai05-level1-project\train_annotations_merged.json"

# JSON 파일들을 저장할 새 폴더 (각자에 맞게)
SPLIT_DIR = r"C:\Users\yth91\OneDrive\바탕 화면\코드잇 데이터\초급\ai05-level1-project\train_annotations_split"


# 출력 폴더 생성 
os.makedirs(SPLIT_DIR, exist_ok=True)

print(f"원본 파일 로드 : {MERGED_JSON_PATH}")
with open(MERGED_JSON_PATH, 'r', encoding='utf-8') as f:
    coco_data = json.load(f)

# 카테고리 정보는 모든 분할 파일에 공통으로 포함
categories = coco_data.get("categories", [])

# 어노테이션들을 image_id 기준으로 미리 그룹화
print("어노테이션 그룹화 ")
annotations_by_image = defaultdict(list)
for ann in coco_data.get("annotations", []):
    annotations_by_image[ann['image_id']].append(ann)

#images리스트를 순회하며 각 이미지별로 파일을 생성
image_list = coco_data.get("images", [])
print(f"{len(image_list)}개로 분할")        # 이거랑 전체 이미지 수랑 맞아야 함

for image_info in image_list:
    image_id = image_info['id']
    file_name = image_info['file_name']
    
    # 이 이미지에 해당하는 어노테이션 목록 가져오기
    image_annotations = annotations_by_image.get(image_id, [])
    
    # 새 딕셔너리 생성
    new_coco_data = {
        "images": [image_info],             #이미지는 이 파일에 해당하는 1개만 리스트에 넣음
        "annotations": image_annotations,   #이 이미지에 속한 어노테이션만 넣음
        "categories": categories            #카테고리 리스트는 전체를 그대로 복사
    }
    
    # 저장할 파일명 (원본 이미지 파일명에서 png를 json으로 변경)
    out_filename = Path(file_name).with_suffix('.json').name
    out_path = os.path.join(SPLIT_DIR, out_filename)
    
    # JSON 파일로 저장
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(new_coco_data, f, ensure_ascii=False, indent=4)

print(f"저장완료, 저장 위치: {SPLIT_DIR}")              # 위치 잘 확인하세용