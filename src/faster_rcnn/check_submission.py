# check_submission.py
import os, glob, re, json
import pandas as pd

SUB_CSV = "./submission_final.csv"   # 제출 파일명
TEST_DIR = "./test"                   # 테스트 이미지 폴더
CAT_JSON = "./cat_anns_coco.json"     # 대회 카테고리 파일

REQUIRED_COLUMNS = [
    "annotation_id","image_id","category_id",
    "bbox_x","bbox_y","bbox_w","bbox_h","score"
]

ID_REGEX = re.compile(r"(\d+)")

def extract_image_id(fname: str):
    stem = os.path.splitext(fname)[0]
    if stem.isdigit():
        return int(stem)
    m = ID_REGEX.findall(fname)
    return int(m[-1]) if m else None

def main():
    # 0) 제출 CSV 로드 + 헤더 확인
    if not os.path.exists(SUB_CSV):
        raise FileNotFoundError(f"제출 파일 없음: {SUB_CSV}")

    df = pd.read_csv(SUB_CSV)
    df.columns = [c.strip() for c in df.columns]
    if list(df.columns) != REQUIRED_COLUMNS:
        raise AssertionError(f"헤더 불일치: {list(df.columns)}\n필요 헤더: {REQUIRED_COLUMNS}")
    print(f"rows: {len(df):,}")

    # 1) 기본 규칙
    if not df["annotation_id"].is_unique:
        dup_cnt = df["annotation_id"].duplicated().sum()
        raise AssertionError(f"annotation_id 중복 {dup_cnt}건")
    if not (df["annotation_id"] > 0).all():
        raise AssertionError("annotation_id는 양수여야 합니다.")

    if not df["score"].between(0, 1).all():
        bad = df.loc[~df["score"].between(0, 1)].head()
        raise AssertionError(f"score가 [0,1] 범위를 벗어났습니다. 예시:\n{bad}")

    for c in ["bbox_x","bbox_y","bbox_w","bbox_h"]:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise AssertionError(f"{c}는 숫자여야 합니다.")
        if not (df[c] >= 0).all():
            bad = df.loc[~(df[c] >= 0)].head()
            raise AssertionError(f"{c}에 음수값 존재. 예시:\n{bad}")
    if not (df["bbox_w"] > 0).all() or not (df["bbox_h"] > 0).all():
        bad = df.loc[(df["bbox_w"] <= 0) | (df["bbox_h"] <= 0)].head()
        raise AssertionError(f"bbox_w/bbox_h가 0 이하인 항목 존재. 예시:\n{bad}")

    # 2) 허용 category_id 집합 검사
    with open(CAT_JSON, "r", encoding="utf-8") as f:
        cats = json.load(f)["categories"]
    allowed_cat_ids = {int(c["id"]) for c in cats}

    used_cats = set(df["category_id"].astype(int))
    bad_cats = used_cats - allowed_cat_ids
    if bad_cats:
        sample = sorted(list(bad_cats))[:20]
        raise AssertionError(f"허용되지 않는 category_id 존재: {sample} (총 {len(bad_cats)}개)")
    print(f"OK: category_id 집합 일치 (총 {len(used_cats)}개 사용)")

    # 3) 테스트 이미지 커버리지
    img_paths = []
    for ext in ("*.jpg","*.jpeg","*.png","*.JPG","*.PNG","*.JPEG","*.bmp","*.tif","*.tiff"):
        img_paths += glob.glob(os.path.join(TEST_DIR, "**", ext), recursive=True)

    ids_from_test = set()
    for p in img_paths:
        iid = extract_image_id(os.path.basename(p))
        if iid is not None:
            ids_from_test.add(iid)

    ids_from_sub = set(df["image_id"].astype(int))


    miss  = ids_from_test - ids_from_sub
    extra = ids_from_sub  - ids_from_test



    # (선택) 이미지당 예측 분포 간단 확인
    gb = df.groupby("image_id").size()
    print(gb.describe())

    # 4) 요약
    if (not miss) and (not extra):
        print("✅ 검사 통과 (형식, 값 범위, 카테고리, 커버리지 OK)")
    else:
        print("⚠️ 커버리지 이슈가 있습니다. 제출 전 확인 필요.")

if __name__ == "__main__":
    main()
