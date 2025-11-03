from rembg import remove
from PIL import Image
from tqdm import tqdm
import numpy as np
from pathlib import Path

# make_crops.py를 써 crop된 이미지의 배경 제거
def u2net_remove_folder(in_dir: Path, out_dir: Path, suffix="_nobg", # in_dir : crops 폴더, out_dir : 저장 폴더, suffix : 접미사
                        alpha_matting=True,
                        alpha_matting_foreground_threshold=240,
                        alpha_matting_background_threshold=10,
                        alpha_matting_erode_size=10):
    exts = {".jpg", ".jpeg", ".png"}
    files = [p for p in in_dir.rglob("*") if p.suffix.lower() in exts]
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in tqdm(files, desc="U2Net remove bg"):
        rel = p.relative_to(in_dir)
        dst_dir = out_dir / rel.parent
        dst_dir.mkdir(parents=True, exist_ok=True)
        out_path = dst_dir / f"{p.stem}{suffix}.png"

        with Image.open(p).convert("RGBA") as im: # Alpha로 투명색을 넣기 위함
            out = remove(
                im,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold, # 알약의 보존 정도
                alpha_matting_background_threshold=alpha_matting_background_threshold, # 배경의 지워짐 정도
                alpha_matting_erode_size=alpha_matting_erode_size, # 경계 부분 잡음 제거
            )
            # 가장자리 살짝 부드럽게
            arr = np.array(out)

            Image.fromarray(arr, mode="RGBA").save(out_path)
            
# u2net_remove_folder(IN_DIR, OUT_DIR)