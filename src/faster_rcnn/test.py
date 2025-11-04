import torch
from torch.utils.data import DataLoader
from model import build_model
from engine import collate_fn
from datasets import build_test_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 & 가중치
model = build_model(num_classes=75, weights=None).to(device)
chkpt = torch.load("./checkpoints/best.pt", map_location=device)  # 경고는 무해/아래 참고
state = chkpt.get("model", chkpt)
model.load_state_dict(state, strict=True)
print("best.pt 로드")

# 데이터로더
ds_test = build_test_dataset("./test")
test_loader = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

model.eval()
with torch.no_grad():
    for imgs, metas in test_loader:
        imgs = [img.to(device) for img in imgs]
        outputs = model(imgs)
        for meta, out in zip(metas, outputs):
            print(f"{meta['file_name']} → {len(out['boxes'])}개 탐지됨")








