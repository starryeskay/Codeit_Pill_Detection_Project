# engine.py
"""
엔진
- train_one_epoch: warmup + loss 로깅
- validate_loss: detection loss 계산(모델은 train 모드여야 함)
- evaluate_coco_map: COCOeval로 AP/mAP
- save_checkpoint: state_dict 저장
"""

from typing import Dict
import torch
from torch import nn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def collate_fn(batch):
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, dataloader, device, epoch, print_freq=50):
    model.train()
    lr_scheduler = None
    if epoch == 0:
        warmup_iters = min(1000, len(dataloader) - 1)
        warmup_factor = 1.0 / 1000
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
    running = 0.0
    for i, (images, targets) in enumerate(dataloader):
        images  = [img.to(device) for img in images]
        targets = [{k:v.to(device) for k,v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None and i < lr_scheduler.total_iters:
            lr_scheduler.step()

        running += losses.item()
        if (i+1) % print_freq == 0:
            details = {k: round(v.item(), 4) for k, v in loss_dict.items()}
            print(f"Epoch {epoch} | Iter {i+1}/{len(dataloader)} | loss: {running/(i+1):.4f} | details: {details}")
    return running / max(1, len(dataloader))

@torch.no_grad()
def validate_loss(model, dataloader, device):
    model.train()  # detection 손실은 train 모드에서만 산출됨
    total, count = 0.0, 0
    for images, targets in dataloader:
        images  = [img.to(device) for img in images]
        targets = [{k:v.to(device) for k,v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        total += sum(loss for loss in loss_dict.values()).item()
        count += 1
    return total / max(1, count)

def _xyxy_to_xywh(boxes: torch.Tensor):
    xywh = boxes.clone()
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
    xywh[:, 0] = boxes[:, 0]
    xywh[:, 1] = boxes[:, 1]
    return xywh

@torch.no_grad()
def evaluate_coco_map(model, dataloader, dataset, device, max_dets=100):
    model.eval()
    results, img_ids = [], []
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for out, tgt in zip(outputs, targets):
            boxes  = out.get('boxes',  torch.empty(0)).detach().cpu()
            scores = out.get('scores', torch.empty(0)).detach().cpu()
            labels = out.get('labels', torch.empty(0)).detach().cpu()
            if boxes.numel() == 0:
                continue
            boxes_xywh = _xyxy_to_xywh(boxes)
            image_id = int(tgt['image_id'].item()) if isinstance(tgt['image_id'], torch.Tensor) else int(tgt['image_id'])
            img_ids.append(image_id)
            keep = min(max_dets, boxes_xywh.size(0))
            for b, s, l in zip(boxes_xywh[:keep], scores[:keep], labels[:keep]):
                results.append({
                    'image_id': image_id,
                    'category_id': int(l),
                    'bbox': [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                    'score': float(s)
                })
    coco_gt: COCO = dataset.coco
    if len(results) == 0:
        print('[WARN] 예측 결과가 비어 있어 mAP를 계산할 수 없습니다.')
        return {k: float('nan') for k in ['AP','AP50','AP75','APs','APm','APl']}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.imgIds = getattr(dataset, 'ids', None) or list(set(img_ids))
    coco_eval.params.maxDets = [1, 10, max_dets]
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    stats = coco_eval.stats
    return {'AP': float(stats[0]), 'AP50': float(stats[1]), 'AP75': float(stats[2]),
            'APs': float(stats[3]), 'APm': float(stats[4]), 'APl': float(stats[5])}

def save_checkpoint(model, optimizer, epoch, path):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    torch.save(state, path)
