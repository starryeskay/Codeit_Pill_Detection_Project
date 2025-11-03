from tqdm import tqdm
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
    
def train_each_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for imgs, targets in tqdm(data_loader, desc=f"Train", leave=False):
        imgs = [img.to(device) for img in imgs]
        targets = [
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
            for t in targets
        ]

        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        
    return total_loss / len(data_loader)
  
@torch.no_grad()
def evaluate(model, data_loader, device):
    val_losses = 0.0
    # 0.75~0.95까지 범위 지정
    metric_75 = MeanAveragePrecision(iou_thresholds=torch.arange(0.75, 1.00, 0.05).tolist())
    
    for imgs, targets in tqdm(data_loader, desc="Val"):
        
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                   for t in targets]
        
        model.train()
        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())
        val_losses += losses.item()
        
        model.eval()
        preds = model(imgs)
        
        preds_cpu = [{k: v.cpu() for k, v in p.items()} for p in preds] # CPU로 변환해야 안정적으로 동작
        targets_cpu = [{"boxes": t["boxes"].detach().cpu(),
                        "labels": t["labels"].detach().cpu(),}
                       for t in targets]

        try:
            metric_75.update(preds_cpu, targets_cpu)

        except Exception as e:
            print(f"metric update skipped : {e}") # 혹시 모를 경우를 위함
            continue
        
    metrics_75 = metric_75.compute()
 
    return val_losses / len(data_loader), metrics_75