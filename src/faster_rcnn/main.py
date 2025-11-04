# main.py
"""
엔트리포인트:
- 인자 파싱
- 데이터셋/로더/모델/옵티마이저/스케줄러 구축
- 학습 루프 + 조기종료 + 시각화 + mAP 평가 + 체크포인트
"""

import os
from pathlib import Path
import argparse
import torch
import random
from data_load import data_paths
from datasets import build_datasets_from_paths
from model import build_model
from engine import (
    collate_fn, train_one_epoch, validate_loss,
    evaluate_coco_map, save_checkpoint
)
from viz import visualize_val_grid, plot_curves

"""
해야할 것
    1. test폴더 내 이미지 평가하기
    2. 학습 때 매 epoch마다 loss 기록해서 plt로 표시하여 시각화하기
    3. Loss가 급격히 올라가는 지점의 epoch에서 train멈추기 (조기 종료 잘 작동하면 안해도 됨)

"""
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, default=None, help='데이터셋 루트(미지정시 data_load 내부 기본값 사용)')
    p.add_argument('--num_classes', type=int, default=75, help='배경 포함 클래스 개수')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--workers', type=int, default=0)  # Win 대응
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight_decay', type=float, default=0.0005)
    p.add_argument('--trainable_backbone_layers', type=int, default=3)
    p.add_argument('--weights', type=str, default='default', choices=['default', 'none'])
    p.add_argument('--out', type=str, default='./checkpoints')
    p.add_argument('--print_freq', type=int, default=5)

    # 평가/조기종료/시각화
    p.add_argument('--eval_map', action='store_true', help='학습 종료 후 최종 mAP 계산')
    p.add_argument('--eval_map_every', type=int, default=0, help='N>0이면 매 N epoch마다 mAP 계산')
    p.add_argument('--early_stop_metric', type=str, default='val_loss', choices=['val_loss','AP'])
    p.add_argument('--early_stop_patience', type=int, default=0)
    p.add_argument('--early_stop_min_delta', type=float, default=0.0)

    p.add_argument('--viz_every', type=int, default=1, help='검증 그리드 시각화 주기(비활성=0)')
    p.add_argument('--viz_num_images', type=int, default=12, help='그리드 표시 이미지 수')
    p.add_argument('--viz_score_thr', type=float, default=0.5, help='예측 박스 score threshold')
    return p.parse_args()

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # 데이터 경로
    train_img, train_ann, val_img, val_ann, test_img = data_paths(args.data_root)

    # Dataset & DataLoader
    ds_train, ds_val = build_datasets_from_paths(train_img, train_ann, val_img, val_ann)
    train_loader = torch.utils.data.DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn
    )


    # Model
    model = build_model(
        num_classes=args.num_classes,
        weights=args.weights,
        trainable_backbone_layers=args.trainable_backbone_layers
    ).to(device)

    # Optimizer & Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_sched  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    # 기록
    history = {'train_loss': [], 'val_loss': []}
    if args.eval_map_every > 0 or args.early_stop_metric == 'AP' or args.eval_map:
        history['AP'] = []

    # 학습 + 조기종료
    Path(args.out).mkdir(parents=True, exist_ok=True)
    best_metric = float('inf') if args.early_stop_metric == 'val_loss' else float('-inf')
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        tr_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=args.print_freq)
        val_loss = validate_loss(model, val_loader, device)
        lr_sched.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        print(f"\nEpoch {epoch} 완료. train_loss={tr_loss:.4f} | val_loss={val_loss:.4f}")

        # 주기적 mAP
        epoch_ap = None
        if args.eval_map_every and (epoch + 1) % args.eval_map_every == 0:
            print('\n[Eval] COCO mAP 평가 중...')
            map_metrics = evaluate_coco_map(model, val_loader, ds_val, device)
            epoch_ap = map_metrics.get('AP', None)
            print('[Eval] mAP 결과:', map_metrics)
        if 'AP' in history:
            history['AP'].append(epoch_ap)

        # 주기적 시각화
        if args.viz_every and (epoch + 1) % args.viz_every == 0:
            out_img = visualize_val_grid(
                model, ds_val, device, args.out, epoch=epoch,
                num_images=args.viz_num_images, ncols=4, score_thr=args.viz_score_thr, draw_gt=True
            )
            print(f"[Viz] 저장됨: {out_img}")

        # 조기 종료
        if args.early_stop_metric == 'val_loss':
            current = val_loss
            improved = (best_metric - current) > args.early_stop_min_delta
        else:
            if epoch_ap is None:
                print('[EarlyStop][AP] 에폭별 mAP 계산 중...')
                map_metrics = evaluate_coco_map(model, val_loader, ds_val, device)
                epoch_ap = map_metrics.get('AP', None)
                print('[EarlyStop][AP] mAP 결과:', map_metrics)
                if 'AP' in history:
                    history['AP'][-1] = epoch_ap
            current = float('-inf') if epoch_ap is None else epoch_ap
            improved = (current - best_metric) > args.early_stop_min_delta

        # 저장
        save_checkpoint(model, optimizer, epoch, os.path.join(args.out, 'last.pt'))
        if improved:
            best_metric = current
            save_checkpoint(model, optimizer, epoch, os.path.join(args.out, 'best.pt'))
            epochs_no_improve = 0
            tag = 'val_loss' if args.early_stop_metric == 'val_loss' else 'AP'
            print(f"[Best] {tag} 개선 → best.pt 저장 ({tag}={best_metric:.4f})")
        else:
            epochs_no_improve += 1
            if args.early_stop_patience > 0:
                print(f"[EarlyStop] 개선 없음: {epochs_no_improve}/{args.early_stop_patience} 에폭")
                if epochs_no_improve >= args.early_stop_patience:
                    print('[EarlyStop] 조기 종료 트리거됨.')
                    break

    # 최종 mAP (옵션)
    if args.eval_map:
        print('\n[Final Eval] COCO mAP 평가 중...')
        map_metrics = evaluate_coco_map(model, val_loader, ds_val, device)
        print('[Final Eval] mAP 결과:', map_metrics)
        if 'AP' in history and (len(history['AP']) < args.epochs):
            history['AP'].append(map_metrics.get('AP', None))

    # 곡선 저장
    plot_curves(history, args.out)
    print('학습 완료.')



if __name__ == '__main__':
    main()
