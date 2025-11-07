"""
Training script for Multimodal Sequential Recommender
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
import os
import json
from tqdm import tqdm
from typing import Dict, Optional
import numpy as np

from models.multimodal_recommender import MultimodalRecommender
from utils.metrics import evaluate_all_metrics
from utils.losses import BPRLoss, InfoNCELoss


class DummyDataset(Dataset):
    """
    示例数据集
    实际使用时需要替换为真实数据集
    """

    def __init__(
        self,
        num_samples: int = 1000,
        num_items: int = 10000,
        seq_length: int = 20,
        modality_dims: Dict[str, int] = None
    ):
        self.num_samples = num_samples
        self.num_items = num_items
        self.seq_length = seq_length

        if modality_dims is None:
            modality_dims = {'text': 768, 'image': 2048, 'audio': 128}

        self.modality_dims = modality_dims

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 生成随机序列
        item_ids = torch.randint(1, self.num_items, (self.seq_length,))
        target_item = torch.randint(1, self.num_items, (1,)).item()

        # 生成随机多模态特征
        multimodal_features = {
            modality: torch.randn(self.seq_length, dim)
            for modality, dim in self.modality_dims.items()
        }

        # 序列长度（随机变化）
        seq_length = torch.randint(self.seq_length // 2, self.seq_length + 1, (1,)).item()

        return {
            'item_ids': item_ids,
            'multimodal_features': multimodal_features,
            'target_items': target_item,
            'seq_lengths': seq_length
        }


def collate_fn(batch):
    """批处理函数"""
    item_ids = torch.stack([item['item_ids'] for item in batch])
    target_items = torch.tensor([item['target_items'] for item in batch])
    seq_lengths = torch.tensor([item['seq_lengths'] for item in batch])

    # 收集多模态特征
    modality_keys = batch[0]['multimodal_features'].keys()
    multimodal_features = {
        modality: torch.stack([item['multimodal_features'][modality] for item in batch])
        for modality in modality_keys
    }

    return {
        'item_ids': item_ids,
        'multimodal_features': multimodal_features,
        'target_items': target_items,
        'seq_lengths': seq_lengths
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int
) -> Dict[str, float]:
    """训练一个epoch"""
    model.train()

    total_loss = 0
    total_rec_loss = 0
    total_dis_loss = 0
    total_div_loss = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch in pbar:
        # 移动到设备
        item_ids = batch['item_ids'].to(device)
        target_items = batch['target_items'].to(device)
        seq_lengths = batch['seq_lengths'].to(device)
        multimodal_features = {
            k: v.to(device) for k, v in batch['multimodal_features'].items()
        }

        # 前向传播
        outputs = model(
            item_ids=item_ids,
            multimodal_features=multimodal_features,
            seq_lengths=seq_lengths,
            target_items=target_items,
            return_loss=True,
            return_explanations=False
        )

        loss = outputs['loss']

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 记录损失
        total_loss += loss.item()
        total_rec_loss += outputs['recommendation_loss'].item()
        total_dis_loss += outputs['disentangled_loss'] if isinstance(outputs['disentangled_loss'], float) else outputs['disentangled_loss'].item()
        total_div_loss += outputs['diversity_loss'] if isinstance(outputs['diversity_loss'], float) else outputs['diversity_loss'].item()

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'rec_loss': f'{outputs["recommendation_loss"].item():.4f}'
        })

    # 计算平均损失
    num_batches = len(dataloader)
    metrics = {
        'loss': total_loss / num_batches,
        'rec_loss': total_rec_loss / num_batches,
        'dis_loss': total_dis_loss / num_batches,
        'div_loss': total_div_loss / num_batches
    }

    return metrics


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str
) -> Dict[str, float]:
    """验证"""
    model.eval()

    metrics = evaluate_all_metrics(
        model,
        dataloader,
        device,
        k_list=[5, 10, 20]
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train Multimodal Recommender')

    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--item_embed_dim', type=int, default=256)
    parser.add_argument('--disentangled_dim', type=int, default=128)
    parser.add_argument('--num_interests', type=int, default=4)
    parser.add_argument('--quantum_state_dim', type=int, default=256)
    parser.add_argument('--num_items', type=int, default=10000)
    parser.add_argument('--max_seq_length', type=int, default=20)

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=5)

    # 损失权重
    parser.add_argument('--alpha_recon', type=float, default=1.0)
    parser.add_argument('--alpha_causal', type=float, default=0.5)
    parser.add_argument('--alpha_diversity', type=float, default=0.1)

    # 其他
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--log_interval', type=int, default=10)

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    print(f"Training on device: {args.device}")
    print(f"Configuration: {json.dumps(vars(args), indent=2)}")

    # 模态维度
    modality_dims = {
        'text': 768,
        'image': 2048,
        'audio': 128
    }

    # 创建数据集
    print("Creating datasets...")
    train_dataset = DummyDataset(
        num_samples=5000,
        num_items=args.num_items,
        seq_length=args.max_seq_length,
        modality_dims=modality_dims
    )

    val_dataset = DummyDataset(
        num_samples=1000,
        num_items=args.num_items,
        seq_length=args.max_seq_length,
        modality_dims=modality_dims
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 创建模型
    print("Creating model...")
    model = MultimodalRecommender(
        modality_dims=modality_dims,
        disentangled_dim=args.disentangled_dim,
        num_disentangled_dims=3,
        num_interests=args.num_interests,
        quantum_state_dim=args.quantum_state_dim,
        hidden_dim=args.hidden_dim,
        item_embed_dim=args.item_embed_dim,
        num_items=args.num_items,
        max_seq_length=args.max_seq_length,
        alpha_recon=args.alpha_recon,
        alpha_causal=args.alpha_causal,
        alpha_diversity=args.alpha_diversity,
        use_quantum_computing=False
    ).to(args.device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 优化器和调度器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2
    )

    # 训练循环
    best_ndcg = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        # 训练
        train_metrics = train_epoch(model, train_loader, optimizer, args.device, epoch)

        print(f"\nTraining metrics:")
        for key, value in train_metrics.items():
            print(f"  {key}: {value:.4f}")

        # 验证
        if epoch % args.log_interval == 0:
            print(f"\nValidating...")
            val_metrics = validate(model, val_loader, args.device)

            print(f"\nValidation metrics:")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")

            # 保存最佳模型
            if val_metrics['NDCG@10'] > best_ndcg:
                best_ndcg = val_metrics['NDCG@10']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                }, os.path.join(args.save_dir, 'best_model.pt'))
                print(f"\n✓ Saved best model (NDCG@10: {best_ndcg:.4f})")

        # 更新学习率
        scheduler.step()

        # 定期保存检查点
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pt'))

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best NDCG@10: {best_ndcg:.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
