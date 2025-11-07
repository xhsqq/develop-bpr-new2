"""
Training script for QMCSR Model
完整的QMCSR框架训练脚本（简化版，避免过拟合）
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import yaml
from tqdm import tqdm
from typing import Dict
import numpy as np
from datetime import datetime

from models.qmcsr_complete import QMCSRRecommender
from data.dataloader import get_dataloaders
from utils.evaluation import FullLibraryEvaluator, get_train_items_per_user


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


class Trainer:
    """QMCSR训练器"""

    def __init__(self, config: Dict, device: str = 'cuda'):
        self.config = config
        self.device = device

        # 加载数据
        print("Loading data...")
        self.train_loader, self.valid_loader, self.test_loader, self.num_items = get_dataloaders(
            category=config['data']['category'],
            batch_size=config['training']['batch_size'],
            max_seq_length=config['data']['max_seq_length'],
            num_workers=config.get('num_workers', 4)
        )

        # 构建模型
        print("Building model...")
        self.model = QMCSRRecommender(
            text_dim=config['model']['text_dim'],
            image_dim=config['model']['image_dim'],
            item_embed_dim=config['model']['item_embed_dim'],
            num_items=self.num_items,
            disentangled_dim=config['model']['disentangled_dim'],
            num_interests=config['model']['num_interests'],
            hidden_dim=config['model']['hidden_dim'],
            max_seq_length=config['data']['max_seq_length'],
            alpha_ortho=config['loss']['alpha_ortho'],
            alpha_causal=config['loss']['alpha_causal'],
            dropout=config['model']['dropout']
        ).to(device)

        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        # 学习率调度器
        if config['training']['scheduler']['type'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config['training']['epochs'],
                eta_min=config['training']['scheduler']['min_lr']
            )
        else:
            self.scheduler = None

        # 评估器
        self.evaluator = FullLibraryEvaluator(
            num_items=self.num_items,
            k_list=config['evaluation']['k_list'],
            device=device
        )

        # 训练历史
        self.best_ndcg = 0.0
        self.patience_counter = 0

        # 日志
        if config['logging']['tensorboard']:
            log_dir = os.path.join(
                config['logging']['log_dir'],
                datetime.now().strftime('%Y%m%d_%H%M%S')
            )
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None

        # 保存目录
        self.save_dir = config['checkpoint']['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)

    def train_epoch(self, epoch: int, use_causal: bool = False):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_rec_loss = 0.0
        total_ortho_loss = 0.0
        total_causal_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # 数据移到device
            item_ids = batch['item_ids'].to(self.device)
            text_features = batch['text_features'].to(self.device)
            image_features = batch['image_features'].to(self.device)
            seq_lengths = batch['seq_lengths'].to(self.device)
            target_items = batch['target_items'].to(self.device)

            # ⭐⭐⭐ 修复：传入数据集提供的candidate_items（如果存在）
            candidate_items = None
            if 'candidate_items' in batch:
                candidate_items = batch['candidate_items'].to(self.device)

            # 前向传播
            outputs = self.model(
                item_ids=item_ids,
                text_features=text_features,
                image_features=image_features,
                seq_lengths=seq_lengths,
                target_items=target_items,
                candidate_items=candidate_items,  # ⭐ 传入负样本
                use_causal=use_causal,
                return_loss=True
            )

            loss = outputs['loss']

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['max_grad_norm']
            )

            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            total_rec_loss += outputs['rec_loss'].item()
            total_ortho_loss += outputs['ortho_loss'].item()
            if 'causal_loss' in outputs:
                total_causal_loss += outputs['causal_loss'].item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'rec': f"{outputs['rec_loss'].item():.4f}",
                'ortho': f"{outputs['ortho_loss'].item():.4f}"
            })

            # 日志
            if self.writer and batch_idx % self.config['logging']['log_interval'] == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/loss', loss.item(), global_step)
                self.writer.add_scalar('train/rec_loss', outputs['rec_loss'].item(), global_step)
                self.writer.add_scalar('train/ortho_loss', outputs['ortho_loss'].item(), global_step)

        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader)
        avg_rec_loss = total_rec_loss / len(self.train_loader)
        avg_ortho_loss = total_ortho_loss / len(self.train_loader)
        avg_causal_loss = total_causal_loss / len(self.train_loader) if use_causal else 0.0

        print(f"\nEpoch {epoch} Train - Loss: {avg_loss:.4f}, "
              f"Rec: {avg_rec_loss:.4f}, Ortho: {avg_ortho_loss:.4f}, "
              f"Causal: {avg_causal_loss:.4f}")

        return {
            'loss': avg_loss,
            'rec_loss': avg_rec_loss,
            'ortho_loss': avg_ortho_loss,
            'causal_loss': avg_causal_loss
        }

    @torch.no_grad()
    def evaluate(self, epoch: int, use_causal: bool = False):
        """评估模型"""
        self.model.eval()

        # 收集所有预测和目标
        all_predictions = []
        all_targets = []

        for batch in tqdm(self.valid_loader, desc="Evaluating"):
            item_ids = batch['item_ids'].to(self.device)
            text_features = batch['text_features'].to(self.device)
            image_features = batch['image_features'].to(self.device)
            seq_lengths = batch['seq_lengths'].to(self.device)
            target_items = batch['target_items'].to(self.device)

            # 预测
            outputs = self.model(
                item_ids=item_ids,
                text_features=text_features,
                image_features=image_features,
                seq_lengths=seq_lengths,
                use_causal=use_causal,
                return_loss=False
            )

            logits = outputs['logits']

            # 过滤训练物品
            if self.config['evaluation']['filter_train_items']:
                for i in range(logits.size(0)):
                    train_items = item_ids[i].cpu().numpy()
                    logits[i, train_items] = -1e9

            all_predictions.append(logits)
            all_targets.append(target_items)

        # 拼接所有批次
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # 计算指标
        metrics = self.evaluator.evaluate_batch(all_predictions, all_targets)

        # 打印结果
        print(f"\nEpoch {epoch} Validation:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

        # 写入tensorboard
        if self.writer:
            for metric_name, value in metrics.items():
                self.writer.add_scalar(f'valid/{metric_name}', value, epoch)

        return metrics

    def train(self):
        """完整训练流程"""
        print("\n" + "=" * 50)
        print("Starting QMCSR Training")
        print("=" * 50)

        # 渐进式训练配置
        phase1_epochs = self.config['training']['progressive']['phase1_epochs']
        phase2_epochs = self.config['training']['progressive']['phase2_epochs']
        total_epochs = self.config['training']['epochs']

        best_epoch = 0

        for epoch in range(1, total_epochs + 1):
            # 决定是否使用因果去偏
            if epoch <= phase1_epochs:
                use_causal = self.config['training']['progressive']['use_causal_phase1']
                phase = "Phase 1 (Base)"
            else:
                use_causal = self.config['training']['progressive']['use_causal_phase2']
                phase = "Phase 2 (Causal)"

            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch}/{total_epochs} - {phase}")
            print(f"{'=' * 50}")

            # 训练
            train_metrics = self.train_epoch(epoch, use_causal=use_causal)

            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"Learning rate: {current_lr:.6f}")

            # 评估
            if epoch % self.config['logging']['eval_interval'] == 0:
                valid_metrics = self.evaluate(epoch, use_causal=use_causal)

                # 监控指标
                monitor_metric = self.config['checkpoint']['monitor']
                current_ndcg = valid_metrics.get(monitor_metric, 0.0)

                # 保存最佳模型
                if current_ndcg > self.best_ndcg:
                    self.best_ndcg = current_ndcg
                    best_epoch = epoch
                    self.patience_counter = 0

                    if self.config['checkpoint']['save_best']:
                        save_path = os.path.join(self.save_dir, 'best_model.pt')
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'metrics': valid_metrics,
                            'config': self.config
                        }, save_path)
                        print(f"✓ Saved best model (NDCG@10: {current_ndcg:.4f})")
                else:
                    self.patience_counter += 1

                # 早停
                patience = self.config['training']['early_stopping']['patience']
                if self.patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    print(f"Best model at epoch {best_epoch} with NDCG@10: {self.best_ndcg:.4f}")
                    break

        # 保存最后的模型
        if self.config['checkpoint']['save_last']:
            save_path = os.path.join(self.save_dir, 'last_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config
            }, save_path)
            print(f"✓ Saved last model")

        # 关闭tensorboard
        if self.writer:
            self.writer.close()

        print("\n" + "=" * 50)
        print("Training completed!")
        print(f"Best NDCG@10: {self.best_ndcg:.4f} at epoch {best_epoch}")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='Train QMCSR Model')
    parser.add_argument('--config', type=str, default='config_qmcsr.yaml',
                        help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 加载配置
    config = load_config(args.config)

    # 设置device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # 创建训练器并训练
    trainer = Trainer(config, device=args.device)
    trainer.train()


if __name__ == '__main__':
    main()
