"""
Training script for Multimodal Recommender on Amazon datasets
使用真实Amazon数据训练，全库评估，无负采样
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import json
import yaml
from tqdm import tqdm
from typing import Dict, Optional
import numpy as np
from datetime import datetime
import math

from models.multimodal_recommender import MultimodalRecommender
from data.dataloader import get_dataloaders
from utils.evaluation import FullLibraryEvaluator, get_train_items_per_user


def load_config(config_path: str) -> Dict:
    """
    从YAML配置文件加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def merge_config_with_args(config: Dict, args: argparse.Namespace) -> argparse.Namespace:
    """
    合并配置文件和命令行参数（命令行参数优先）
    
    Args:
        config: 从YAML加载的配置
        args: 命令行参数
        
    Returns:
        合并后的参数
    """
    # 参数名映射：config -> args
    param_mapping = {
        'learning_rate': 'lr',
        'num_epochs': 'epochs',  # ⭐ 添加映射：num_epochs -> epochs
        'warmup_epochs': 'warmup_epochs',
        'lr_warmup_epochs': 'warmup_epochs',  # ⭐ 添加映射：lr_warmup_epochs -> warmup_epochs
        'weight_decay': 'weight_decay',
        'batch_size': 'batch_size',
        'epochs': 'epochs',
        'eval_interval': 'eval_interval',
        'eval_every': 'eval_interval',  # ⭐ 添加映射：eval_every -> eval_interval
        'dropout': 'dropout'
    }
    
    # 模型配置
    if 'model' in config:
        model_config = config['model']
        for key, value in model_config.items():
            setattr(args, key, value)
            print(f"  ✓ Set model.{key} = {value}")
    
    # 损失权重配置（优先级最高）
    if 'loss' in config:
        loss_config = config['loss']
        for key, value in loss_config.items():
            if key.startswith('alpha_'):
                setattr(args, key, value)
                print(f"  ✓ Set loss.{key} = {value}")
    
    # 训练配置
    training_batch_size_from_config = None
    if 'training' in config:
        train_config = config['training']
        for key, value in train_config.items():
            if key == 'early_stopping':
                # 处理早停配置
                if isinstance(value, bool):
                    # 如果是布尔值，只设置是否启用
                    pass
                elif isinstance(value, dict):
                    if 'patience' in value:
                        args.early_stopping_patience = value['patience']
                    if 'min_delta' in value:
                        args.early_stopping_min_delta = value['min_delta']
            elif key == 'batch_size':
                # 显式优先使用 training.batch_size（避免被 data.batch_size 覆盖）
                setattr(args, 'batch_size', value)
                training_batch_size_from_config = value
                print(f"  ✓ Set batch_size = {value} (from training)")
            elif key in param_mapping:
                # 使用映射后的参数名（配置文件优先级高于默认值）
                arg_name = param_mapping[key]
                setattr(args, arg_name, value)
                print(f"  ✓ Set {arg_name} = {value}")
            else:
                # 其他参数直接设置（如果args中已有该属性，也覆盖）
                setattr(args, key, value)
                print(f"  ✓ Set {key} = {value}")
    
    # 数据配置
    if 'data' in config:
        data_config = config['data']
        for key, value in data_config.items():
            if key == 'batch_size' and training_batch_size_from_config is not None:
                # 如果已经在training中设置了batch_size，则跳过data.batch_size
                print(f"  ↪ Skip data.batch_size={value} (overridden by training.batch_size={training_batch_size_from_config})")
                continue
            if hasattr(args, key):
                setattr(args, key, value)
                print(f"  ✓ Set data.{key} = {value}")
    
    # 评估配置
    if 'evaluation' in config:
        eval_config = config['evaluation']
        for key, value in eval_config.items():
            if key in param_mapping:
                arg_name = param_mapping[key]
                setattr(args, arg_name, value)
                print(f"  ✓ Set evaluation.{key} -> {arg_name} = {value}")
            elif hasattr(args, key):
                setattr(args, key, value)
                print(f"  ✓ Set evaluation.{key} = {value}")
    
    # 日志配置
    if 'logging' in config:
        logging_config = config['logging']
        if 'use_tensorboard' in logging_config:
            args.use_tensorboard = logging_config['use_tensorboard']
        if 'log_dir' in logging_config:
            args.log_dir = logging_config['log_dir']
        if 'exp_name' in logging_config and logging_config['exp_name']:
            args.exp_name = logging_config['exp_name']
        if 'save_dir' in logging_config:
            args.save_dir = logging_config['save_dir']
    
    # 消融实验配置
    if 'ablation' in config:
        ablation_config = config['ablation']
        for key, value in ablation_config.items():
            setattr(args, f'ablation_{key}', value)
    
    return args


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        """
        Args:
            patience: 容忍的epoch数
            min_delta: 最小改进幅度
            mode: 'max' 或 'min'，表示指标越大越好还是越小越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        """
        检查是否应该早停
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
            
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False




def check_gradient_health(model: nn.Module, batch_idx: int) -> Dict[str, float]:
    """
    检查梯度健康状态，返回各模块的梯度统计
    
    Returns:
        各模块的梯度范数字典
    """
    gradient_stats = {
        'item_embedding': 0.0,
        'disentangled': 0.0,
        'quantum': 0.0,
        'causal': 0.0,
        'other': 0.0,
        'nan_count': 0,
        'inf_count': 0
    }
    
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
            
        grad_norm = param.grad.norm().item()
        
        # 检查异常梯度
        if torch.isnan(param.grad).any():
            gradient_stats['nan_count'] += 1
            print(f"⚠️  Batch {batch_idx}: NaN gradient in {name}")
            param.grad.zero_()
            continue
            
        if torch.isinf(param.grad).any():
            gradient_stats['inf_count'] += 1
            print(f"⚠️  Batch {batch_idx}: Inf gradient in {name}")
            param.grad.zero_()
            continue
        
        # 分类统计
        if 'item_embedding' in name:
            gradient_stats['item_embedding'] += grad_norm
        elif 'disentangled' in name:
            gradient_stats['disentangled'] += grad_norm
        elif 'quantum' in name:
            gradient_stats['quantum'] += grad_norm
        elif 'causal' in name:
            gradient_stats['causal'] += grad_norm
        else:
            gradient_stats['other'] += grad_norm
    
    return gradient_stats


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    grad_clip: float = 1.0
) -> Dict[str, float]:
    """训练一个epoch"""
    model.train()

    total_loss = 0
    total_rec_loss = 0
    total_dis_loss = 0
    total_div_loss = 0
    # 已移除：正交性损失累计
    total_causal_loss = 0
    
    # 梯度监控
    total_rec_grad = 0
    total_aux_grad = 0
    grad_samples = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(pbar):
        # 移动到设备
        item_ids = batch['item_ids'].to(device)
        target_items = batch['target_items'].to(device)
        seq_lengths = batch['seq_lengths'].to(device)
        multimodal_features = {
            k: v.to(device) for k, v in batch['multimodal_features'].items()
        }
        
        # ⭐ 负采样数据（如果有）
        candidate_items = batch.get('candidate_items')
        labels = batch.get('labels')
        if candidate_items is not None:
            candidate_items = candidate_items.to(device)
        if labels is not None:
            labels = labels.to(device)

        # 前向传播
        outputs = model(
            item_ids=item_ids,
            multimodal_features=multimodal_features,
            seq_lengths=seq_lengths,
            target_items=target_items,
            candidate_items=candidate_items,  # ⭐ 负采样候选物品
            labels=labels,  # ⭐ 标签
            return_loss=True,
            return_explanations=False
        )

        loss = outputs['loss']
        
        # ⭐ NaN检测：一旦发现loss为NaN立即终止并报告
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n{'='*80}")
            print(f"🚨 FATAL ERROR: Loss became {'NaN' if torch.isnan(loss) else 'Inf'} at batch {batch_idx}")
            print(f"{'='*80}")
            print(f"Loss breakdown:")
            for key in ['recommendation_loss', 'disentangled_loss', 'diversity_loss', 'causal_loss']:
                if key in outputs:
                    val = outputs[key]
                    print(f"  {key}: {val}")
            print(f"\nModel state:")
            print(f"  temperature: {model.temperature.item():.6f}")
            print(f"  kl_anneal_factor: {model.kl_anneal_factor:.6f}")
            print(f"\nThis indicates a critical numerical instability.")
            print(f"Please check the model architecture or reduce learning rate.")
            raise RuntimeError("Training failed due to NaN/Inf loss")

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # ⭐ 梯度健康检查（裁剪前）
        if batch_idx % 50 == 0:
            grad_health = check_gradient_health(model, batch_idx)
            
            rec_grad_norm_before = 0
            aux_grad_norm_before = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if 'recommendation_head' in name or 'item_embedding' in name:
                        rec_grad_norm_before += grad_norm
                    else:
                        aux_grad_norm_before += grad_norm
        
        # 梯度裁剪（从配置传入）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        
        # 梯度监控（每50个batch记录一次）- 裁剪后
        if batch_idx % 50 == 0:
            rec_grad_norm_after = 0
            aux_grad_norm_after = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    # 将序列编码器与量子编码器也计入rec梯度，占主损失链路
                    if ('item_embedding' in name or
                        'sequence_encoder' in name or
                        'quantum_encoder' in name or
                        'output_projection' in name):
                        rec_grad_norm_after += grad_norm
                    else:
                        aux_grad_norm_after += grad_norm
            
            # 记录裁剪后的梯度（用于计算平均值）
            total_rec_grad += rec_grad_norm_after
            total_aux_grad += aux_grad_norm_after
            grad_samples += 1
            
            # 如果梯度被严重裁剪，打印警告
            if aux_grad_norm_before > 100:  # 如果辅助梯度过大
                print(f"\n⚠️  Batch {batch_idx}: Gradient clipping applied!")
                print(f"   Before: rec={rec_grad_norm_before:.2f}, aux={aux_grad_norm_before:.2f}")
                print(f"   After:  rec={rec_grad_norm_after:.2f}, aux={aux_grad_norm_after:.2f}")
            
            # 如果检测到异常梯度
            if grad_health['nan_count'] > 0 or grad_health['inf_count'] > 0:
                print(f"\n⚠️  Batch {batch_idx}: Detected {grad_health['nan_count']} NaN and {grad_health['inf_count']} Inf gradients (已清零)")
        
        optimizer.step()

        # 记录损失
        total_loss += loss.item()
        total_rec_loss += outputs['recommendation_loss'].item()

        dis_loss = outputs['disentangled_loss']
        if isinstance(dis_loss, torch.Tensor):
            dis_loss = dis_loss.item()
        total_dis_loss += dis_loss

        div_loss = outputs['diversity_loss']
        if isinstance(div_loss, torch.Tensor):
            div_loss = div_loss.item()
        total_div_loss += div_loss
        
        # 已移除：正交性损失记录
        
        causal_loss = outputs.get('causal_loss', 0.0)
        if isinstance(causal_loss, torch.Tensor):
            causal_loss = causal_loss.item()
        total_causal_loss += causal_loss

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'rec': f'{outputs["recommendation_loss"].item():.4f}',
            'cau': f'{causal_loss:.4f}'
        })

    # 计算平均损失
    num_batches = len(dataloader)
    metrics = {
        'loss': total_loss / num_batches,
        'rec_loss': total_rec_loss / num_batches,
        'dis_loss': total_dis_loss / num_batches,
        'div_loss': total_div_loss / num_batches,
        'causal_loss': total_causal_loss / num_batches
    }
    
    # 添加梯度监控信息
    if grad_samples > 0:
        avg_rec_grad = total_rec_grad / grad_samples
        avg_aux_grad = total_aux_grad / grad_samples
        metrics['rec_grad_norm'] = avg_rec_grad
        metrics['aux_grad_norm'] = avg_aux_grad
        print(f"\n📊 梯度监控: rec_grad={avg_rec_grad:.2f}, aux_grad={avg_aux_grad:.2f}, ratio={avg_rec_grad/(avg_aux_grad+1e-8):.2f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train Multimodal Recommender on Amazon')

    # 数据参数
    parser.add_argument('--category', type=str, default='beauty',
                       choices=['beauty', 'games', 'sports'],
                       help='Amazon dataset category')
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--max_seq_length', type=int, default=50)
    parser.add_argument('--use_text_features', action='store_true',
                       help='Use text features (slower)')

    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--item_embed_dim', type=int, default=128)
    parser.add_argument('--disentangled_dim', type=int, default=64)
    parser.add_argument('--num_interests', type=int, default=4)
    parser.add_argument('--quantum_state_dim', type=int, default=128)

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)  # 降低学习率从3e-4到1e-4，防止NaN
    parser.add_argument('--warmup_epochs', type=int, default=5)  # 添加Warmup，前5个epoch线性增长
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    # 损失权重（默认值与config.yaml保持一致）
    parser.add_argument('--alpha_recon', type=float, default=0.1, help='解耦表征损失权重')
    parser.add_argument('--alpha_causal', type=float, default=0.2, help='因果推断损失权重')
    parser.add_argument('--alpha_diversity', type=float, default=0.05, help='多样性损失权重')
    # 已移除：alpha_orthogonality（简化后不再使用）

    # 评估参数
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--filter_train_items', action='store_true',
                       help='Filter training items during evaluation')

    # 配置文件
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file (overrides defaults)')
    
    # TensorBoard日志
    parser.add_argument('--use_tensorboard', action='store_true',
                       help='Enable TensorBoard logging')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='TensorBoard log directory')
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name for logging')
    
    # 消融实验参数
    parser.add_argument('--ablation_no_disentangled', action='store_true',
                       help='Ablation: disable disentangled representation')
    parser.add_argument('--ablation_no_causal', action='store_true',
                       help='Ablation: disable causal inference')
    parser.add_argument('--ablation_no_quantum', action='store_true',
                       help='Ablation: disable quantum-inspired encoder')
    parser.add_argument('--ablation_no_multimodal', action='store_true',
                       help='Ablation: use only item embeddings (no multimodal features)')
    parser.add_argument('--ablation_text_only', action='store_true',
                       help='Ablation: use only text features')
    parser.add_argument('--ablation_image_only', action='store_true',
                       help='Ablation: use only image features')
    
    # 其他
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='checkpoints')

    args = parser.parse_args()
    
    # 加载配置文件（如果提供）
    if args.config:
        print(f"\n{'='*80}")
        print(f"Loading config from {args.config}")
        print('='*80)
        config = load_config(args.config)
        args = merge_config_with_args(config, args)
        print('='*80)
        print("✓ Config loaded and merged successfully")
        print('='*80 + '\n')

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 生成实验名称
    if args.exp_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ablation_suffix = ""
        if args.ablation_no_disentangled:
            ablation_suffix += "_no_dis"
        if args.ablation_no_causal:
            ablation_suffix += "_no_cau"
        if args.ablation_no_quantum:
            ablation_suffix += "_no_qua"
        if args.ablation_no_multimodal:
            ablation_suffix += "_no_mm"
        if args.ablation_text_only:
            ablation_suffix += "_text"
        if args.ablation_image_only:
            ablation_suffix += "_image"
        args.exp_name = f"{args.category}_{timestamp}{ablation_suffix}"
    
    # 创建保存目录
    save_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化TensorBoard
    writer = None
    if args.use_tensorboard:
        log_path = os.path.join(args.log_dir, args.exp_name)
        os.makedirs(log_path, exist_ok=True)
        writer = SummaryWriter(log_path)
        print(f"✓ TensorBoard logging enabled: {log_path}")
        print(f"  Run: tensorboard --logdir={args.log_dir}")

    # 保存配置
    config_save_path = os.path.join(save_dir, 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"✓ Config saved to: {config_save_path}")

    print("\n" + "=" * 80)
    print(f"Training on Amazon {args.category.upper()} dataset")
    print(f"Experiment: {args.exp_name}")
    print("=" * 80)
    print(f"Device: {args.device}")
    
    # 打印消融实验设置
    ablation_info = []
    if args.ablation_no_disentangled:
        ablation_info.append("❌ Disentangled Representation")
    if args.ablation_no_causal:
        ablation_info.append("❌ Causal Inference")
    if args.ablation_no_quantum:
        ablation_info.append("❌ Quantum-Inspired Encoder")
    if args.ablation_no_multimodal:
        ablation_info.append("❌ Multimodal Features")
    if args.ablation_text_only:
        ablation_info.append("📝 Text Only")
    if args.ablation_image_only:
        ablation_info.append("🖼️  Image Only")
    
    if ablation_info:
        print("\n🔬 Ablation Study:")
        for info in ablation_info:
            print(f"  {info}")
    
    print("=" * 80 + "\n")

    # 创建数据加载器
    print("Loading data...")
    
    # ⭐ 从config读取高级配置
    advanced_config = config.get('advanced', {})
    # 负采样：支持 use_config_negatives
    use_config_negatives = advanced_config.get('use_config_negatives', False)
    num_negatives = advanced_config.get('num_negatives', 4) if use_config_negatives else 4
    # 温度与MC采样数
    use_config_temperature = advanced_config.get('use_config_temperature', False)
    use_config_mc_samples = advanced_config.get('use_config_mc_samples', False)
    temperature = advanced_config.get('temperature', 0.2) if use_config_temperature else 0.2
    num_mc_samples = advanced_config.get('num_mc_samples', 10) if use_config_mc_samples else 10
    # 因果模块集成数与目标ITE
    num_ensembles = advanced_config.get('num_ensembles', 5)
    target_ite = advanced_config.get('target_ite', 0.4)
    print(f"✓ Using {num_negatives} negative samples per positive")
    
    train_loader, valid_loader, test_loader, dataset_info = get_dataloaders(
        category=args.category,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_seq_length=args.max_seq_length,
        use_text_features=args.use_text_features,
        num_negatives=num_negatives  # ⭐ 使用config配置
    )

    num_users = dataset_info['num_users']
    num_items = dataset_info['num_items']

    print(f"Dataset: {num_users} users, {num_items} items")
    print(f"Train: {dataset_info['train_size']} samples")
    print(f"Valid: {dataset_info['valid_size']} samples")
    print(f"Test: {dataset_info['test_size']} samples\n")

    # 模态维度配置 (标准多模态: 文本 + 图像)
    modality_dims = {
        'text': 768,    # BERT-base / RoBERTa-base
        'image': 2048   # ResNet50
    }
    
    # ⭐ 从config读取投影层配置（新增）
    model_config = config.get('model', {})
    modality_proj_dims = model_config.get('modality_proj_dims', None)
    if modality_proj_dims:
        print(f"✓ Using modality projection: text {modality_dims['text']}→{modality_proj_dims['text']}, "
              f"image {modality_dims['image']}→{modality_proj_dims['image']}")

    # 创建模型
    print("Creating model...")
    # ⭐ 从config读取因果损失内部权重
    causal_loss_weights = config.get('loss', {}).get('causal_weights', {
        'magnitude': 1.0
    })
    # 从loss配置读取beta/gamma（用于解耦模块）
    loss_config = config.get('loss', {})
    beta = loss_config.get('beta', 0.5)
    # 已移除：gamma（简化后忽略该权重）
    
    model = MultimodalRecommender(
        modality_dims=modality_dims,
        modality_proj_dims=modality_proj_dims,  # ⭐ 新增：传递投影层配置
        disentangled_dim=args.disentangled_dim,
        num_disentangled_dims=3,
        num_interests=args.num_interests,
        quantum_state_dim=args.quantum_state_dim,
        hidden_dim=args.hidden_dim,
        item_embed_dim=args.item_embed_dim,
        num_items=num_items,
        max_seq_length=args.max_seq_length,
        alpha_recon=args.alpha_recon,
        alpha_causal=args.alpha_causal,
        alpha_diversity=args.alpha_diversity,
        # 已移除：alpha_orthogonality（不再传递）
        causal_loss_weights=causal_loss_weights,  # ⭐ 传递因果损失权重
        num_negatives=num_negatives,  # ⭐ 传递负采样数量
        use_quantum_computing=False,
        # ⭐ 从配置接线的额外参数
        beta=beta,
        # 已移除：gamma（不再传递）
        temperature=temperature,
        num_mc_samples=num_mc_samples,
        num_ensembles=num_ensembles,
        target_ite=target_ite
    ).to(args.device)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params / 1e6:.2f}M ({num_trainable / 1e6:.2f}M trainable)")
    print(f"  Temperature: {temperature} (use_config={use_config_temperature})")
    print(f"  Disentangled beta: beta={beta}")
    print(f"  MC samples: {num_mc_samples} (use_config={use_config_mc_samples})")
    print(f"  Causal ensembles: {num_ensembles}")
    print(f"  Target ITE: {target_ite}")

    # ⭐ 优化器 - 不同模块使用不同学习率
    print("\n" + "=" * 80)
    print("🔧 Creating optimizer with differentiated learning rates:")
    
    param_groups = []
    
    # 1. Item Embedding - 高学习率（推荐任务核心）
    item_emb_params = []
    if hasattr(model, 'item_embedding'):
        item_emb_params = list(model.item_embedding.parameters())
    param_groups.append({
        'params': item_emb_params,
        'lr': args.lr * 3.0,  # 3倍基础学习率（从2x提高，加速收敛）
        'weight_decay': args.weight_decay,
        'name': 'item_embedding'
    })
    print(f"  ✓ Item Embedding: lr={args.lr * 3.0:.2e}, {len(item_emb_params)} params")
    
    # 2. Disentangled Module - 中等学习率
    disentangled_params = []
    # 修正属性名：模型中为 disentangled_module
    if hasattr(model, 'disentangled_module'):
        disentangled_params = list(model.disentangled_module.parameters())
    param_groups.append({
        'params': disentangled_params,
        'lr': args.lr,
        'weight_decay': args.weight_decay * 2,  # 更强的正则化
        'name': 'disentangled'
    })
    print(f"  ✓ Disentangled Module: lr={args.lr:.2e}, {len(disentangled_params)} params")
    
    # 3. Quantum Encoder - 中等学习率
    quantum_params = []
    # 修正属性名：模型中为 quantum_encoder
    if hasattr(model, 'quantum_encoder'):
        quantum_params = list(model.quantum_encoder.parameters())
    param_groups.append({
        'params': quantum_params,
        'lr': args.lr * 0.5,
        'weight_decay': args.weight_decay,
        'name': 'quantum'
    })
    print(f"  ✓ Quantum Encoder: lr={args.lr * 0.5:.2e}, {len(quantum_params)} params")
    
    # 4. Causal Module - 低学习率（复杂模块）
    causal_params = []
    # 修正属性名：模型中为 causal_module
    if hasattr(model, 'causal_module'):
        causal_params = list(model.causal_module.parameters())
    param_groups.append({
        'params': causal_params,
        'lr': args.lr * 0.2,  # 低学习率（从0.1x提高到0.2x）
        'weight_decay': args.weight_decay,
        'name': 'causal'
    })
    print(f"  ✓ Causal Module: lr={args.lr * 0.2:.2e}, {len(causal_params)} params")
    
    # 5. 其他参数 - 基础学习率
    param_ids = set()
    for pg in param_groups:
        param_ids.update(id(p) for p in pg['params'])
    
    other_params = [p for p in model.parameters() if id(p) not in param_ids]
    param_groups.append({
        'params': other_params,
        'lr': args.lr,
        'weight_decay': args.weight_decay * 0.5,
        'name': 'others'
    })
    print(f"  ✓ Other Parameters: lr={args.lr:.2e}, {len(other_params)} params")
    
    optimizer = optim.AdamW(param_groups)
    print("=" * 80 + "\n")
    
    # 学习率调度器 (Warmup + Cosine Annealing) ⭐
    def lr_lambda(current_epoch):
        """Warmup + Cosine衰减"""
        if current_epoch < args.warmup_epochs:
            # Warmup阶段: 从1e-5线性增长到base_lr
            warm = (current_epoch + 1) / args.warmup_epochs
            return min(1.0, max(0.0, warm))
        else:
            # Cosine衰减阶段
            progress = (current_epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # 早停机制（从配置文件读取，如果没有则使用默认值）
    patience = getattr(args, 'early_stopping_patience', 10)
    min_delta = getattr(args, 'early_stopping_min_delta', 0.001)
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, mode='max')

    # 创建评估器
    evaluator = FullLibraryEvaluator(
        num_items=num_items,
        k_list=[5, 10, 20, 50]
    )

    # 获取训练集物品（用于过滤评估）
    if args.filter_train_items:
        print("Building train item filters...")
        train_items_per_user = get_train_items_per_user(train_loader.dataset)
        print(f"✓ Built filters for {len(train_items_per_user)} users\n")
    else:
        train_items_per_user = None

    # 训练循环
    best_ndcg = 0.0
    best_epoch = 0

    print("\n" + "=" * 80)
    print("🚀 Starting Training...")
    print(f"   Total Epochs: {args.epochs}")
    # 显示KL退火轮数（来自配置 kl_anneal_epochs）
    print(f"   KL Annealing: {getattr(args, 'kl_anneal_epochs', 20)} epochs")
    print(f"   Loss Weights: recon={args.alpha_recon}, causal={args.alpha_causal}, div={args.alpha_diversity}")
    # 评估过滤状态
    print(f"   Eval Filter: {'ON' if args.filter_train_items else 'OFF'}")
    print("=" * 80 + "\n")

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        print("-" * 80)
        
        # ⭐ KL退火：前N个epoch逐渐增加KL权重，避免VAE后验坍塌
        kl_anneal_epochs = getattr(args, 'kl_anneal_epochs', 20)
        if epoch <= kl_anneal_epochs:
            model.kl_anneal_factor = epoch / kl_anneal_epochs
            if epoch == 1:
                print(f"🔥 KL Annealing: factor will increase from 0 to 1.0 over {kl_anneal_epochs} epochs")
            if epoch % 5 == 0 or epoch == kl_anneal_epochs:
                print(f"   📊 KL anneal factor: {model.kl_anneal_factor:.3f}")
        else:
            model.kl_anneal_factor = 1.0

        # ⭐ 因果损失预热：前warmup_epochs逐步增权，避免早期干扰排序学习
        causal_warmup = getattr(args, 'warmup_epochs', 5)
        # 放缓因果损失预热：先到0.5，再逐步到1.0，避免早期辅助损失主导
        if causal_warmup > 0:
            if epoch <= causal_warmup:
                # 前warmup阶段线性到0.5
                model.alpha_causal_runtime = min(0.5, max(0.0, (epoch / float(causal_warmup)) * 0.5))
            else:
                # 之后逐步从0.5增长到1.0（默认20轮）
                causal_ramp_epochs = getattr(args, 'causal_ramp_epochs', 20)
                if causal_ramp_epochs > 0:
                    ramp_progress = min(1.0, max(0.0, (epoch - causal_warmup) / float(causal_ramp_epochs)))
                    model.alpha_causal_runtime = 0.5 + 0.5 * ramp_progress
                else:
                    model.alpha_causal_runtime = 1.0
        else:
            model.alpha_causal_runtime = 1.0
        if epoch == 1 or epoch % 5 == 0 or epoch == causal_warmup:
            print(f"   📊 Causal runtime weight: {model.alpha_causal_runtime:.3f}")
        
        # 训练
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            epoch,
            grad_clip=getattr(args, 'grad_clip', 1.0)
        )

        print(f"\nTraining metrics:")
        for key, value in train_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # TensorBoard: 记录训练指标
        if writer:
            for key, value in train_metrics.items():
                writer.add_scalar(f'Train/{key}', value, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # 验证
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            print(f"\nValidating...")

            if train_items_per_user is not None:
                valid_metrics = evaluator.evaluate_with_filter(
                    model, valid_loader, train_items_per_user, args.device
                )
            else:
                valid_metrics = evaluator.evaluate(
                    model, valid_loader, args.device
                )

            print(f"\nValidation metrics:")
            for key, value in valid_metrics.items():
                print(f"  {key}: {value:.4f}")
            
            # TensorBoard: 记录验证指标
            if writer:
                for key, value in valid_metrics.items():
                    writer.add_scalar(f'Valid/{key}', value, epoch)

            # 保存最佳模型
            if valid_metrics['NDCG@10'] > best_ndcg:
                best_ndcg = valid_metrics['NDCG@10']
                best_epoch = epoch

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_metrics': train_metrics,
                    'valid_metrics': valid_metrics,
                    'args': vars(args)
                }, os.path.join(save_dir, 'best_model.pt'))

                print(f"\n✓ Saved best model (NDCG@10: {best_ndcg:.4f})")
            
            # 早停检查
            if early_stopping(valid_metrics['NDCG@10']):
                print(f"\n✋ Early stopping triggered! No improvement for {early_stopping.patience} evaluations.")
                print(f"Best NDCG@10: {best_ndcg:.4f} at epoch {best_epoch}")
                break

        # 更新学习率
        scheduler.step()
        # ⭐ 学习率下限限制（lr_min）
        lr_min = getattr(args, 'lr_min', None)
        if lr_min is not None:
            for pg in optimizer.param_groups:
                if pg['lr'] < lr_min:
                    pg['lr'] = lr_min

        print(f"\nLearning rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("=" * 80 + "\n")

    # 在测试集上评估最佳模型
    print("\n" + "=" * 80)
    print("Testing best model...")
    print("=" * 80 + "\n")

    # 加载最佳模型（兼容 PyTorch 2.6 默认 weights_only=True）
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pt'), weights_only=False, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']}")

    # 测试
    if train_items_per_user is not None:
        test_metrics = evaluator.evaluate_with_filter(
            model, test_loader, train_items_per_user, args.device
        )
    else:
        test_metrics = evaluator.evaluate(
            model, test_loader, args.device
        )

    print(f"\nTest metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")

    # 保存测试结果
    results = {
        'exp_name': args.exp_name,
        'best_epoch': best_epoch,
        'valid_metrics': checkpoint['valid_metrics'],
        'test_metrics': test_metrics,
        'ablation_settings': {
            'no_disentangled': args.ablation_no_disentangled,
            'no_causal': args.ablation_no_causal,
            'no_quantum': args.ablation_no_quantum,
            'no_multimodal': args.ablation_no_multimodal,
            'text_only': args.ablation_text_only,
            'image_only': args.ablation_image_only
        }
    }

    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # TensorBoard: 记录最终测试指标
    if writer:
        for key, value in test_metrics.items():
            writer.add_scalar(f'Test/{key}', value, best_epoch)
        
        # 添加超参数和最终指标
        hparams = {
            'hidden_dim': args.hidden_dim,
            'item_embed_dim': args.item_embed_dim,
            'disentangled_dim': args.disentangled_dim,
            'num_interests': args.num_interests,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'alpha_recon': args.alpha_recon,
            'alpha_causal': args.alpha_causal,
            'alpha_diversity': args.alpha_diversity
        }
        writer.add_hparams(hparams, {
            'hparam/test_ndcg10': test_metrics['NDCG@10'],
            'hparam/test_hr10': test_metrics['HR@10'],
            'hparam/test_mrr': test_metrics['MRR']
        })
        
        writer.close()
        print(f"✓ TensorBoard logs saved to: {os.path.join(args.log_dir, args.exp_name)}")

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Experiment: {args.exp_name}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation NDCG@10: {checkpoint['valid_metrics']['NDCG@10']:.4f}")
    print(f"Test NDCG@10: {test_metrics['NDCG@10']:.4f}")
    print(f"Test HR@10: {test_metrics['HR@10']:.4f}")
    print(f"Test MRR: {test_metrics['MRR']:.4f}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
