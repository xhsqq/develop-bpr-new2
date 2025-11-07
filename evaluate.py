"""
评估脚本 - 加载已训练模型并在测试集上评估
Usage:
    python evaluate.py --checkpoint checkpoints/beauty_20231201_120000/best_model.pt --category beauty
"""

import torch
import argparse
import json
import os
from typing import Dict

from models.multimodal_recommender import MultimodalRecommender
from data.dataloader import get_dataloaders
from utils.evaluation import FullLibraryEvaluator, get_train_items_per_user


def load_model_from_checkpoint(checkpoint_path: str, device: str = 'cuda') -> tuple:
    """
    从检查点加载模型

    Args:
        checkpoint_path: 检查点路径
        device: 设备

    Returns:
        (model, checkpoint_dict)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    # PyTorch 2.6 默认 weights_only=True 会导致包含非权重对象的字典无法反序列化
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)

    # 从checkpoint中获取配置
    if 'args' in checkpoint:
        args = argparse.Namespace(**checkpoint['args'])
    else:
        raise ValueError("Checkpoint does not contain 'args'. Cannot reconstruct model.")

    # 重建模型
    modality_dims = {
        'text': 768,
        'image': 2048
    }

    model = MultimodalRecommender(
        modality_dims=modality_dims,
        disentangled_dim=args.disentangled_dim,
        num_disentangled_dims=3,
        num_interests=args.num_interests,
        quantum_state_dim=args.quantum_state_dim,
        hidden_dim=args.hidden_dim,
        item_embed_dim=args.item_embed_dim,
        num_items=checkpoint['model_state_dict']['item_embedding.weight'].size(0) - 1,  # 减去padding
        max_seq_length=args.max_seq_length,
        alpha_recon=args.alpha_recon,
        alpha_causal=args.alpha_causal,
        alpha_diversity=args.alpha_diversity,
        use_quantum_computing=False
    ).to(device)

    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("✓ Model loaded successfully")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    return model, checkpoint


def evaluate_model(
    model,
    test_loader,
    evaluator: FullLibraryEvaluator,
    device: str,
    train_items_per_user: Dict = None,
    show_progress: bool = True
) -> Dict[str, float]:
    """
    评估模型

    Args:
        model: 推荐模型
        test_loader: 测试数据加载器
        evaluator: 评估器
        device: 设备
        train_items_per_user: 训练集物品（用于过滤）
        show_progress: 是否显示进度

    Returns:
        评估指标字典
    """
    print("\n" + "=" * 80)
    print("Evaluating model on test set...")
    print("=" * 80)

    if train_items_per_user is not None:
        print("Using filtered evaluation (excluding training items)")
        metrics = evaluator.evaluate_with_filter(
            model, test_loader, train_items_per_user, device
        )
    else:
        print("Using full evaluation (including all items)")
        metrics = evaluator.evaluate(
            model, test_loader, device
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained multimodal recommender')

    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (e.g., checkpoints/exp_name/best_model.pt)')
    parser.add_argument('--category', type=str, required=True,
                       choices=['beauty', 'games', 'sports'],
                       help='Amazon dataset category')

    # 数据参数
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_text_features', action='store_true',
                       help='Use text features (slower)')

    # 评估参数
    parser.add_argument('--filter_train_items', action='store_true',
                       help='Filter training items during evaluation')
    parser.add_argument('--k_list', type=int, nargs='+', default=[5, 10, 20, 50],
                       help='List of K values for Top-K evaluation')

    # 其他
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results (default: same dir as checkpoint)')

    args = parser.parse_args()

    # 检查检查点文件是否存在
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print("\n" + "=" * 80)
    print("Multimodal Recommender - Model Evaluation")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Category: {args.category}")
    print(f"Device: {args.device}")
    print("=" * 80 + "\n")

    # 加载模型
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, args.device)

    # 获取模型配置中的 max_seq_length
    if 'args' in checkpoint:
        max_seq_length = checkpoint['args']['max_seq_length']
    else:
        max_seq_length = 50  # 默认值

    # 加载数据
    print("\nLoading test data...")
    _, _, test_loader, dataset_info = get_dataloaders(
        category=args.category,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_seq_length=max_seq_length,
        use_text_features=args.use_text_features,
        num_negatives=0
    )

    print(f"✓ Test set: {dataset_info['test_size']} samples")
    print(f"✓ Dataset: {dataset_info['num_users']} users, {dataset_info['num_items']} items\n")

    # 创建评估器
    evaluator = FullLibraryEvaluator(
        num_items=dataset_info['num_items'],
        k_list=args.k_list
    )

    # 获取训练集物品（用于过滤评估）
    train_items_per_user = None
    if args.filter_train_items:
        print("Building train item filters...")
        # 需要加载训练集
        train_loader, _, _, _ = get_dataloaders(
            category=args.category,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_seq_length=max_seq_length,
            use_text_features=args.use_text_features,
            num_negatives=0
        )
        train_items_per_user = get_train_items_per_user(train_loader.dataset)
        print(f"✓ Built filters for {len(train_items_per_user)} users\n")

    # 评估模型
    test_metrics = evaluate_model(
        model,
        test_loader,
        evaluator,
        args.device,
        train_items_per_user
    )

    # 打印结果
    print("\n" + "=" * 80)
    print("📊 Test Results")
    print("=" * 80)
    for key, value in sorted(test_metrics.items()):
        print(f"  {key}: {value:.4f}")
    print("=" * 80 + "\n")

    # 保存结果
    if args.output is None:
        # 默认保存在checkpoint同目录
        checkpoint_dir = os.path.dirname(args.checkpoint)
        args.output = os.path.join(checkpoint_dir, 'evaluation_results.json')

    results = {
        'checkpoint': args.checkpoint,
        'category': args.category,
        'test_metrics': test_metrics,
        'config': {
            'filter_train_items': args.filter_train_items,
            'k_list': args.k_list,
            'batch_size': args.batch_size
        }
    }

    # 如果checkpoint中有训练指标，也保存
    if 'valid_metrics' in checkpoint:
        results['valid_metrics'] = checkpoint['valid_metrics']
    if 'train_metrics' in checkpoint:
        results['train_metrics'] = checkpoint['train_metrics']

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"✓ Results saved to: {args.output}\n")

    # 如果checkpoint中有验证集指标，比较一下
    if 'valid_metrics' in checkpoint and 'NDCG@10' in checkpoint['valid_metrics']:
        valid_ndcg = checkpoint['valid_metrics']['NDCG@10']
        test_ndcg = test_metrics['NDCG@10']
        print("=" * 80)
        print("📈 Performance Comparison")
        print("=" * 80)
        print(f"  Validation NDCG@10: {valid_ndcg:.4f}")
        print(f"  Test NDCG@10:       {test_ndcg:.4f}")
        print(f"  Difference:         {test_ndcg - valid_ndcg:+.4f}")
        print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
