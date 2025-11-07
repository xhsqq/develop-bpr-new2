"""
Data loaders for Amazon datasets
"""

from torch.utils.data import DataLoader
from .dataset import AmazonDataset, collate_fn
from typing import Tuple, Dict


def get_dataloaders(
    category: str,
    data_dir: str = 'data/processed',
    batch_size: int = 64,
    num_workers: int = 4,
    max_seq_length: int = 50,
    use_text_features: bool = False,  # 默认False以加快速度
    pin_memory: bool = True,
    num_negatives: int = 500  # ⭐⭐⭐ 修复训练-评估Gap：增加到500（降低Gap从121x到24x）
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    获取训练、验证和测试数据加载器

    Args:
        category: 数据集类别
        data_dir: 数据目录
        batch_size: 批次大小
        num_workers: 数据加载进程数
        max_seq_length: 最大序列长度
        use_text_features: 是否使用文本特征
        pin_memory: 是否pin memory
        num_negatives: 负采样数量（训练集用）

    Returns:
        (train_loader, valid_loader, test_loader, dataset_info)
    """
    # 创建数据集
    train_dataset = AmazonDataset(
        category=category,
        split='train',
        data_dir=data_dir,
        max_seq_length=max_seq_length,
        use_text_features=use_text_features,
        num_negatives=num_negatives  # ⭐ 传递负采样参数
    )

    valid_dataset = AmazonDataset(
        category=category,
        split='valid',
        data_dir=data_dir,
        max_seq_length=max_seq_length,
        use_text_features=use_text_features
    )

    test_dataset = AmazonDataset(
        category=category,
        split='test',
        data_dir=data_dir,
        max_seq_length=max_seq_length,
        use_text_features=use_text_features
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory
    )

    # 数据集信息
    dataset_info = {
        'num_users': train_dataset.num_users,
        'num_items': train_dataset.num_items,
        'train_size': len(train_dataset),
        'valid_size': len(valid_dataset),
        'test_size': len(test_dataset),
        'max_seq_length': max_seq_length
    }

    print("\n" + "=" * 80)
    print(f"Data Loaders for {category.upper()}")
    print("=" * 80)
    print(f"Train: {len(train_dataset)} sequences, {len(train_loader)} batches")
    print(f"Valid: {len(valid_dataset)} sequences, {len(valid_loader)} batches")
    print(f"Test:  {len(test_dataset)} sequences, {len(test_loader)} batches")
    print(f"Users: {dataset_info['num_users']}, Items: {dataset_info['num_items']}")
    print("=" * 80 + "\n")

    return train_loader, valid_loader, test_loader, dataset_info
