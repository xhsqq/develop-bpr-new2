"""
Dataset classes for Amazon sequential recommendation
"""

import os
import pickle
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import numpy as np
from transformers import AutoTokenizer


class AmazonDataset(Dataset):
    """
    Amazon序列推荐数据集
    支持全库评估（无负采样）
    """

    def __init__(
        self,
        category: str,
        split: str = 'train',
        data_dir: str = 'data/processed',
        max_seq_length: int = 50,
        text_encoder: str = 'bert-base-uncased',
        use_text_features: bool = True,
        num_negatives: int = 4  # ⭐ 新增：负采样数量
    ):
        """
        Args:
            category: 数据集类别 ('beauty', 'games', 'sports')
            split: 数据划分 ('train', 'valid', 'test')
            data_dir: 处理后数据的目录
            max_seq_length: 最大序列长度
            text_encoder: 文本编码器名称
            use_text_features: 是否使用文本特征
            num_negatives: 负采样数量（训练时使用，评估时为0表示全库）
        """
        self.category = category
        self.split = split
        self.data_dir = os.path.join(data_dir, category)
        self.max_seq_length = max_seq_length
        self.use_text_features = use_text_features
        self.num_negatives = num_negatives if split == 'train' else 0  # 只在训练时负采样

        # 加载数据
        self._load_data()

        # 初始化文本tokenizer
        if use_text_features:
            print(f"Loading tokenizer: {text_encoder}")
            self.tokenizer = AutoTokenizer.from_pretrained(text_encoder)
        else:
            self.tokenizer = None
        
        # ⭐ 构建物品集合（用于负采样）
        if self.num_negatives > 0:
            self.all_items = list(range(1, self.num_items + 1))  # 物品ID从1开始
            print(f"✓ Negative sampling enabled: {self.num_negatives} negatives per positive")

    def _load_data(self):
        """加载预处理后的数据"""
        # 加载序列数据
        sequences_path = os.path.join(self.data_dir, f'{self.split}_sequences.pkl')
        with open(sequences_path, 'rb') as f:
            self.sequences = pickle.load(f)

        # 加载映射
        mappings_path = os.path.join(self.data_dir, 'mappings.pkl')
        with open(mappings_path, 'rb') as f:
            mappings = pickle.load(f)
            self.num_users = mappings['num_users']
            self.num_items = mappings['num_items']
            self.id2item = mappings['id2item']

        # 加载物品特征
        features_path = os.path.join(self.data_dir, 'item_features.pkl')
        with open(features_path, 'rb') as f:
            self.item_features = pickle.load(f)

        # 加载预计算的多模态特征
        self.precomputed_text = None
        self.precomputed_image = None
        
        text_feat_path = os.path.join(self.data_dir, 'text_features.pkl')
        image_feat_path = os.path.join(self.data_dir, 'image_features.pkl')
        
        if os.path.exists(text_feat_path):
            print(f"Loading precomputed text features...")
            with open(text_feat_path, 'rb') as f:
                self.precomputed_text = pickle.load(f)
            print(f"✓ Loaded {len(self.precomputed_text)} text features")
        else:
            print(f"⚠ Text features not found at {text_feat_path}")
            print(f"  Run: python scripts/extract_text_features.py --category {self.category}")
        
        if os.path.exists(image_feat_path):
            print(f"Loading precomputed image features...")
            with open(image_feat_path, 'rb') as f:
                self.precomputed_image = pickle.load(f)
            print(f"✓ Loaded {len(self.precomputed_image)} image features")
        else:
            print(f"⚠ Image features not found at {image_feat_path}")
            print(f"  Run: python scripts/extract_image_features.py --category {self.category}")

        print(f"Loaded {len(self.sequences)} sequences for {self.split}")
        print(f"Number of users: {self.num_users}, Number of items: {self.num_items}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        user_id = sequence['user_id']
        history = sequence['history']
        target = sequence['target']
        seq_length = sequence['seq_length']

        # Pad序列到最大长度
        padded_history = self._pad_sequence(history)

        # 获取多模态特征
        multimodal_features = self._get_multimodal_features(padded_history)

        # ⭐ 负采样（仅训练集）
        if self.num_negatives > 0:
            # 采样负样本（排除历史交互过的物品）
            history_set = set(history + [target])
            negative_items = []
            attempts = 0
            max_attempts = self.num_negatives * 10
            
            while len(negative_items) < self.num_negatives and attempts < max_attempts:
                neg_item = np.random.choice(self.all_items)
                if neg_item not in history_set and neg_item not in negative_items:
                    negative_items.append(neg_item)
                attempts += 1
            
            # 构建候选物品（1正 + K负）
            candidate_items = [target] + negative_items
            labels = [1.0] + [0.0] * len(negative_items)
            
            return {
                'user_id': torch.tensor(user_id, dtype=torch.long),
                'item_ids': torch.tensor(padded_history, dtype=torch.long),
                'target_item': torch.tensor(target, dtype=torch.long),
                'candidate_items': torch.tensor(candidate_items, dtype=torch.long),  # ⭐ 候选物品
                'labels': torch.tensor(labels, dtype=torch.float32),  # ⭐ 标签 (1正 + K个0负)
                'seq_length': torch.tensor(seq_length, dtype=torch.long),
                'multimodal_features': multimodal_features
            }
        else:
            # 评估时不负采样，使用全库
            return {
                'user_id': torch.tensor(user_id, dtype=torch.long),
                'item_ids': torch.tensor(padded_history, dtype=torch.long),
                'target_item': torch.tensor(target, dtype=torch.long),
                'seq_length': torch.tensor(seq_length, dtype=torch.long),
                'multimodal_features': multimodal_features
            }

    def _pad_sequence(self, sequence: List[int]) -> List[int]:
        """Pad或截断序列到固定长度"""
        if len(sequence) >= self.max_seq_length:
            return sequence[-self.max_seq_length:]
        else:
            # 0 用于padding
            return [0] * (self.max_seq_length - len(sequence)) + sequence

    def _get_multimodal_features(self, item_ids: List[int]) -> Dict[str, torch.Tensor]:
        """
        获取物品的多模态特征

        返回:
            - text: 文本特征 (seq_len, text_dim)
            - image: 图像特征 (seq_len, image_dim)
        """
        features = {
            'text': [],
            'image': []
        }

        for item_id in item_ids:
            if item_id == 0:  # Padding
                # Padding特征
                text_feat = torch.zeros(768)   # BERT/RoBERTa维度
                image_feat = torch.zeros(2048)  # ResNet50维度
            else:
                # 获取真实特征
                item_feat = self.item_features.get(item_id, {})

                # 文本特征 (标题 + 描述) - 传入item_id
                text_feat = self._encode_text(item_id, item_feat)

                # 图像特征 - 传入item_id
                image_feat = self._encode_image(item_id, item_feat)

            features['text'].append(text_feat)
            features['image'].append(image_feat)

        # Stack所有特征
        features['text'] = torch.stack(features['text'])
        features['image'] = torch.stack(features['image'])

        return features

    def _encode_text(self, item_id: int, item_features: Dict) -> torch.Tensor:
        """
        编码文本特征 (Title + Description)
        
        优先使用预计算的BERT特征，如果不可用则使用fallback
        """
        # 1. 优先使用预计算特征
        if self.precomputed_text is not None and item_id in self.precomputed_text:
            feat = self.precomputed_text[item_id]
            return torch.from_numpy(feat).float()
        
        # 2. Fallback: 基于item_id的确定性特征
        # 使用item_id作为随机种子，确保相同item_id总是得到相同特征
        np.random.seed(item_id % 2**32)
        return torch.from_numpy(np.random.randn(768).astype(np.float32))

    def _encode_image(self, item_id: int, item_features: Dict) -> torch.Tensor:
        """
        编码图像特征
        
        优先使用预计算的ResNet特征，如果不可用则使用fallback
        """
        # 1. 优先使用预计算特征
        if self.precomputed_image is not None and item_id in self.precomputed_image:
            feat = self.precomputed_image[item_id]
            return torch.from_numpy(feat).float()
        
        # 2. Fallback: 基于item_id的确定性特征
        # 使用item_id作为随机种子，确保相同item_id总是得到相同特征
        np.random.seed(item_id % 2**32)
        return torch.from_numpy(np.random.randn(2048).astype(np.float32))


class SequentialDataset(Dataset):
    """
    简化的序列数据集
    用于快速测试
    """

    def __init__(
        self,
        sequences: List[Dict],
        num_items: int,
        max_seq_length: int = 50
    ):
        self.sequences = sequences
        self.num_items = num_items
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        user_id = sequence['user_id']
        history = sequence['history']
        target = sequence['target']
        seq_length = min(len(history), self.max_seq_length)

        # Pad序列
        if len(history) >= self.max_seq_length:
            padded_history = history[-self.max_seq_length:]
        else:
            padded_history = [0] * (self.max_seq_length - len(history)) + history

        # 生成随机多模态特征 (文本 + 图像)
        multimodal_features = {
            'text': torch.randn(self.max_seq_length, 768),    # BERT特征
            'image': torch.randn(self.max_seq_length, 2048)   # ResNet特征
        }

        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'item_ids': torch.tensor(padded_history, dtype=torch.long),
            'target_item': torch.tensor(target, dtype=torch.long),
            'seq_length': torch.tensor(seq_length, dtype=torch.long),
            'multimodal_features': multimodal_features
        }


def collate_fn(batch):
    """自定义collate函数"""
    user_ids = torch.stack([item['user_id'] for item in batch])
    item_ids = torch.stack([item['item_ids'] for item in batch])
    target_items = torch.stack([item['target_item'] for item in batch])
    seq_lengths = torch.stack([item['seq_length'] for item in batch])

    # 收集多模态特征
    modality_keys = batch[0]['multimodal_features'].keys()
    multimodal_features = {
        modality: torch.stack([item['multimodal_features'][modality] for item in batch])
        for modality in modality_keys
    }

    result = {
        'user_ids': user_ids,
        'item_ids': item_ids,
        'target_items': target_items,
        'seq_lengths': seq_lengths,
        'multimodal_features': multimodal_features
    }
    
    # ⭐ 处理负采样数据（如果存在）
    if 'candidate_items' in batch[0]:
        result['candidate_items'] = torch.stack([item['candidate_items'] for item in batch])
    if 'labels' in batch[0]:
        result['labels'] = torch.stack([item['labels'] for item in batch])
    
    return result
