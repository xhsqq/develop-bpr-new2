#!/usr/bin/env python
"""
提取BERT文本特征
从商品的标题和描述中提取语义特征
"""

import os
import sys
import pickle
import argparse
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_bert_features(
    category: str,
    data_dir: str = 'data/processed',
    model_name: str = 'bert-base-uncased',
    batch_size: int = 32,
    max_length: int = 128,
    device: str = None
):
    """
    离线提取BERT文本特征
    
    Args:
        category: 数据集类别 (beauty, games, sports)
        data_dir: 数据目录
        model_name: BERT模型名称
        batch_size: 批处理大小
        max_length: 最大文本长度
        device: 设备 (cuda/cpu)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 80)
    print(f"Extracting BERT Text Features for {category.upper()}")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Batch Size: {batch_size}")
    print(f"Max Length: {max_length}")
    print("=" * 80)
    
    # 1. 加载BERT模型
    print("\n[1/4] Loading BERT model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        model.to(device)
        print(f"✓ Loaded {model_name}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("\nPlease install transformers:")
        print("  pip install transformers")
        return
    
    # 2. 加载物品特征
    print("\n[2/4] Loading item features...")
    item_features_path = os.path.join(data_dir, category, 'item_features.pkl')
    
    if not os.path.exists(item_features_path):
        print(f"✗ Item features not found: {item_features_path}")
        print("Please run preprocess_amazon.py first")
        return
    
    with open(item_features_path, 'rb') as f:
        item_features = pickle.load(f)
    
    print(f"✓ Loaded {len(item_features)} items")
    
    # 3. 提取特征
    print(f"\n[3/4] Extracting BERT features...")
    text_features = {}
    
    # 准备所有文本
    item_ids = list(item_features.keys())
    texts = []
    
    for item_id in item_ids:
        feat = item_features[item_id]
        title = feat.get('title', '')
        description = feat.get('description', '')
        
        # 拼接标题和描述
        if title and description:
            text = f"{title} [SEP] {description}"
        elif title:
            text = title
        elif description:
            text = description
        else:
            text = ""  # 空文本
        
        texts.append(text)
    
    # 批处理提取特征
    num_batches = (len(texts) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(texts))
            
            batch_texts = texts[start_idx:end_idx]
            batch_ids = item_ids[start_idx:end_idx]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            ).to(device)
            
            # BERT编码
            outputs = model(**inputs)
            
            # 使用[CLS] token的表示
            batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # 保存特征
            for item_id, feat in zip(batch_ids, batch_features):
                text_features[item_id] = feat
    
    print(f"✓ Extracted {len(text_features)} text features")
    
    # 4. 保存特征
    print("\n[4/4] Saving features...")
    output_path = os.path.join(data_dir, category, 'text_features.pkl')
    
    with open(output_path, 'wb') as f:
        pickle.dump(text_features, f)
    
    print(f"✓ Saved to {output_path}")
    
    # 统计信息
    feature_dim = next(iter(text_features.values())).shape[0]
    print(f"\nStatistics:")
    print(f"  Total items: {len(text_features)}")
    print(f"  Feature dim: {feature_dim}")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    print("\n" + "=" * 80)
    print("✓ Text feature extraction complete!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Extract BERT text features')
    parser.add_argument(
        '--category',
        type=str,
        choices=['beauty', 'games', 'sports', 'all'],
        default='beauty',
        help='Dataset category'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed',
        help='Data directory'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='bert-base-uncased',
        help='BERT model name (bert-base-uncased, roberta-base, etc.)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for feature extraction'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=128,
        help='Maximum text length'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device (cuda/cpu)'
    )
    
    args = parser.parse_args()
    
    if args.category == 'all':
        categories = ['beauty', 'games', 'sports']
    else:
        categories = [args.category]
    
    for category in categories:
        extract_bert_features(
            category=category,
            data_dir=args.data_dir,
            model_name=args.model,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device
        )
        print()


if __name__ == '__main__':
    main()

