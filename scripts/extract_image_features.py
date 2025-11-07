#!/usr/bin/env python
"""
提取ResNet图像特征
从商品图片中提取视觉特征

注意: Amazon数据集中的图片URL可能已失效
本脚本提供了两种方案:
1. 从URL下载图片并提取特征
2. 使用随机特征作为占位符（如果图片不可用）
"""

import os
import sys
import pickle
import argparse
import numpy as np
import torch
from tqdm import tqdm
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple, Optional

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_and_preprocess_image(
    item_id: int,
    image_url: str,
    transform: transforms.Compose,
    timeout: int
) -> Tuple[int, Optional[torch.Tensor], bool]:
    """
    下载并预处理单张图片
    
    Returns:
        (item_id, image_tensor, success)
    """
    if not image_url:
        return item_id, None, False
    
    try:
        response = requests.get(image_url, timeout=timeout)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img_tensor = transform(img)
        
        return item_id, img_tensor, True
        
    except Exception:
        return item_id, None, False


def extract_resnet_features(
    category: str,
    data_dir: str = 'data/processed',
    model_name: str = 'resnet50',
    batch_size: int = 32,
    device: str = None,
    download_timeout: int = 5,
    use_fallback: bool = True,
    num_workers: int = 16  # 并行下载线程数
):
    """
    离线提取ResNet图像特征
    
    Args:
        category: 数据集类别
        data_dir: 数据目录
        model_name: 模型名称 (resnet50, resnet101)
        batch_size: 批处理大小
        device: 设备
        download_timeout: 下载超时时间(秒)
        use_fallback: 如果图片不可用，是否使用fallback特征
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 80)
    print(f"Extracting ResNet Image Features for {category.upper()}")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Batch Size: {batch_size}")
    print(f"Download Timeout: {download_timeout}s")
    print(f"Use Fallback: {use_fallback}")
    print("=" * 80)
    
    # 1. 加载ResNet模型
    print("\n[1/4] Loading ResNet model...")
    try:
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
        elif model_name == 'resnet101':
            model = models.resnet101(pretrained=True)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # 移除最后的分类层，使用全局平均池化后的特征
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.eval()
        model.to(device)
        
        print(f"✓ Loaded {model_name}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("\nPlease install torchvision:")
        print("  pip install torchvision")
        return
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 2. 加载物品特征
    print("\n[2/4] Loading item features...")
    item_features_path = os.path.join(data_dir, category, 'item_features.pkl')
    
    if not os.path.exists(item_features_path):
        print(f"✗ Item features not found: {item_features_path}")
        return
    
    with open(item_features_path, 'rb') as f:
        item_features = pickle.load(f)
    
    print(f"✓ Loaded {len(item_features)} items")
    
    # 3. 并行下载和提取特征
    print(f"\n[3/4] Extracting ResNet features (parallel downloading)...")
    print(f"Workers: {num_workers} threads")
    print(f"Timeout: {download_timeout}s per image")
    
    image_features = {}
    failed_count = 0
    success_count = 0
    
    # 准备下载任务
    download_tasks = []
    for item_id, feat in item_features.items():
        image_url = feat.get('image', '') or feat.get('imUrl', '')
        download_tasks.append((item_id, image_url))
    
    # 并行下载图片
    downloaded_images = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有下载任务
        futures = {
            executor.submit(
                download_and_preprocess_image,
                item_id,
                image_url,
                transform,
                download_timeout
            ): item_id
            for item_id, image_url in download_tasks
        }
        
        # 处理完成的任务
        pbar = tqdm(total=len(download_tasks), desc="Downloading images")
        
        for future in as_completed(futures):
            item_id, img_tensor, success = future.result()
            downloaded_images.append((item_id, img_tensor, success))
            
            if success:
                success_count += 1
            else:
                failed_count += 1
            
            # 每100张显示一次成功率
            processed = success_count + failed_count
            if processed % 100 == 0:
                success_rate = success_count / processed * 100
                pbar.set_postfix({
                    'success': f'{success_rate:.1f}%',
                    'ok': success_count,
                    'fail': failed_count
                })
            
            pbar.update(1)
        
        pbar.close()
    
    print(f"✓ Download complete: {success_count}/{len(download_tasks)} succeeded ({success_count/len(download_tasks)*100:.1f}%)")
    
    # 4. 批量提取ResNet特征
    print(f"\n[4/5] Extracting ResNet features from downloaded images...")
    
    # 分离成功和失败的图片
    successful_items = [(item_id, img_tensor) for item_id, img_tensor, success in downloaded_images if success]
    failed_items = [item_id for item_id, _, success in downloaded_images if not success]
    
    # 批量处理成功下载的图片
    if successful_items:
        with torch.no_grad():
            for i in tqdm(range(0, len(successful_items), batch_size), desc="Extracting features"):
                batch_items = successful_items[i:i+batch_size]
                
                # 准备batch
                batch_tensors = torch.stack([img for _, img in batch_items]).to(device)
                
                # 提取特征
                feat_vecs = model(batch_tensors)  # (batch, 2048, 1, 1)
                feat_vecs = feat_vecs.squeeze(-1).squeeze(-1).cpu().numpy()  # (batch, 2048)
                
                # 保存特征
                for (item_id, _), feat_vec in zip(batch_items, feat_vecs):
                    image_features[item_id] = feat_vec.astype(np.float32)
    
    # 为失败的图片生成fallback特征
    if failed_items:
        for item_id in failed_items:
            if use_fallback:
                np.random.seed(item_id)
                image_features[item_id] = np.random.randn(2048).astype(np.float32)
            else:
                image_features[item_id] = np.zeros(2048, dtype=np.float32)
    
    print(f"\n✓ Extracted {len(image_features)} image features")
    print(f"  Real ResNet: {success_count} ({success_count/len(image_features)*100:.1f}%)")
    print(f"  Fallback:    {failed_count} ({failed_count/len(image_features)*100:.1f}%)")
    
    # 5. 保存特征
    print("\n[5/5] Saving features...")
    output_path = os.path.join(data_dir, category, 'image_features.pkl')
    
    with open(output_path, 'wb') as f:
        pickle.dump(image_features, f)
    
    print(f"✓ Saved to {output_path}")
    
    # 统计信息
    feature_dim = next(iter(image_features.values())).shape[0]
    print(f"\nStatistics:")
    print(f"  Total items: {len(image_features)}")
    print(f"  Feature dim: {feature_dim}")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    print("\n" + "=" * 80)
    print("✓ Image feature extraction complete!")
    print("=" * 80)
    
    if failed_count > 0 and success_count == 0:
        print("\n⚠ Warning: No images were successfully downloaded.")
        print("This is expected for Amazon datasets as image URLs may be outdated.")
        print("Using fallback features (deterministic based on item_id).")


def main():
    parser = argparse.ArgumentParser(description='Extract ResNet image features')
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
        default='resnet50',
        choices=['resnet50', 'resnet101'],
        help='ResNet model name'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=5,
        help='Download timeout in seconds'
    )
    parser.add_argument(
        '--no_fallback',
        action='store_true',
        help='Do not use fallback features (use zeros instead)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device (cuda/cpu)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=16,
        help='Number of parallel download threads'
    )
    
    args = parser.parse_args()
    
    if args.category == 'all':
        categories = ['beauty', 'games', 'sports']
    else:
        categories = [args.category]
    
    for category in categories:
        extract_resnet_features(
            category=category,
            data_dir=args.data_dir,
            model_name=args.model,
            batch_size=args.batch_size,
            device=args.device,
            download_timeout=args.timeout,
            use_fallback=not args.no_fallback,
            num_workers=args.num_workers
        )
        print()


if __name__ == '__main__':
    main()

