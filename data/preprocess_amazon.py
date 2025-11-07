"""
Preprocess Amazon datasets with leave-one-out splitting
确保没有数据泄漏：训练集不能看到验证/测试集的物品特征
"""

import json
import os
import pickle
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm


class AmazonPreprocessor:
    """Amazon数据集预处理器，使用留一法划分"""

    def __init__(
        self,
        category: str,
        raw_dir: str = 'data/raw',
        processed_dir: str = 'data/processed',
        min_user_interactions: int = 5,
        min_item_interactions: int = 5
    ):
        self.category = category
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions

        os.makedirs(processed_dir, exist_ok=True)

        # 映射字典
        self.user2id = {}
        self.item2id = {}
        self.id2item = {}

    def load_raw_data(self) -> Tuple[List[Dict], Dict]:
        """加载原始数据"""
        reviews_path = os.path.join(self.raw_dir, f'{self.category}_reviews.json')
        meta_path = os.path.join(self.raw_dir, f'{self.category}_meta.json')

        print(f"Loading reviews from {reviews_path}...")
        with open(reviews_path, 'r') as f:
            reviews = json.load(f)

        print(f"Loading metadata from {meta_path}...")
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        # 创建metadata字典
        meta_dict = {}
        for item in tqdm(metadata, desc="Building metadata dict"):
            if 'asin' in item:
                meta_dict[item['asin']] = item

        print(f"Loaded {len(reviews)} reviews and {len(meta_dict)} items")

        return reviews, meta_dict

    def filter_k_core(self, reviews: List[Dict], k: int = 5) -> List[Dict]:
        """
        K-core filtering: 保留至少有k个交互的用户和物品
        迭代过滤直到收敛
        """
        print(f"\nApplying {k}-core filtering...")

        filtered_reviews = reviews.copy()
        prev_size = len(filtered_reviews) + 1

        iteration = 0
        while prev_size > len(filtered_reviews):
            prev_size = len(filtered_reviews)
            iteration += 1

            # 统计用户和物品的交互次数
            user_counts = defaultdict(int)
            item_counts = defaultdict(int)

            for review in filtered_reviews:
                user_counts[review['reviewerID']] += 1
                item_counts[review['asin']] += 1

            # 过滤
            filtered_reviews = [
                review for review in filtered_reviews
                if user_counts[review['reviewerID']] >= k and
                   item_counts[review['asin']] >= k
            ]

            print(f"  Iteration {iteration}: {len(filtered_reviews)} reviews remaining")

        print(f"✓ K-core filtering complete: {len(filtered_reviews)} reviews")

        return filtered_reviews

    def build_mappings(self, reviews: List[Dict]):
        """构建用户和物品的ID映射"""
        print("\nBuilding user and item mappings...")

        users = sorted(set(review['reviewerID'] for review in reviews))
        items = sorted(set(review['asin'] for review in reviews))

        # 0保留给padding
        self.user2id = {user: idx + 1 for idx, user in enumerate(users)}
        self.item2id = {item: idx + 1 for idx, item in enumerate(items)}
        self.id2item = {idx: item for item, idx in self.item2id.items()}

        print(f"✓ {len(self.user2id)} users, {len(self.item2id)} items")

    def leave_one_out_split(
        self,
        reviews: List[Dict]
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        留一法划分数据集
        - 最后一个交互作为测试集
        - 倒数第二个交互作为验证集
        - 其余作为训练集

        确保没有数据泄漏：
        - 按时间戳排序
        - 训练集中的物品必须在验证/测试集之前出现
        """
        print("\nApplying leave-one-out splitting...")

        # 按用户组织交互
        user_interactions = defaultdict(list)
        for review in reviews:
            user_id = self.user2id[review['reviewerID']]
            item_id = self.item2id[review['asin']]
            timestamp = review.get('unixReviewTime', 0)

            user_interactions[user_id].append({
                'user_id': user_id,
                'item_id': item_id,
                'rating': review.get('overall', 0),
                'timestamp': timestamp,
                'review_text': review.get('reviewText', ''),
                'summary': review.get('summary', '')
            })

        # 按时间戳排序每个用户的交互
        for user_id in user_interactions:
            user_interactions[user_id].sort(key=lambda x: x['timestamp'])

        train_data = []
        valid_data = []
        test_data = []

        # 留一法划分
        for user_id, interactions in tqdm(user_interactions.items(), desc="Splitting"):
            n_interactions = len(interactions)

            if n_interactions < 3:
                # 如果交互少于3个，全部放入训练集
                train_data.extend(interactions)
            else:
                # 训练集：前n-2个
                train_data.extend(interactions[:-2])
                # 验证集：倒数第2个
                valid_data.append(interactions[-2])
                # 测试集：最后1个
                test_data.append(interactions[-1])

        print(f"✓ Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")

        return train_data, valid_data, test_data

    def build_sequential_data(
        self,
        train_data: List[Dict],
        valid_data: List[Dict],
        test_data: List[Dict],
        max_seq_length: int = 50
    ) -> Dict:
        """
        构建序列数据
        对于每个用户，构建历史序列用于预测下一个物品

        重要：确保没有数据泄漏
        - 验证集：只使用训练集的历史
        - 测试集：使用训练集+验证集的历史
        - 过滤冷启动物品：验证/测试集中的target必须在训练集中出现过
        """
        print("\nBuilding sequential data...")

        # 组织每个用户的训练交互序列
        user_train_sequences = defaultdict(list)
        for interaction in train_data:
            user_train_sequences[interaction['user_id']].append(interaction['item_id'])

        # ⭐ 修改点1：收集训练集中的所有物品
        train_items = set()
        for item_seq in user_train_sequences.values():
            train_items.update(item_seq)
        print(f"  训练集中的物品数: {len(train_items)}")

        # 构建训练样本
        train_sequences = []
        for user_id, item_sequence in user_train_sequences.items():
            if len(item_sequence) < 2:
                continue

            # 对于训练集，使用滑动窗口生成多个样本
            for i in range(1, len(item_sequence)):
                history = item_sequence[:i][-max_seq_length:]  # 最多保留max_seq_length个
                target = item_sequence[i]

                train_sequences.append({
                    'user_id': user_id,
                    'history': history,
                    'target': target,
                    'seq_length': len(history)
                })

        # ⭐ 修改点2：构建验证样本（只使用训练集历史，过滤冷启动物品）
        valid_sequences = []
        valid_cold_count = 0
        
        for interaction in valid_data:
            user_id = interaction['user_id']
            target_item = interaction['item_id']
            history = user_train_sequences[user_id][-max_seq_length:]

            # ⭐ 冷启动检查：target必须在训练集中出现过
            if target_item not in train_items:
                valid_cold_count += 1
                continue  # 跳过冷启动物品

            if len(history) > 0:
                valid_sequences.append({
                    'user_id': user_id,
                    'history': history,
                    'target': target_item,
                    'seq_length': len(history)
                })

        # ⭐ 修改点3：扩展可用物品集合（训练集 + 验证集）
        valid_items = train_items.copy()
        for seq in valid_sequences:
            valid_items.add(seq['target'])

        # ⭐ 修改点3：构建测试样本（使用训练集+验证集历史，过滤冷启动物品）
        test_sequences = []
        test_cold_count = 0
        
        for interaction in test_data:
            user_id = interaction['user_id']
            target_item = interaction['item_id']

            # 历史 = 训练集交互 + 验证集交互
            history = user_train_sequences[user_id].copy()

            # 添加验证集的item（如果存在）
            valid_items_for_user = [v['item_id'] for v in valid_data if v['user_id'] == user_id]
            if valid_items_for_user:
                history.append(valid_items_for_user[0])

            history = history[-max_seq_length:]

            # ⭐ 冷启动检查：target必须在(训练集+验证集)中出现过
            if target_item not in valid_items:
                test_cold_count += 1
                continue  # 跳过冷启动物品

            if len(history) > 0:
                test_sequences.append({
                    'user_id': user_id,
                    'history': history,
                    'target': target_item,
                    'seq_length': len(history)
                })

        # ⭐ 修改点4：添加统计输出
        print(f"✓ Train sequences: {len(train_sequences)}")
        print(f"✓ Valid sequences: {len(valid_sequences)}")
        if valid_cold_count > 0:
            total_valid = len(valid_sequences) + valid_cold_count
            print(f"  └─ 过滤了 {valid_cold_count} 个冷启动样本 ({valid_cold_count/total_valid*100:.1f}%)")
        
        print(f"✓ Test sequences: {len(test_sequences)}")
        if test_cold_count > 0:
            total_test = len(test_sequences) + test_cold_count
            print(f"  └─ 过滤了 {test_cold_count} 个冷启动样本 ({test_cold_count/total_test*100:.1f}%)")

        return {
            'train': train_sequences,
            'valid': valid_sequences,
            'test': test_sequences
        }

    def extract_item_features(self, meta_dict: Dict) -> Dict:
        """
        提取物品的多模态特征

        注意：只提取训练集中出现的物品的特征，避免数据泄漏
        """
        print("\nExtracting item features...")

        item_features = {}

        for item_orig_id, item_id in tqdm(self.item2id.items(), desc="Processing items"):
            if item_orig_id not in meta_dict:
                # 如果没有metadata，使用默认值
                item_features[item_id] = {
                    'title': '',
                    'description': '',
                    'brand': '',
                    'categories': [],
                    'price': 0.0,
                    'image': ''  # 添加空的图片URL
                }
                continue

            meta = meta_dict[item_orig_id]

            # 提取文本特征
            title = meta.get('title', '')
            description = meta.get('description', '')
            if isinstance(description, list):
                description = ' '.join(description)

            brand = meta.get('brand', '')
            categories = meta.get('category', [])
            if isinstance(categories, str):
                categories = [categories]

            price = meta.get('price', 0.0)
            if isinstance(price, str):
                try:
                    price = float(price.replace('$', '').replace(',', ''))
                except:
                    price = 0.0

            # 提取图片URL（用于真实图像特征提取）
            image_url = meta.get('imUrl', '') or meta.get('image', '')
            
            item_features[item_id] = {
                'title': title,
                'description': description,
                'brand': brand,
                'categories': categories,
                'price': price,
                'image': image_url  # 添加图片URL
            }

        print(f"✓ Extracted features for {len(item_features)} items")

        return item_features

    def get_statistics(self, sequences: Dict) -> Dict:
        """计算数据集统计信息"""
        stats = {}

        for split in ['train', 'valid', 'test']:
            data = sequences[split]

            seq_lengths = [s['seq_length'] for s in data]
            unique_users = len(set(s['user_id'] for s in data))
            unique_items = len(set(s['target'] for s in data))

            stats[split] = {
                'num_sequences': len(data),
                'num_users': unique_users,
                'num_items': unique_items,
                'avg_seq_length': float(np.mean(seq_lengths)),
                'min_seq_length': int(np.min(seq_lengths)),
                'max_seq_length': int(np.max(seq_lengths)),
            }

        return stats

    def save_processed_data(
        self,
        sequences: Dict,
        item_features: Dict,
        stats: Dict
    ):
        """保存处理后的数据"""
        output_dir = os.path.join(self.processed_dir, self.category)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nSaving processed data to {output_dir}...")

        # 保存序列数据
        for split in ['train', 'valid', 'test']:
            output_path = os.path.join(output_dir, f'{split}_sequences.pkl')
            with open(output_path, 'wb') as f:
                pickle.dump(sequences[split], f)
            print(f"✓ Saved {split} sequences to {output_path}")

        # 保存物品特征
        features_path = os.path.join(output_dir, 'item_features.pkl')
        with open(features_path, 'wb') as f:
            pickle.dump(item_features, f)
        print(f"✓ Saved item features to {features_path}")

        # 保存映射
        mappings = {
            'user2id': self.user2id,
            'item2id': self.item2id,
            'id2item': self.id2item,
            'num_users': len(self.user2id),
            'num_items': len(self.item2id)
        }
        mappings_path = os.path.join(output_dir, 'mappings.pkl')
        with open(mappings_path, 'wb') as f:
            pickle.dump(mappings, f)
        print(f"✓ Saved mappings to {mappings_path}")

        # 保存统计信息
        stats_path = os.path.join(output_dir, 'statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Saved statistics to {stats_path}")

    def process(self, max_seq_length: int = 50):
        """完整的预处理流程"""
        print("=" * 80)
        print(f"Processing {self.category.upper()} dataset")
        print("=" * 80)

        # 1. 加载原始数据
        reviews, meta_dict = self.load_raw_data()

        # 2. K-core过滤
        filtered_reviews = self.filter_k_core(
            reviews,
            k=max(self.min_user_interactions, self.min_item_interactions)
        )

        # 3. 构建映射
        self.build_mappings(filtered_reviews)

        # 4. 留一法划分
        train_data, valid_data, test_data = self.leave_one_out_split(filtered_reviews)

        # 5. 构建序列数据
        sequences = self.build_sequential_data(
            train_data, valid_data, test_data, max_seq_length
        )

        # 6. 提取物品特征
        item_features = self.extract_item_features(meta_dict)

        # 7. 计算统计信息
        stats = self.get_statistics(sequences)

        # 8. 打印统计信息
        print("\n" + "=" * 80)
        print("Dataset Statistics")
        print("=" * 80)
        for split, split_stats in stats.items():
            print(f"\n{split.upper()}:")
            for key, value in split_stats.items():
                print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")

        # 9. 保存处理后的数据
        self.save_processed_data(sequences, item_features, stats)

        print("\n" + "=" * 80)
        print(f"✓ {self.category.upper()} dataset preprocessing complete!")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Preprocess Amazon datasets')
    parser.add_argument(
        '--category',
        type=str,
        choices=['beauty', 'games', 'sports', 'all'],
        default='all',
        help='Dataset category to process'
    )
    parser.add_argument(
        '--raw_dir',
        type=str,
        default='data/raw',
        help='Directory containing raw data'
    )
    parser.add_argument(
        '--processed_dir',
        type=str,
        default='data/processed',
        help='Directory to save processed data'
    )
    parser.add_argument(
        '--min_interactions',
        type=int,
        default=5,
        help='Minimum number of interactions for users and items'
    )
    parser.add_argument(
        '--max_seq_length',
        type=int,
        default=50,
        help='Maximum sequence length'
    )

    args = parser.parse_args()

    categories = ['beauty', 'games', 'sports'] if args.category == 'all' else [args.category]

    for category in categories:
        preprocessor = AmazonPreprocessor(
            category=category,
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            min_user_interactions=args.min_interactions,
            min_item_interactions=args.min_interactions
        )
        preprocessor.process(max_seq_length=args.max_seq_length)
        print("\n")


if __name__ == '__main__':
    main()
