"""
Full-library evaluation without negative sampling
全库评估（无负采样）
"""

import torch
import numpy as np
from typing import Dict, List
from tqdm import tqdm


class FullLibraryEvaluator:
    """
    全库评估器
    对所有物品计算分数，不使用负采样
    """

    def __init__(self, num_items: int, k_list: List[int] = [5, 10, 20, 50]):
        """
        Args:
            num_items: 物品总数
            k_list: Top-K列表
        """
        self.num_items = num_items
        self.k_list = sorted(k_list)

    @torch.no_grad()
    def evaluate(
        self,
        model,
        dataloader,
        device: str = 'cuda'
    ) -> Dict[str, float]:
        """
        全库评估

        Args:
            model: 推荐模型
            dataloader: 数据加载器
            device: 设备

        Returns:
            包含各种指标的字典
        """
        model.eval()

        all_hr = {k: [] for k in self.k_list}
        all_ndcg = {k: [] for k in self.k_list}
        all_mrr = []

        for batch in tqdm(dataloader, desc="Evaluating"):
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
                target_items=None,
                return_loss=False,
                return_explanations=False
            )

            # 获取所有物品的预测分数 (batch, num_items)
            logits = outputs['recommendation_logits']
            # ⭐ 直接使用logits排序，不要用softmax（会导致分数过于平滑）
            # softmax在有大量物品时会压缩分数差异，影响排序效果
            scores = logits  # 直接使用原始logits

            # 对每个样本计算指标
            for i in range(scores.size(0)):
                score = scores[i]  # (num_items,)
                target = target_items[i].item()

                # 排序得到Top-K
                _, top_k_indices = torch.topk(score, k=max(self.k_list), dim=-1)
                top_k_items = top_k_indices.cpu().numpy()

                # 计算HR@K和NDCG@K
                for k in self.k_list:
                    top_k = top_k_items[:k]

                    # Hit Rate
                    hit = 1.0 if target in top_k else 0.0
                    all_hr[k].append(hit)

                    # NDCG
                    if target in top_k:
                        rank = np.where(top_k == target)[0][0] + 1
                        ndcg = 1.0 / np.log2(rank + 1)
                    else:
                        ndcg = 0.0
                    all_ndcg[k].append(ndcg)

                # MRR
                if target in top_k_items:
                    rank = np.where(top_k_items == target)[0][0] + 1
                    mrr = 1.0 / rank
                else:
                    mrr = 0.0
                all_mrr.append(mrr)

        # 计算平均指标
        metrics = {}

        for k in self.k_list:
            metrics[f'HR@{k}'] = np.mean(all_hr[k])
            metrics[f'NDCG@{k}'] = np.mean(all_ndcg[k])

        metrics['MRR'] = np.mean(all_mrr)

        return metrics

    @torch.no_grad()
    def evaluate_with_filter(
        self,
        model,
        dataloader,
        train_items_per_user: Dict[int, set],
        device: str = 'cuda'
    ) -> Dict[str, float]:
        """
        全库评估（过滤训练集物品）
        不考虑用户在训练集中已经交互过的物品

        Args:
            model: 推荐模型
            dataloader: 数据加载器
            train_items_per_user: 每个用户在训练集中交互过的物品集合
            device: 设备

        Returns:
            包含各种指标的字典
        """
        model.eval()

        all_hr = {k: [] for k in self.k_list}
        all_ndcg = {k: [] for k in self.k_list}
        all_mrr = []

        for batch in tqdm(dataloader, desc="Evaluating (filtered)"):
            # 移动到设备
            item_ids = batch['item_ids'].to(device)
            target_items = batch['target_items'].to(device)
            seq_lengths = batch['seq_lengths'].to(device)
            user_ids = batch.get('user_ids', None)

            multimodal_features = {
                k: v.to(device) for k, v in batch['multimodal_features'].items()
            }

            # 前向传播
            outputs = model(
                item_ids=item_ids,
                multimodal_features=multimodal_features,
                seq_lengths=seq_lengths,
                target_items=None,
                return_loss=False,
                return_explanations=False
            )

            # 获取所有物品的预测分数 (batch, num_items)
            logits = outputs['recommendation_logits']
            # ⭐ 直接使用logits排序，不要用softmax（会导致分数过于平滑）
            scores = logits  # 直接使用原始logits

            # 对每个样本计算指标
            for i in range(scores.size(0)):
                score = scores[i].cpu().numpy()  # (num_items,)
                target = target_items[i].item()

                # 过滤训练集物品
                if user_ids is not None:
                    user_id = user_ids[i].item()
                    if user_id in train_items_per_user:
                        train_items = train_items_per_user[user_id]
                        # 将训练集物品的分数设为-inf
                        for item in train_items:
                            if item < len(score):
                                score[item] = -np.inf

                # 排序得到Top-K
                top_k_indices = np.argsort(score)[::-1][:max(self.k_list)]

                # 计算HR@K和NDCG@K
                for k in self.k_list:
                    top_k = top_k_indices[:k]

                    # Hit Rate
                    hit = 1.0 if target in top_k else 0.0
                    all_hr[k].append(hit)

                    # NDCG
                    if target in top_k:
                        rank = np.where(top_k == target)[0][0] + 1
                        ndcg = 1.0 / np.log2(rank + 1)
                    else:
                        ndcg = 0.0
                    all_ndcg[k].append(ndcg)

                # MRR
                if target in top_k_indices:
                    rank = np.where(top_k_indices == target)[0][0] + 1
                    mrr = 1.0 / rank
                else:
                    mrr = 0.0
                all_mrr.append(mrr)

        # 计算平均指标
        metrics = {}

        for k in self.k_list:
            metrics[f'HR@{k}'] = np.mean(all_hr[k])
            metrics[f'NDCG@{k}'] = np.mean(all_ndcg[k])

        metrics['MRR'] = np.mean(all_mrr)

        return metrics


def get_train_items_per_user(train_dataset) -> Dict[int, set]:
    """
    获取每个用户在训练集中交互过的物品

    Args:
        train_dataset: 训练数据集

    Returns:
        {user_id: set of item_ids}
    """
    train_items = {}

    for sequence in train_dataset.sequences:
        user_id = sequence['user_id']
        history = sequence['history']

        if user_id not in train_items:
            train_items[user_id] = set()

        train_items[user_id].update(history)

    return train_items
