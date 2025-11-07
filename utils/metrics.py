"""
Evaluation metrics for recommendation systems
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from sklearn.metrics import ndcg_score, roc_auc_score


class RecommendationMetrics:
    """推荐系统评估指标"""

    @staticmethod
    def hit_rate(predictions: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
        """
        Hit Rate @ K (HR@K)
        衡量Top-K推荐中是否包含目标物品

        Args:
            predictions: 预测的物品ID (batch, k)
            targets: 目标物品ID (batch,)
            k: Top-K

        Returns:
            HR@K分数
        """
        hits = 0
        for pred, target in zip(predictions, targets):
            if target.item() in pred[:k].tolist():
                hits += 1

        return hits / len(targets)

    @staticmethod
    def ndcg(predictions: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
        """
        Normalized Discounted Cumulative Gain @ K (NDCG@K)

        Args:
            predictions: 预测分数 (batch, num_items)
            targets: 目标物品ID (batch,)
            k: Top-K

        Returns:
            NDCG@K分数
        """
        batch_size = predictions.size(0)
        ndcg_scores = []

        for i in range(batch_size):
            # 创建相关性标签
            relevance = torch.zeros(predictions.size(1))
            relevance[targets[i]] = 1

            # 获取Top-K预测
            top_k_indices = torch.topk(predictions[i], k=k)[1]

            # 计算NDCG
            pred_relevance = relevance[top_k_indices].cpu().numpy().reshape(1, -1)
            true_relevance = relevance.cpu().numpy().reshape(1, -1)

            try:
                ndcg_score_val = ndcg_score(true_relevance, pred_relevance, k=k)
                ndcg_scores.append(ndcg_score_val)
            except:
                # 如果所有相关性都是0，跳过
                ndcg_scores.append(0.0)

        return np.mean(ndcg_scores)

    @staticmethod
    def mrr(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Mean Reciprocal Rank (MRR)
        目标物品在推荐列表中的平均倒数排名

        Args:
            predictions: 预测的物品ID (batch, num_items)
            targets: 目标物品ID (batch,)

        Returns:
            MRR分数
        """
        reciprocal_ranks = []

        for pred, target in zip(predictions, targets):
            try:
                rank = (pred == target).nonzero(as_tuple=True)[0][0].item() + 1
                reciprocal_ranks.append(1.0 / rank)
            except:
                reciprocal_ranks.append(0.0)

        return np.mean(reciprocal_ranks)

    @staticmethod
    def recall(predictions: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
        """
        Recall @ K
        Top-K推荐中目标物品的召回率

        Args:
            predictions: 预测的物品ID (batch, k)
            targets: 目标物品ID (batch,) 或 (batch, num_targets)
            k: Top-K

        Returns:
            Recall@K分数
        """
        if targets.dim() == 1:
            # 单个目标物品
            return RecommendationMetrics.hit_rate(predictions, targets, k)
        else:
            # 多个目标物品
            recalls = []
            for pred, target_list in zip(predictions, targets):
                target_set = set(target_list.tolist())
                pred_set = set(pred[:k].tolist())
                recall_val = len(target_set & pred_set) / len(target_set) if len(target_set) > 0 else 0.0
                recalls.append(recall_val)

            return np.mean(recalls)

    @staticmethod
    def precision(predictions: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
        """
        Precision @ K
        Top-K推荐中目标物品的准确率

        Args:
            predictions: 预测的物品ID (batch, k)
            targets: 目标物品ID (batch,) 或 (batch, num_targets)
            k: Top-K

        Returns:
            Precision@K分数
        """
        if targets.dim() == 1:
            # 单个目标物品
            return RecommendationMetrics.hit_rate(predictions, targets, k)
        else:
            # 多个目标物品
            precisions = []
            for pred, target_list in zip(predictions, targets):
                target_set = set(target_list.tolist())
                pred_set = set(pred[:k].tolist())
                precision_val = len(target_set & pred_set) / k if k > 0 else 0.0
                precisions.append(precision_val)

            return np.mean(precisions)

    @staticmethod
    def map_score(predictions: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
        """
        Mean Average Precision @ K (MAP@K)

        Args:
            predictions: 预测的物品ID (batch, k)
            targets: 目标物品ID (batch,) 或 (batch, num_targets)
            k: Top-K

        Returns:
            MAP@K分数
        """
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        ap_scores = []
        for pred, target_list in zip(predictions, targets):
            target_set = set(target_list.tolist())
            hits = 0
            precision_sum = 0.0

            for i, item in enumerate(pred[:k].tolist()):
                if item in target_set:
                    hits += 1
                    precision_sum += hits / (i + 1)

            ap = precision_sum / min(len(target_set), k) if len(target_set) > 0 else 0.0
            ap_scores.append(ap)

        return np.mean(ap_scores)

    @staticmethod
    def coverage(predictions: torch.Tensor, num_items: int) -> float:
        """
        Catalog Coverage
        推荐系统覆盖的物品比例

        Args:
            predictions: 预测的物品ID (batch, k)
            num_items: 物品总数

        Returns:
            Coverage分数
        """
        unique_items = torch.unique(predictions).numel()
        return unique_items / num_items

    @staticmethod
    def diversity(item_embeddings: torch.Tensor) -> float:
        """
        推荐多样性
        基于物品嵌入的余弦距离

        Args:
            item_embeddings: 推荐物品的嵌入 (num_recommendations, dim)

        Returns:
            平均余弦距离
        """
        # 归一化
        normalized = torch.nn.functional.normalize(item_embeddings, dim=-1)

        # 计算两两余弦相似度
        similarity_matrix = torch.matmul(normalized, normalized.t())

        # 提取上三角（不包括对角线）
        mask = torch.triu(torch.ones_like(similarity_matrix), diagonal=1).bool()
        similarities = similarity_matrix[mask]

        # 多样性 = 1 - 平均相似度
        diversity_score = 1.0 - similarities.mean().item()

        return diversity_score

    @staticmethod
    def novelty(predictions: torch.Tensor, item_popularity: torch.Tensor) -> float:
        """
        推荐新颖性
        推荐不太流行的物品

        Args:
            predictions: 预测的物品ID (batch, k)
            item_popularity: 每个物品的流行度 (num_items,)

        Returns:
            平均新颖性分数（负对数流行度）
        """
        popularities = item_popularity[predictions]
        novelty_scores = -torch.log(popularities + 1e-8)
        return novelty_scores.mean().item()

    @staticmethod
    def auc(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Area Under ROC Curve (AUC)

        Args:
            predictions: 预测分数 (batch, num_items)
            targets: 二值标签 (batch, num_items)

        Returns:
            AUC分数
        """
        try:
            auc = roc_auc_score(
                targets.cpu().numpy().flatten(),
                predictions.cpu().numpy().flatten()
            )
            return auc
        except:
            return 0.5  # 如果计算失败，返回随机猜测的AUC


class DisentanglementMetrics:
    """解耦表征评估指标"""

    @staticmethod
    def mig(z: torch.Tensor, factors: torch.Tensor) -> float:
        """
        Mutual Information Gap (MIG)
        衡量解耦维度与真实因子之间的互信息

        Args:
            z: 解耦表征 (batch, num_dimensions, dim)
            factors: 真实因子 (batch, num_factors)

        Returns:
            MIG分数
        """
        from sklearn.metrics import mutual_info_score

        batch_size, num_dims, dim = z.size()
        num_factors = factors.size(1)

        # 计算每个解耦维度与每个真实因子的互信息
        mi_matrix = np.zeros((num_dims, num_factors))

        for i in range(num_dims):
            z_dim = z[:, i, :].mean(dim=-1).cpu().numpy()  # 对维度取平均
            for j in range(num_factors):
                factor = factors[:, j].cpu().numpy()
                mi_matrix[i, j] = mutual_info_score(
                    np.digitize(z_dim, bins=10),
                    np.digitize(factor, bins=10)
                )

        # MIG: 每个因子的最大MI与次大MI之差的平均
        sorted_mi = np.sort(mi_matrix, axis=0)
        mig_score = np.mean(sorted_mi[-1, :] - sorted_mi[-2, :])

        return mig_score

    @staticmethod
    def sap_score(z: torch.Tensor, factors: torch.Tensor) -> float:
        """
        SAP Score (Separated Attribute Predictability)
        衡量解耦维度是否能独立预测单个因子

        Args:
            z: 解耦表征 (batch, num_dimensions, dim)
            factors: 真实因子 (batch, num_factors)

        Returns:
            SAP分数
        """
        from sklearn.svm import LinearSVC
        from sklearn.preprocessing import StandardScaler

        batch_size, num_dims, dim = z.size()
        num_factors = factors.size(1)

        # 准备数据
        z_flat = z.mean(dim=-1).cpu().numpy()  # (batch, num_dims)
        factors_np = factors.cpu().numpy()

        scaler = StandardScaler()
        z_scaled = scaler.fit_transform(z_flat)

        sap_scores = []

        for j in range(num_factors):
            # 对每个因子训练分类器
            y = np.digitize(factors_np[:, j], bins=10)

            # 计算每个维度的预测准确率
            dim_scores = []
            for i in range(num_dims):
                X = z_scaled[:, i:i+1]
                try:
                    clf = LinearSVC(random_state=0, max_iter=1000)
                    clf.fit(X, y)
                    score = clf.score(X, y)
                    dim_scores.append(score)
                except:
                    dim_scores.append(0.0)

            # SAP: 最高分数与次高分数之差
            sorted_scores = sorted(dim_scores, reverse=True)
            if len(sorted_scores) >= 2:
                sap_scores.append(sorted_scores[0] - sorted_scores[1])
            else:
                sap_scores.append(0.0)

        return np.mean(sap_scores)


class CausalMetrics:
    """因果推断评估指标"""

    @staticmethod
    def ate_error(estimated_ate: torch.Tensor, true_ate: torch.Tensor) -> float:
        """
        Average Treatment Effect Error
        估计的ATE与真实ATE之间的误差

        Args:
            estimated_ate: 估计的ATE (num_treatments,)
            true_ate: 真实的ATE (num_treatments,)

        Returns:
            MAE
        """
        return torch.abs(estimated_ate - true_ate).mean().item()

    @staticmethod
    def calibration_score(
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
        num_bins: int = 10
    ) -> float:
        """
        校准分数
        衡量不确定性估计的可靠性

        Args:
            predictions: 预测值 (batch,)
            uncertainties: 不确定性估计 (batch,)
            targets: 真实值 (batch,)
            num_bins: 箱数

        Returns:
            Expected Calibration Error (ECE)
        """
        # 将预测按不确定性分组
        sorted_indices = torch.argsort(uncertainties)
        bin_size = len(predictions) // num_bins

        ece = 0.0
        for i in range(num_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < num_bins - 1 else len(predictions)

            bin_indices = sorted_indices[start_idx:end_idx]
            bin_predictions = predictions[bin_indices]
            bin_targets = targets[bin_indices]
            bin_uncertainties = uncertainties[bin_indices]

            # 平均误差
            bin_error = torch.abs(bin_predictions - bin_targets).mean()

            # 平均不确定性
            bin_uncertainty = bin_uncertainties.mean()

            # ECE: 误差与不确定性之差的绝对值
            ece += torch.abs(bin_error - bin_uncertainty).item() * len(bin_indices)

        ece /= len(predictions)
        return ece


def evaluate_all_metrics(
    model,
    dataloader,
    device: str = 'cuda',
    k_list: List[int] = [5, 10, 20]
) -> dict:
    """
    评估所有指标

    Args:
        model: 推荐模型
        dataloader: 数据加载器
        device: 设备
        k_list: Top-K列表

    Returns:
        包含所有指标的字典
    """
    model.eval()
    metrics = RecommendationMetrics()

    all_predictions = []
    all_targets = []
    all_scores = []

    with torch.no_grad():
        for batch in dataloader:
            item_ids = batch['item_ids'].to(device)
            multimodal_features = {
                k: v.to(device) for k, v in batch['multimodal_features'].items()
            }
            targets = batch['target_items'].to(device)
            seq_lengths = batch.get('seq_lengths', None)
            if seq_lengths is not None:
                seq_lengths = seq_lengths.to(device)

            # 预测
            outputs = model(
                item_ids,
                multimodal_features,
                seq_lengths,
                target_items=None,
                return_loss=False
            )

            scores = torch.softmax(outputs['recommendation_logits'], dim=-1)
            _, predictions = torch.topk(scores, k=max(k_list), dim=-1)

            all_predictions.append(predictions)
            all_targets.append(targets)
            all_scores.append(scores)

    # 拼接所有结果
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_scores = torch.cat(all_scores, dim=0)

    # 计算各种指标
    results = {}

    for k in k_list:
        results[f'HR@{k}'] = metrics.hit_rate(all_predictions, all_targets, k)
        results[f'NDCG@{k}'] = metrics.ndcg(all_scores, all_targets, k)
        results[f'Recall@{k}'] = metrics.recall(all_predictions, all_targets, k)
        results[f'Precision@{k}'] = metrics.precision(all_predictions, all_targets, k)
        results[f'MAP@{k}'] = metrics.map_score(all_predictions, all_targets, k)

    results['MRR'] = metrics.mrr(all_predictions, all_targets)

    return results
