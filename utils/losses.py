"""
Custom loss functions for multimodal recommendation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking Loss
    用于隐式反馈的推荐
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pos_scores: 正样本得分 (batch,)
            neg_scores: 负样本得分 (batch, num_negatives)

        Returns:
            BPR损失
        """
        # pos_scores: (batch,) -> (batch, 1)
        pos_scores = pos_scores.unsqueeze(-1)

        # BPR loss: -log(sigmoid(pos - neg))
        diff = pos_scores - neg_scores
        loss = -F.logsigmoid(diff).mean()

        return loss


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss (对比学习损失)
    用于学习更好的表示
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query: torch.Tensor,
        positive: torch.Tensor,
        negatives: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: 查询向量 (batch, dim)
            positive: 正样本向量 (batch, dim)
            negatives: 负样本向量 (batch, num_negatives, dim) 或 (num_negatives, dim)
            labels: 标签（如果提供，使用标准对比学习）

        Returns:
            InfoNCE损失
        """
        # 归一化
        query = F.normalize(query, dim=-1)
        positive = F.normalize(positive, dim=-1)

        # 计算正样本相似度
        pos_sim = torch.sum(query * positive, dim=-1) / self.temperature  # (batch,)

        if negatives is not None:
            # 归一化负样本
            negatives = F.normalize(negatives, dim=-1)

            if negatives.dim() == 2:
                # (num_negatives, dim) -> (batch, num_negatives)
                neg_sim = torch.matmul(query, negatives.t()) / self.temperature
            else:
                # (batch, num_negatives, dim)
                neg_sim = torch.sum(
                    query.unsqueeze(1) * negatives, dim=-1
                ) / self.temperature

            # InfoNCE loss
            logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # (batch, 1 + num_neg)
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits, labels)
        else:
            # 自监督对比学习
            # 使用batch内的其他样本作为负样本
            logits = torch.matmul(query, positive.t()) / self.temperature  # (batch, batch)
            labels = torch.arange(logits.size(0), device=logits.device)
            loss = F.cross_entropy(logits, labels)

        return loss


class TripletLoss(nn.Module):
    """
    Triplet Loss
    确保正样本比负样本更接近anchor
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            anchor: 锚点 (batch, dim)
            positive: 正样本 (batch, dim)
            negative: 负样本 (batch, dim) 或 (batch, num_negatives, dim)

        Returns:
            Triplet损失
        """
        pos_dist = F.pairwise_distance(anchor, positive, p=2)

        if negative.dim() == 2:
            neg_dist = F.pairwise_distance(anchor, negative, p=2)
        else:
            # (batch, num_negatives, dim)
            anchor_expanded = anchor.unsqueeze(1).expand_as(negative)
            neg_dist = F.pairwise_distance(
                anchor_expanded.reshape(-1, anchor.size(-1)),
                negative.reshape(-1, negative.size(-1)),
                p=2
            ).view(negative.size(0), negative.size(1))
            neg_dist = neg_dist.min(dim=-1)[0]  # 使用最近的负样本

        loss = F.relu(pos_dist - neg_dist + self.margin).mean()
        return loss


class ListwiseLoss(nn.Module):
    """
    Listwise Ranking Loss
    用于列表级别的排序学习
    """

    def __init__(self, loss_type: str = 'listnet'):
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            scores: 预测分数 (batch, num_items)
            labels: 真实相关性 (batch, num_items)

        Returns:
            Listwise损失
        """
        if self.loss_type == 'listnet':
            # ListNet: 使用Top-1概率分布的交叉熵
            pred_probs = F.softmax(scores, dim=-1)
            true_probs = F.softmax(labels, dim=-1)

            loss = -torch.sum(true_probs * torch.log(pred_probs + 1e-8), dim=-1).mean()

        elif self.loss_type == 'listmle':
            # ListMLE: 最大化似然估计
            # 使用Plackett-Luce模型
            sorted_labels, indices = torch.sort(labels, descending=True, dim=-1)
            sorted_scores = torch.gather(scores, -1, indices)

            # 计算log likelihood
            log_likelihood = 0
            for i in range(sorted_scores.size(-1)):
                log_sum_exp = torch.logsumexp(sorted_scores[:, i:], dim=-1)
                log_likelihood += sorted_scores[:, i] - log_sum_exp

            loss = -log_likelihood.mean()

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss


class DiversityLoss(nn.Module):
    """
    Diversity Loss
    鼓励推荐结果的多样性
    """

    def __init__(self, diversity_type: str = 'determinantal'):
        super().__init__()
        self.diversity_type = diversity_type

    def forward(self, item_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            item_embeddings: 推荐物品的嵌入 (batch, num_items, dim)

        Returns:
            多样性损失（负数表示奖励多样性）
        """
        if self.diversity_type == 'determinantal':
            # 使用行列式点过程 (DPP) 鼓励多样性
            # 计算相似度矩阵
            normalized_embeddings = F.normalize(item_embeddings, dim=-1)
            similarity = torch.matmul(
                normalized_embeddings,
                normalized_embeddings.transpose(-2, -1)
            )  # (batch, num_items, num_items)

            # 多样性 = -相似度
            # 最小化相似度 = 最大化多样性
            diversity = -similarity.mean()

        elif self.diversity_type == 'coverage':
            # Coverage-based diversity
            # 最大化嵌入空间的覆盖范围
            mean_embedding = item_embeddings.mean(dim=1, keepdim=True)
            variance = ((item_embeddings - mean_embedding) ** 2).sum(dim=-1).mean()
            diversity = -variance  # 负数表示奖励高方差

        else:
            raise ValueError(f"Unknown diversity type: {self.diversity_type}")

        return diversity


class CausalRegularizationLoss(nn.Module):
    """
    Causal Regularization Loss
    正则化因果效应估计
    """

    def __init__(self, reg_type: str = 'treatment_balance'):
        super().__init__()
        self.reg_type = reg_type

    def forward(
        self,
        propensity_scores: torch.Tensor,
        treatment_effects: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            propensity_scores: 倾向得分 (batch, num_treatments)
            treatment_effects: 处理效应 (batch, num_treatments)

        Returns:
            因果正则化损失
        """
        if self.reg_type == 'treatment_balance':
            # 鼓励处理分配的平衡
            # 倾向得分应该接近均匀分布
            uniform_dist = torch.ones_like(propensity_scores) / propensity_scores.size(-1)
            balance_loss = F.kl_div(
                propensity_scores.log(),
                uniform_dist,
                reduction='batchmean'
            )
            return balance_loss

        elif self.reg_type == 'effect_smoothness':
            # 鼓励处理效应的平滑性
            # 相邻处理的效应不应该差异太大
            diff = treatment_effects[:, 1:] - treatment_effects[:, :-1]
            smoothness_loss = (diff ** 2).mean()
            return smoothness_loss

        else:
            raise ValueError(f"Unknown regularization type: {self.reg_type}")
