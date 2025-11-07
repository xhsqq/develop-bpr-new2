"""
Improved Quantum-Inspired Multi-Interest Encoder
改进的量子启发多兴趣编码器：严格的相位、幺正性和量子测量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math
import numpy as np


class SimplifiedAttention(nn.Module):
    """简化的单头注意力（适合小数据集）"""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** -0.5
        
        # 简单的QKV投影（实数）
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        简单注意力
        Args:
            x: (batch, num_interests, dim)
        Returns:
            output: (batch, num_interests, dim)
        """
        B, N, D = x.shape

        # QKV投影
        qkv = self.qkv(x).reshape(B, N, 3, D).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意力计算
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # 加权求和
        out = torch.matmul(attn, v)
        out = self.out(out)

        return out


class SimplifiedPooling(nn.Module):
    """简化的池化层（替代复杂的量子测量）"""

    def __init__(self, num_interests: int, dim: int, output_dim: int):
        super().__init__()
        
        # 简单的可学习权重
        self.attention_weights = nn.Parameter(torch.randn(num_interests) / math.sqrt(num_interests))
        self.projection = nn.Linear(dim, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, num_interests, dim)
        Returns:
            output: (batch, output_dim)
            weights: (batch, num_interests)
        """
        # 计算注意力权重
        weights = F.softmax(self.attention_weights, dim=0)
        weights = weights.unsqueeze(0).expand(x.size(0), -1)  # (batch, num_interests)
        
        # 加权池化
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)  # (batch, dim)
        
        # 投影到输出维度
        output = self.projection(pooled)
        
        return output, weights


class ImprovedQuantumEncoder(nn.Module):
    """
    极简量子编码器（适合小数据集，但保留核心量子特性）
    
    核心量子特性（保留）：
    1. ✅ 量子叠加态（相位编码）
    2. ✅ 干涉效应（可学习混合）
    3. ✅ 概率测量（Born rule）
    
    大幅简化：
    1. 移除复数运算（用相位权重代替）
    2. 简化干涉（直接可学习权重）
    3. 降低qubit_dim（128→64）
    4. 减少兴趣数（8→4，更适合小数据集）
    """

    def __init__(
        self,
        input_dim: int = 192,
        num_interests: int = 4,  # ⭐⭐ 进一步减少：8→4
        qubit_dim: int = 64,     # ⭐⭐ 降低维度：128→64
        output_dim: int = 256,
        hidden_dim: int = 512,
        use_quantum_computing: bool = False
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_interests = num_interests
        self.qubit_dim = qubit_dim
        self.output_dim = output_dim

        # ===1. 兴趣编码器（简化，无复数）===
        self.interest_encoder = nn.Sequential(
            nn.Linear(input_dim, num_interests * qubit_dim),
            nn.LayerNorm(num_interests * qubit_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # ===2. 量子相位编码（保留量子特性）===
        self.phase_encoder = nn.Sequential(
            nn.Linear(input_dim, num_interests),
            nn.Tanh()  # [-1, 1] → 相位权重
        )

        # ===3. 可学习的兴趣混合权重（替代幺正干涉）===
        self.interest_mixing = nn.Parameter(
            torch.eye(num_interests) * 0.8 + torch.ones(num_interests, num_interests) * 0.05
        )

        # ===4. 简化的注意力===
        self.attention = SimplifiedAttention(dim=qubit_dim)

        # ===5. 测量投影===
        self.measurement_projection = nn.Linear(qubit_dim, output_dim)
        
        # ===初始化：让兴趣更分散===
        self._init_orthogonal_interests()
    
    def _init_orthogonal_interests(self):
        """
        正交初始化兴趣编码器，让不同兴趣一开始就更分散
        """
        with torch.no_grad():
            # 对interest_encoder的权重进行正交初始化
            weight = self.interest_encoder[0].weight
            if weight.size(0) >= weight.size(1):
                # 输出维度 >= 输入维度，可以做正交初始化
                nn.init.orthogonal_(weight)
            
            # 对interest_mixing进行更强的对角化（减少不同兴趣间的混合）
            self.interest_mixing.data = torch.eye(self.num_interests) * 0.9 + \
                                        torch.randn(self.num_interests, self.num_interests) * 0.02

    def forward(
        self,
        x: torch.Tensor,
        return_all_interests: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        极简量子前向传播（保留核心概念但大幅简化）
        
        Args:
            x: (batch, input_dim) or (batch, seq_len, input_dim)
            return_all_interests: 是否返回所有兴趣

        Returns:
            包含多兴趣表示和测量结果的字典
        """
        batch_size = x.size(0)

        # 如果是序列输入，先聚合
        if x.dim() == 3:
            x = x.mean(dim=1)

        # ===Step 1: 编码为多兴趣表示===
        interests = self.interest_encoder(x).view(
            batch_size, self.num_interests, self.qubit_dim
        )  # (batch, num_interests, qubit_dim)
        
        # ===Step 2: 量子相位调制（核心量子特性）===
        phases = self.phase_encoder(x)  # (batch, num_interests) ∈ [-1, 1]
        
        # 应用相位权重（模拟量子相位）
        phase_weights = torch.sigmoid(phases * np.pi)  # (batch, num_interests) ∈ [0, 1]
        interests_modulated = interests * phase_weights.unsqueeze(-1)  # (batch, num_interests, qubit_dim)
        
        # 归一化
        interests_modulated = F.normalize(interests_modulated, dim=-1)

        # ===Step 3: 兴趣混合（简化的干涉效应）===
        # 应用可学习的混合矩阵
        mixing_weights = F.softmax(self.interest_mixing, dim=-1)  # (num_interests, num_interests)
        
        # 混合兴趣: interests' = mixing_weights @ interests
        interests_mixed = torch.matmul(
            mixing_weights.unsqueeze(0),  # (1, num_interests, num_interests)
            interests_modulated  # (batch, num_interests, qubit_dim)
        )  # (batch, num_interests, qubit_dim)

        # ===Step 4: 兴趣间交互（注意力）===
        interests_attended = self.attention(interests_mixed)  # (batch, num_interests, qubit_dim)

        # ===Step 5: 量子测量（Born rule概率坍缩）===
        # 计算测量概率（基于范数）
        interest_norms = torch.norm(interests_attended, dim=-1)  # (batch, num_interests)
        measurement_probs = F.softmax(interest_norms, dim=-1)  # (batch, num_interests)
        
        # 概率性坍缩（加权求和）
        collapsed_state = (interests_attended * measurement_probs.unsqueeze(-1)).sum(dim=1)  # (batch, qubit_dim)
        
        # 投影到输出空间
        output = self.measurement_projection(collapsed_state)  # (batch, output_dim)

        # ===Step 6: 简化的量子度量===
        diversity = self._compute_diversity(interests_attended)
        phase_variance = phases.var(dim=1).mean()

        results = {
            'output': output,  # (batch, output_dim)
            'measurement_probabilities': measurement_probs,
            'superposed_state_real': collapsed_state,
            'superposed_state_imag': torch.zeros_like(collapsed_state),  # 兼容性
            'interference_strength': measurement_probs,
            'metrics': {
                'diversity': diversity,
                'phase_variance': phase_variance
            }
        }

        if return_all_interests:
            results['individual_interests_real'] = interests_attended
            results['individual_interests_imag'] = torch.zeros_like(interests_attended)  # 兼容性

        return results
    
    def _compute_diversity(self, interests: torch.Tensor) -> torch.Tensor:
        """
        计算兴趣多样性（改进版）
        
        目标：鼓励不同兴趣向量尽可能正交
        """
        # 归一化
        interests_norm = F.normalize(interests, dim=-1)  # (batch, num_interests, dim)
        
        # 计算两两相似度矩阵
        similarity = torch.bmm(
            interests_norm, 
            interests_norm.transpose(1, 2)
        )  # (batch, num_interests, num_interests)
        
        # 只看非对角线元素（不同兴趣间的相似度）
        # 目标：让相似度尽可能小（接近0），即让兴趣正交
        mask = 1 - torch.eye(self.num_interests, device=interests.device)
        mask = mask.unsqueeze(0).expand(similarity.size(0), -1, -1)
        
        # 非对角线相似度的平方和（惩罚任何非零相似度）
        off_diag_sim = (similarity * mask) ** 2
        similarity_penalty = off_diag_sim.sum(dim=[1, 2]) / (mask.sum(dim=[1, 2]) + 1e-8)
        
        # diversity损失 = 相似度惩罚的平均
        # 注意：这是一个"损失"，越小越好（表示兴趣越正交）
        diversity_loss = similarity_penalty.mean()
        
        return diversity_loss



def compute_quantum_losses(quantum_output: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    极简量子损失函数（只保留多样性项）
    """
    metrics = quantum_output['metrics']

    # 兴趣多样性（鼓励不同兴趣相互独立）
    diversity = metrics['diversity']
    diversity_loss = F.relu(0.5 - diversity)  # 鼓励diversity > 0.5

    # 仅返回多样性损失
    return diversity_loss


# 保持向后兼容性的别名
QuantumInspiredMultiInterestEncoder = ImprovedQuantumEncoder
