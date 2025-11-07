"""
QMCSR: Quantum-Inspired Multi-Modal Causal Sequential Recommendation
完整但简化的实现，避免过拟合

核心创新：
1. Aesthetic-Emotional Disentanglement (2维解耦)
2. Quantum-Inspired Multi-Interest Encoder (复数表示)
3. Disentanglement-driven Causal Inference (维度级因果去偏)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


# ==================== Module 2: Aesthetic-Emotional Disentanglement ====================

class AestheticEmotionalDisentanglement(nn.Module):
    """
    美学-情感解耦模块（简化版，避免过拟合）

    创新点：
    1. Visual → Aesthetic (美学吸引力)
    2. Text → Emotional (情感煽动性)
    3. 正交约束：L_ortho = |cos(h_aes, h_emo)|
    """

    def __init__(
        self,
        text_dim: int = 768,
        image_dim: int = 2048,
        latent_dim: int = 32,  # ⭐ 简化：更小的维度
        dropout: float = 0.2
    ):
        super().__init__()

        self.latent_dim = latent_dim

        # Visual → Aesthetic
        self.visual_encoder = nn.Sequential(
            nn.Linear(image_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, latent_dim)
        )

        # Text → Emotional
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, latent_dim)
        )

        # Intensity learners (学习强度)
        self.aesthetic_intensity = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Sigmoid()
        )

        self.emotional_intensity = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Sigmoid()
        )

        # Fusion weights (自适应融合权重)
        self.fusion_net = nn.Sequential(
            nn.Linear(latent_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1)
        )

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        return_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            text_features: (batch, seq_len, text_dim) or (batch, text_dim)
            image_features: (batch, seq_len, image_dim) or (batch, image_dim)

        Returns:
            解耦特征和损失
        """
        # 处理序列维度
        if text_features.dim() == 3:
            batch_size, seq_len, _ = text_features.size()
            text_features = text_features.view(-1, text_features.size(-1))
            image_features = image_features.view(-1, image_features.size(-1))
            is_sequence = True
        else:
            batch_size = text_features.size(0)
            is_sequence = False

        # Aesthetic (从视觉提取)
        h_aes_base = self.visual_encoder(image_features)
        intensity_aes = self.aesthetic_intensity(h_aes_base)
        h_aes = h_aes_base * intensity_aes

        # Emotional (从文本提取)
        h_emo_base = self.text_encoder(text_features)
        intensity_emo = self.emotional_intensity(h_emo_base)
        h_emo = h_emo_base * intensity_emo

        # Fusion (自适应融合)
        combined = torch.cat([h_aes, h_emo], dim=-1)
        fusion_weights = self.fusion_net(combined)
        h_fused = fusion_weights[:, 0:1] * h_aes + fusion_weights[:, 1:2] * h_emo

        # 恢复序列维度
        if is_sequence:
            h_aes = h_aes.view(batch_size, seq_len, -1)
            h_emo = h_emo.view(batch_size, seq_len, -1)
            h_fused = h_fused.view(batch_size, seq_len, -1)

        # 拼接解耦特征
        h_disentangled = torch.cat([h_aes, h_emo], dim=-1)

        results = {
            'h_disentangled': h_disentangled,
            'h_aes': h_aes,
            'h_emo': h_emo,
            'h_fused': h_fused,
            'fusion_weights': fusion_weights
        }

        # 正交约束损失
        if return_loss:
            # 计算余弦相似度（取绝对值，因为我们希望正交）
            h_aes_norm = F.normalize(h_aes_base, dim=-1)
            h_emo_norm = F.normalize(h_emo_base, dim=-1)
            cosine_sim = (h_aes_norm * h_emo_norm).sum(dim=-1).abs()
            ortho_loss = cosine_sim.mean()

            results['loss'] = ortho_loss
            results['ortho_loss'] = ortho_loss

        return results


# ==================== Module 3: Quantum-Inspired Multi-Interest Encoder ====================

class QuantumMultiInterestEncoder(nn.Module):
    """
    量子启发的多兴趣编码器（简化版）

    核心思想：
    1. 为K个兴趣提取幅度（重要性）和相位（特性）
    2. 构造复数态：ψ_k = A_k * e^(i*φ_k)
    3. 干涉：相似兴趣增强，相反兴趣抵消
    4. 测量：复数态坍缩为实数表示
    """

    def __init__(
        self,
        input_dim: int,
        num_interests: int = 4,  # ⭐ 简化：减少兴趣数量
        output_dim: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_interests = num_interests
        self.output_dim = output_dim

        # 幅度网络 (Amplitude)
        self.amplitude_net = nn.Sequential(
            nn.Linear(input_dim, num_interests),
            nn.Softmax(dim=-1)  # 归一化为概率分布
        )

        # 相位网络 (Phase)
        self.phase_net = nn.Sequential(
            nn.Linear(input_dim, num_interests),
            nn.Tanh()  # [-1, 1] → [-π, π]
        )

        # 兴趣表示网络
        self.interest_projection = nn.Linear(input_dim, num_interests * output_dim)

        # 测量投影
        self.measurement_head = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_all_interests: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            x: (batch, input_dim) or (batch, seq_len, input_dim)

        Returns:
            多兴趣表示和测量结果
        """
        # 处理序列输入
        if x.dim() == 3:
            x = x[:, -1, :]  # 取最后一个时间步

        batch_size = x.size(0)

        # Step 1: 提取幅度和相位
        A = self.amplitude_net(x)  # (batch, K)
        phi = self.phase_net(x) * math.pi  # (batch, K) ∈ [-π, π]

        # Step 2: 生成兴趣表示
        interests = self.interest_projection(x).view(
            batch_size, self.num_interests, self.output_dim
        )  # (batch, K, D)

        # Step 3: 构造复数态（幅度 + 相位）
        # real = A * cos(φ), imag = A * sin(φ)
        real_parts = A.unsqueeze(-1) * torch.cos(phi).unsqueeze(-1) * interests
        imag_parts = A.unsqueeze(-1) * torch.sin(phi).unsqueeze(-1) * interests

        # Step 4: 干涉（叠加）
        # ψ_total = Σ_k (real_k + i*imag_k)
        real_total = real_parts.sum(dim=1)  # (batch, D)
        imag_total = imag_parts.sum(dim=1)  # (batch, D)

        # Step 5: 测量（Born rule）
        # |ψ|² = real² + imag²
        magnitude_sq = real_total ** 2 + imag_total ** 2
        h_measured = torch.sqrt(magnitude_sq + 1e-8)

        # 投影到输出空间
        output = self.measurement_head(h_measured)

        results = {
            'output': output,
            'amplitudes': A,
            'phases': phi,
            'real_parts': real_parts,
            'imag_parts': imag_parts,
            'superposed_state_real': real_total,
            'superposed_state_imag': imag_total,
            'interference_strength': A  # 用于可视化
        }

        if return_all_interests:
            results['individual_interests_real'] = real_parts
            results['individual_interests_imag'] = imag_parts

        return results


# ==================== Module 4: Disentanglement-driven Causal Inference ====================

class CausalDebiasing(nn.Module):
    """
    解耦驱动的因果推断（简化版）

    核心思想：
    1. 在多兴趣表示上，利用解耦维度做因果去偏
    2. 生成维度级反事实
    3. 估计Individual Treatment Effect (ITE)
    4. 从多兴趣表示中去除bias
    """

    def __init__(
        self,
        seq_repr_dim: int,
        disentangled_dim: int = 32,
        num_dimensions: int = 2,  # Aesthetic + Emotional
        dropout: float = 0.2
    ):
        super().__init__()

        self.disentangled_dim = disentangled_dim
        self.num_dimensions = num_dimensions

        # 用户个性化参数网络（基于多兴趣表示）
        self.user_param_nets = nn.ModuleDict({
            'aesthetic': nn.Sequential(
                nn.Linear(seq_repr_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, disentangled_dim)
            ),
            'emotional': nn.Sequential(
                nn.Linear(seq_repr_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, disentangled_dim)
            )
        })

        # 反事实生成网络
        self.cf_nets = nn.ModuleDict({
            'aesthetic': nn.Sequential(
                nn.Linear(disentangled_dim * 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, disentangled_dim)
            ),
            'emotional': nn.Sequential(
                nn.Linear(disentangled_dim * 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, disentangled_dim)
            )
        })

        # ITE估计网络
        self.ite_net = nn.Sequential(
            nn.Linear(disentangled_dim * 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_dimensions)
        )

        # 干预强度（可学习）
        self.lambda_aes = nn.Parameter(torch.tensor(0.5))
        self.lambda_emo = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        h_seq: torch.Tensor,
        h_aes: torch.Tensor,
        h_emo: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        因果去偏前向传播

        Args:
            h_seq: 多兴趣表示 (batch, dim)
            h_aes: 美学特征 (batch, disentangled_dim)
            h_emo: 情感特征 (batch, disentangled_dim)

        Returns:
            去偏后的表示和因果效应
        """
        # Step 1: 提取用户个性化参数
        u_aes = self.user_param_nets['aesthetic'](h_seq)
        u_emo = self.user_param_nets['emotional'](h_seq)

        # Step 2: 生成反事实
        # "如果产品不那么美，用户还会点击吗？"
        delta_aes = self.cf_nets['aesthetic'](torch.cat([h_aes, u_aes], dim=-1))
        h_aes_cf = h_aes - self.lambda_aes * delta_aes

        # "如果描述不那么煽情，用户还会点击吗？"
        delta_emo = self.cf_nets['emotional'](torch.cat([h_emo, u_emo], dim=-1))
        h_emo_cf = h_emo - self.lambda_emo * delta_emo

        # Step 3: 估计ITE
        ite_input = torch.cat([h_aes, h_aes_cf, h_emo, h_emo_cf], dim=-1)
        ite = self.ite_net(ite_input)  # (batch, 2)

        # Step 4: 去偏（从序列表示中去除bias）
        # 简化：直接用ITE作为去偏信号
        # 需要将ITE投影到h_seq的维度
        ite_aes = ite[:, 0:1]  # (batch, 1)
        ite_emo = ite[:, 1:2]  # (batch, 1)

        # 简单线性去偏
        h_debiased = h_seq - 0.1 * (ite_aes + ite_emo)

        return {
            'h_debiased': h_debiased,
            'h_aes_cf': h_aes_cf,
            'h_emo_cf': h_emo_cf,
            'ite': ite,
            'ite_aes': ite_aes,
            'ite_emo': ite_emo
        }


# ==================== Complete QMCSR Recommender ====================

class QMCSRRecommender(nn.Module):
    """
    完整的QMCSR推荐系统（简化版，避免过拟合）

    Pipeline:
    1. Multi-Modal Feature Extraction
    2. Aesthetic-Emotional Disentanglement
    3. Quantum-Inspired Multi-Interest Encoder
    4. Disentanglement-driven Causal Inference
    5. Prediction
    """

    def __init__(
        self,
        # 模态配置
        text_dim: int = 768,
        image_dim: int = 2048,
        item_embed_dim: int = 64,  # ⭐ 简化：更小的embedding
        num_items: int = 10000,
        # 解耦配置
        disentangled_dim: int = 32,  # ⭐ 简化：更小的维度
        # 量子配置
        num_interests: int = 4,  # ⭐ 简化：更少的兴趣
        # 序列配置
        hidden_dim: int = 64,  # ⭐ 简化：更小的隐藏层
        max_seq_length: int = 50,
        # 损失权重
        alpha_ortho: float = 0.1,
        alpha_causal: float = 0.05,
        # 其他
        dropout: float = 0.2
    ):
        super().__init__()

        self.item_embed_dim = item_embed_dim
        self.disentangled_dim = disentangled_dim
        self.num_items = num_items

        # 损失权重
        self.alpha_ortho = alpha_ortho
        self.alpha_causal = alpha_causal

        # Module 1: 物品嵌入
        self.item_embedding = nn.Embedding(
            num_items + 1, item_embed_dim, padding_idx=0
        )

        # Module 2: 美学-情感解耦
        self.disentanglement = AestheticEmotionalDisentanglement(
            text_dim=text_dim,
            image_dim=image_dim,
            latent_dim=disentangled_dim,
            dropout=dropout
        )

        # 序列编码器（简单GRU）
        total_disentangled_dim = disentangled_dim * 2  # Aesthetic + Emotional
        seq_input_dim = total_disentangled_dim + item_embed_dim
        self.sequence_encoder = nn.GRU(
            seq_input_dim,
            hidden_dim,
            num_layers=1,  # ⭐ 简化：单层
            batch_first=True,
            dropout=0
        )

        # Module 3: 量子多兴趣编码器
        self.quantum_encoder = QuantumMultiInterestEncoder(
            input_dim=hidden_dim,
            num_interests=num_interests,
            output_dim=item_embed_dim,
            dropout=dropout
        )

        # Module 4: 因果去偏（可选，根据训练阶段）
        self.causal_debiasing = CausalDebiasing(
            seq_repr_dim=item_embed_dim,
            disentangled_dim=disentangled_dim,
            num_dimensions=2,
            dropout=dropout
        )

        # Module 5: 预测头（简单点积）
        self.item_bias = nn.Parameter(torch.zeros(num_items + 1))

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        if self.item_embedding.padding_idx is not None:
            self.item_embedding.weight.data[self.item_embedding.padding_idx].zero_()

    def forward(
        self,
        item_ids: torch.Tensor,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
        target_items: Optional[torch.Tensor] = None,
        use_causal: bool = False,
        return_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            item_ids: (batch, seq_len)
            text_features: (batch, seq_len, text_dim)
            image_features: (batch, seq_len, image_dim)
            seq_lengths: (batch,)
            target_items: (batch,) 目标物品
            use_causal: 是否使用因果去偏
            return_loss: 是否计算损失
        """
        batch_size, seq_len = item_ids.size()

        # ===== Module 1: 特征提取 =====
        item_embeddings = self.item_embedding(item_ids)  # (batch, seq_len, embed_dim)

        # ===== Module 2: 解耦 =====
        disentangled_output = self.disentanglement(
            text_features, image_features, return_loss=return_loss
        )
        h_disentangled = disentangled_output['h_disentangled']  # (batch, seq_len, 64)
        h_aes = disentangled_output['h_aes']  # (batch, seq_len, 32)
        h_emo = disentangled_output['h_emo']  # (batch, seq_len, 32)

        # 拼接item embedding
        combined_features = torch.cat([h_disentangled, item_embeddings], dim=-1)

        # ===== 序列编码 =====
        if seq_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                combined_features, seq_lengths.cpu(),
                batch_first=True, enforce_sorted=False
            )
            output, hidden = self.sequence_encoder(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True, total_length=seq_len
            )
        else:
            output, hidden = self.sequence_encoder(combined_features)

        # 取最后一个有效时间步
        if seq_lengths is not None:
            last_indices = (seq_lengths - 1).long()
            h_seq = output[torch.arange(batch_size, device=output.device), last_indices]
        else:
            h_seq = output[:, -1, :]  # (batch, hidden_dim)

        # ===== Module 3: 量子多兴趣编码 =====
        quantum_output = self.quantum_encoder(h_seq)
        h_interests = quantum_output['output']  # (batch, item_embed_dim)

        # ===== Module 4: 因果去偏（可选）=====
        if use_causal and self.alpha_causal > 0:
            # 取最后一个时间步的解耦特征
            h_aes_last = h_aes[:, -1, :] if h_aes.dim() == 3 else h_aes
            h_emo_last = h_emo[:, -1, :] if h_emo.dim() == 3 else h_emo

            causal_output = self.causal_debiasing(
                h_interests, h_aes_last, h_emo_last
            )
            h_final = causal_output['h_debiased']
        else:
            h_final = h_interests
            causal_output = None

        # ===== Module 5: 预测 =====
        # 归一化
        h_final_norm = F.normalize(h_final + 1e-8, p=2, dim=-1)
        item_emb_norm = F.normalize(self.item_embedding.weight + 1e-8, p=2, dim=-1)

        # 点积打分
        logits = torch.matmul(h_final_norm, item_emb_norm.T) + self.item_bias
        logits[:, 0] = -1e9  # 屏蔽padding

        results = {
            'logits': logits,
            'h_final': h_final,
            'h_interests': h_interests,
            'disentangled_output': disentangled_output,
            'quantum_output': quantum_output,
            'causal_output': causal_output
        }

        # ===== 损失计算 =====
        if return_loss and target_items is not None:
            # 推荐损失（BPR）
            batch_indices = torch.arange(batch_size, device=target_items.device)
            pos_scores = logits[batch_indices, target_items]

            # 负采样
            num_negatives = 5  # ⭐ 简化：更少的负样本
            scores_for_neg = logits.clone()
            scores_for_neg[batch_indices, target_items] = -1e9
            batch_idx_exp = batch_indices.unsqueeze(1).expand_as(item_ids)
            scores_for_neg[batch_idx_exp, item_ids] = -1e9

            neg_items = torch.topk(scores_for_neg, k=num_negatives, dim=-1).indices
            neg_scores = logits[batch_indices.unsqueeze(1), neg_items]

            # BPR损失
            rec_loss = -F.logsigmoid(pos_scores.unsqueeze(1) - neg_scores).mean()

            # 正交损失
            ortho_loss = disentangled_output.get('ortho_loss', torch.tensor(0.0))

            # 因果损失
            if use_causal and causal_output is not None:
                ite = causal_output['ite']
                # 鼓励ITE有合理的幅度
                causal_loss = F.smooth_l1_loss(
                    ite.abs().mean(),
                    torch.tensor(0.3, device=ite.device)
                )
            else:
                causal_loss = torch.tensor(0.0, device=logits.device)

            # 总损失
            total_loss = rec_loss + self.alpha_ortho * ortho_loss + self.alpha_causal * causal_loss

            results.update({
                'loss': total_loss,
                'rec_loss': rec_loss,
                'ortho_loss': ortho_loss,
                'causal_loss': causal_loss
            })

        return results

    def predict(
        self,
        item_ids: torch.Tensor,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
        top_k: int = 10,
        use_causal: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """预测Top-K推荐"""
        with torch.no_grad():
            outputs = self.forward(
                item_ids, text_features, image_features,
                seq_lengths, target_items=None,
                use_causal=use_causal, return_loss=False
            )

            logits = outputs['logits']
            top_k_scores, top_k_items = torch.topk(logits, k=top_k, dim=-1)

            return top_k_items, top_k_scores
