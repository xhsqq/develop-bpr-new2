"""
Multimodal Sequential Recommender
整合解耦表征学习、因果推断和量子启发编码器的完整推荐系统
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List

from .disentangled_representation import DisentangledRepresentation
from .causal_inference import CausalInferenceModule
from .quantum_inspired_encoder import QuantumInspiredMultiInterestEncoder
from utils.losses import BPRLoss


class SequenceEncoder(nn.Module):
    """序列编码器，用于处理用户历史行为序列"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        bidirectional: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # 使用GRU或Transformer编码序列
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.1 if num_layers > 1 else 0
        )

        output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_projection = nn.Linear(output_dim, input_dim)

    def forward(self, sequence: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            sequence: (batch, seq_len, input_dim)
            lengths: (batch,) 每个序列的实际长度

        Returns:
            编码后的序列 (batch, seq_len, input_dim)
        """
        batch_size, seq_len, _ = sequence.size()
        
        if lengths is not None:
            # Pack padded sequence
            packed = nn.utils.rnn.pack_padded_sequence(
                sequence, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output, _ = self.gru(packed)
            # ⭐ 修复：指定total_length以确保输出维度正确
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True, total_length=seq_len
            )
        else:
            output, _ = self.gru(sequence)

        output = self.output_projection(output)
        return output


class MultimodalRecommender(nn.Module):
    """
    多模态序列推荐系统
    整合三个核心组件：
    1. 解耦表征学习 (Disentangled Representation)
    2. 因果推断模块 (Causal Inference)
    3. 量子启发多兴趣编码器 (Quantum-Inspired Multi-Interest Encoder)
    """

    def __init__(
        self,
        # 多模态输入配置
        modality_dims: Dict[str, int] = None,
        modality_proj_dims: Dict[str, int] = None,  # ⭐ 新增：投影层维度配置
        # 解耦表征配置
        disentangled_dim: int = 128,
        num_disentangled_dims: int = 3,
        # 量子编码器配置
        num_interests: int = 4,
        quantum_state_dim: int = 256,
        # 通用配置
        hidden_dim: int = 512,
        item_embed_dim: int = 256,
        num_items: int = 10000,
        # 序列配置
        max_seq_length: int = 50,
        # 损失权重 (v0.8.1 - 平衡方案：rec主导但辅助损失仍有意义)
        alpha_recon: float = 0.1,       # 从0.5→0.1 (降低5倍)
        alpha_causal: float = 0.2,      # 从0.5→0.2 (降低2.5倍)
        alpha_diversity: float = 0.05,  # 保持不变
        # 已移除：alpha_orthogonality（简化后不再使用）
        # 因果损失内部权重配置
        causal_loss_weights: Dict[str, float] = None,  # ⭐ 因果损失内部权重
        # 负采样配置
        num_negatives: int = 10,        # ⭐ 负采样数量
        # 额外配置（从外部config接线）
        beta: float = 0.0,              # ⭐ 极简版：禁用KL损失（小数据集建议）
        gamma: float = 0.0,             # ⭐ 极简版：禁用独立性损失（简化训练）
        use_kl_loss: bool = False,      # ⭐ 是否使用KL损失
        use_independence_loss: bool = False,  # ⭐ 是否使用独立性损失
        quantum_version: str = "simplified",  # ⭐ 量子编码器版本："simplified" or "enhanced"
        temperature: float = 0.5,       # ⭐ 推荐打分温度（buffer）
        num_mc_samples: int = 10,       # ⭐ 不确定性量化MC采样数
        num_ensembles: int = 5,         # ⭐ 因果模块集成模型数量（从config接线）
        target_ite: float = 0.4,        # ⭐ 目标ITE幅度（从config接线）
        # 其他
        use_quantum_computing: bool = False,
        dropout: float = 0.1
    ):
        """
        Args:
            modality_dims: 各模态的输入维度，例如 {'text': 768, 'image': 2048}
            disentangled_dim: 每个解耦维度的大小
            num_disentangled_dims: 解耦维度数量（功能、美学、情感）
            num_interests: 用户兴趣数量
            quantum_state_dim: 量子态维度
            hidden_dim: 隐藏层维度
            item_embed_dim: 物品嵌入维度
            num_items: 物品总数
            max_seq_length: 最大序列长度
            alpha_recon: 重构损失权重
            alpha_causal: 因果损失权重
            alpha_diversity: 多样性损失权重
            use_quantum_computing: 是否使用真实量子计算
            dropout: Dropout比率
        """
        super().__init__()

        # 默认模态配置
        if modality_dims is None:
            modality_dims = {
                'text': 768,  # BERT/RoBERTa
                'image': 2048,  # ResNet
                'audio': 128   # Audio features
            }
        
        # ⭐⭐⭐ P1: 添加投影降维层，大幅减少参数量
        # 支持从config读取投影维度，如果没有则使用默认值
        if modality_proj_dims is None:
            # 默认配置：BERT(768) → 64,  ResNet(2048) → 128
            self.projection_dims = {
                'text': 64,     # 768 → 64 (减少91.7%，更激进！)
                'image': 128    # 2048 → 128 (减少93.8%，更激进！)
            }
        else:
            self.projection_dims = modality_proj_dims
        
        self.use_projection = len(self.projection_dims) > 0  # 有配置则启用投影
        
        # 创建投影层
        self.modality_projections = nn.ModuleDict()
        for modality, orig_dim in modality_dims.items():
            if modality in self.projection_dims:
                proj_dim = self.projection_dims[modality]
                self.modality_projections[modality] = nn.Sequential(
                    nn.Linear(orig_dim, proj_dim),
                    nn.LayerNorm(proj_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout * 0.5)  # 轻度dropout防止过拟合
                )
                # 更新模态维度为投影后的维度
                modality_dims[modality] = proj_dim
        
        # ⭐⭐⭐ P0修复：移除item作为模态的配置
        # Item embedding将独立处理，不通过解耦模块

        self.modality_dims = modality_dims
        self.disentangled_dim = disentangled_dim
        self.num_disentangled_dims = num_disentangled_dims
        self.num_interests = num_interests
        self.quantum_state_dim = quantum_state_dim
        self.hidden_dim = hidden_dim
        self.item_embed_dim = item_embed_dim
        self.num_items = num_items
        self.max_seq_length = max_seq_length

        self.alpha_recon = alpha_recon
        self.alpha_causal = alpha_causal
        self.alpha_diversity = alpha_diversity
        # 已移除：正交性损失权重（不再使用）
        # 运行时因果损失权重（由训练脚本调度），默认全量
        self.alpha_causal_runtime = 1.0
        self.num_negatives = num_negatives  # ⭐ 保存负采样数量
        
        # ⭐ 保存解耦模块的内部权重
        self.beta = beta
        self.gamma = gamma

        # ==================== 核心组件 ====================

        # 1. 物品嵌入（+1 因为ID从1开始，0用于padding）
        self.item_embedding = nn.Embedding(num_items + 1, item_embed_dim, padding_idx=0)

        # 2. 解耦表征学习模块（⭐⭐⭐ P0修复：只处理text和image，不包含item）
        self.disentangled_module = DisentangledRepresentation(
            input_dims=modality_dims,  # ⭐ 只包含text和image
            latent_dim=disentangled_dim,  # ⭐ 改进：使用latent_dim参数
            beta=beta,
            gamma=gamma,
            use_kl_loss=use_kl_loss,  # ⭐ 极简版：默认False
            use_independence_loss=use_independence_loss  # ⭐ 极简版：默认False
        )

        # 3. 序列编码器（⭐⭐⭐ P0修复：输入维度 = 解耦维度 + item embedding维度）
        total_disentangled_dim = disentangled_dim * num_disentangled_dims  # 192
        total_input_dim = total_disentangled_dim + item_embed_dim  # 192 + 128 = 320
        self.sequence_encoder = SequenceEncoder(
            input_dim=total_input_dim,  # ⭐ 更新输入维度
            hidden_dim=hidden_dim,
            num_layers=2,
            bidirectional=True
        )

        # 4. 量子启发多兴趣编码器（⭐⭐⭐ 支持版本选择）
        if quantum_version == "enhanced":
            # 增强版：完整量子特性（复数、幺正、干涉、纠缠）- 适合投稿量子期刊
            from models.quantum_encoder_enhanced import QuantumInspiredMultiInterestEncoder as QuantumEnhanced
            self.quantum_encoder = QuantumEnhanced(
                input_dim=total_input_dim,  # ⭐ 320维度
                num_interests=num_interests,
                qubit_dim=quantum_state_dim // 2,
                output_dim=item_embed_dim,
                hidden_dim=hidden_dim,
                use_quantum_computing=use_quantum_computing
            )
        else:
            # 简化版：快速训练（默认）- 适合快速实验
            self.quantum_encoder = QuantumInspiredMultiInterestEncoder(
                input_dim=total_input_dim,  # ⭐ 320维度
                num_interests=num_interests,
                qubit_dim=quantum_state_dim // 2,
                output_dim=item_embed_dim,
                hidden_dim=hidden_dim,
                use_quantum_computing=use_quantum_computing
            )

        # 5. 因果推断模块（⭐⭐⭐ P0修复：传递正确的特征维度）
        self.causal_module = CausalInferenceModule(
            disentangled_dim=disentangled_dim,
            num_dimensions=num_disentangled_dims,
            hidden_dim=hidden_dim,
            num_ensembles=num_ensembles,
            causal_loss_weights=causal_loss_weights,  # ⭐ 传递因果损失权重
            feature_dim=total_input_dim  # ⭐ P0修复：传递完整特征维度（320）
        )

        # ==================== 预测头 ====================

        # ⭐⭐⭐ P0核心修复: 权重共享点积打分（不再用独立分类器）
        # 推荐打分 = user_repr @ item_embedding.weight.T + item_bias
        # 这样rec监督直接作用到item_embedding，加速收敛、提升效果！
        self.item_bias = nn.Parameter(torch.zeros(num_items + 1))  # 每个物品的偏置

        # ⭐ 损失函数
        self.bpr_loss_fn = BPRLoss()
        
        # ⭐⭐⭐ P1修复：温度参数从config读取，不再硬编码
        # temperature=0.2太小，会导致logits范围过小，难以区分正负样本
        # 使用0.3-0.5之间的值更合理
        self.register_buffer('temperature', torch.tensor(temperature))

        # ⭐ 不确定性量化采样数（用于SCM模块）
        self.num_mc_samples = num_mc_samples
        # ⭐ 目标ITE（用于因果损失）
        self.target_ite = target_ite
        
        # ⭐ KL退火因子（训练时动态调整）
        self.kl_anneal_factor = 1.0  # 默认1.0，在训练初期会设为0

        # 维度重要性预测（用于可解释性）（⭐⭐⭐ P0修复：输入维度更新）
        self.dimension_importance_head = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim // 2),  # ⭐ 更新输入维度
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_disentangled_dims)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform初始化
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # 正态分布初始化embedding
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    # padding位置初始化为0
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                # LayerNorm初始化
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # 已移除：正交性损失函数（简化方案不再需要）

    def forward(
        self,
        item_ids: torch.Tensor,
        multimodal_features: Dict[str, torch.Tensor],
        seq_lengths: Optional[torch.Tensor] = None,
        target_items: Optional[torch.Tensor] = None,
        candidate_items: Optional[torch.Tensor] = None,  # ⭐ 新增：候选物品（负采样）
        labels: Optional[torch.Tensor] = None,  # ⭐ 新增：标签（0/1）
        return_loss: bool = True,
        return_explanations: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            item_ids: 物品ID序列 (batch, seq_len)
            multimodal_features: 多模态特征字典，每个值的形状为 (batch, seq_len, feature_dim)
            seq_lengths: 序列实际长度 (batch,)
            target_items: 目标物品ID (batch,)，用于训练（全库模式）
            candidate_items: 候选物品 (batch, num_candidates)，用于训练（采样模式）⭐
            labels: 候选物品的标签 (batch, num_candidates)，0/1 ⭐
            return_loss: 是否计算损失
            return_explanations: 是否返回可解释性信息

        Returns:
            包含预测、损失和其他信息的字典
        """
        batch_size, seq_len = item_ids.size()

        # ==================== 1. 物品嵌入 + 解耦表征学习 ====================
        
        # ⭐⭐⭐ P0核心修复：分离item embedding和多模态特征的处理！
        # Item embedding应该独立学习，不应被解耦模块破坏
        item_embeddings = self.item_embedding(item_ids)  # (batch, seq_len, item_embed_dim)

        # 处理每个时间步的多模态特征（仅text和image）
        all_disentangled_features = []
        disentangled_losses = []
        # ⭐ 保存最后一个时间步的modality_disentangled（用于SCM）
        last_modality_disentangled = None

        for t in range(seq_len):
            # ⭐⭐⭐ P0修复：只用text和image进行解耦，不包含item embedding！
            multimodal_t = {
                modality: features[:, t, :]
                for modality, features in multimodal_features.items()
            }
            # ⭐⭐⭐ 移除：不再将item embedding作为模态输入解耦模块！
            
            # ⭐⭐⭐ P1: 应用投影降维（大幅减少参数量）
            if self.use_projection:
                multimodal_t_proj = {}
                for modality, feat in multimodal_t.items():
                    if modality in self.modality_projections:
                        multimodal_t_proj[modality] = self.modality_projections[modality](feat)
                    else:
                        multimodal_t_proj[modality] = feat
                multimodal_t = multimodal_t_proj

            # 应用解耦表征学习（只对text和image）
            disentangled_output = self.disentangled_module(
                multimodal_t,
                return_loss=return_loss
            )

            # 拼接解耦特征
            z_concat = disentangled_output['z_concat']  # (batch, 192) = 3 * 64
            
            # ⭐⭐⭐ P0修复：将item embedding直接拼接到解耦特征后
            # 这样item embedding可以独立学习，不受VAE噪声影响
            item_emb_t = item_embeddings[:, t, :]  # (batch, item_embed_dim)
            combined_features = torch.cat([z_concat, item_emb_t], dim=-1)  # (batch, 192+128=320)
            
            all_disentangled_features.append(combined_features)

            # ⭐ 保存最后一个时间步的详细信息（用于SCM）
            if t == seq_len - 1:
                last_modality_disentangled = disentangled_output.get('modality_disentangled', {})

            if return_loss:
                disentangled_losses.append(disentangled_output['loss'])

        # Stack所有时间步的解耦特征
        disentangled_sequence = torch.stack(all_disentangled_features, dim=1)  # (batch, seq_len, 320)

        # ==================== 2. 序列编码 ====================

        # 使用序列编码器捕获时序依赖
        encoded_sequence = self.sequence_encoder(disentangled_sequence, seq_lengths)  # (batch, seq_len, total_dim)

        # 取最后一个时间步作为用户表示
        if seq_lengths is not None:
            # 获取每个序列的最后一个有效位置
            last_indices = (seq_lengths - 1).long()
            user_representation = encoded_sequence[
                torch.arange(batch_size, device=encoded_sequence.device),
                last_indices
            ]
        else:
            user_representation = encoded_sequence[:, -1, :]  # (batch, total_dim)

        # ==================== 3. 量子启发多兴趣编码 ====================

        # 使用量子编码器建模用户的多样化兴趣
        quantum_output = self.quantum_encoder(
            user_representation,
            return_all_interests=True
        )

        user_interest_representation = quantum_output['output']  # (batch, item_embed_dim)

        # ==================== 4. 因果推断（⭐ 改进：基于SCM的Pearl三步法） ====================

        # ⭐ 渐进式训练：根据alpha_causal决定是否调用因果模块
        # Phase 1时alpha_causal=0.0，跳过因果推断以加速收敛
        if self.alpha_causal > 0 and last_modality_disentangled and target_items is not None:
            # ⭐ 改进：提取mu和logvar用于SCM的Abduction步骤
            # 从所有模态的解耦结果中聚合mu和logvar
            mu_dict = {}
            logvar_dict = {}
            z_dict = {}

            for dim_name in ['function', 'aesthetics', 'emotion']:
                # 收集所有模态在该维度的mu和logvar
                mu_list = []
                logvar_list = []
                z_list = []

                for modality, disentangled in last_modality_disentangled.items():
                    if dim_name in disentangled:
                        mu_list.append(disentangled[dim_name].get('mu', disentangled[dim_name]['z']))
                        logvar_list.append(disentangled[dim_name].get('logvar', torch.zeros_like(disentangled[dim_name]['z'])))
                        z_list.append(disentangled[dim_name]['z'])

                # 平均聚合（简单策略）
                if mu_list:
                    mu_dict[dim_name] = torch.stack(mu_list).mean(dim=0)
                    logvar_dict[dim_name] = torch.stack(logvar_list).mean(dim=0)
                    z_dict[dim_name] = torch.stack(z_list).mean(dim=0)
                else:
                    # Fallback：从disentangled_sequence提取（⭐ P0修复：只提取解耦部分）
                    # disentangled_sequence的前192维是解耦特征，后128维是item embedding
                    idx_start = {'function': 0, 'aesthetics': 1, 'emotion': 2}[dim_name] * self.disentangled_dim
                    idx_end = idx_start + self.disentangled_dim
                    # 只从前192维（解耦特征）提取
                    z_dict[dim_name] = disentangled_sequence[:, -1, idx_start:idx_end]
                    mu_dict[dim_name] = z_dict[dim_name]
                    logvar_dict[dim_name] = torch.zeros_like(z_dict[dim_name])

            # ⭐⭐⭐ 改进：调用SCM的完整三步推理（支持candidate模式）
            # 创建推荐打分函数（支持全库和candidate两种模式）
            def recommendation_head(quantum_output):
                user_repr_norm = F.normalize(quantum_output + 1e-8, p=2, dim=-1)
                
                if candidate_items is not None:
                    # ⭐ Candidate模式：只计算候选物品的得分
                    candidate_emb = self.item_embedding(candidate_items)  # (batch, num_candidates, dim)
                    candidate_emb_norm = F.normalize(candidate_emb + 1e-8, p=2, dim=-1)
                    
                    # 点积打分
                    user_repr_expanded = user_repr_norm.unsqueeze(1)  # (batch, 1, dim)
                    logits = (user_repr_expanded * candidate_emb_norm).sum(dim=-1) / self.temperature
                    
                    # 添加候选物品的bias
                    candidate_bias = self.item_bias[candidate_items]  # (batch, num_candidates)
                    logits = logits + candidate_bias
                else:
                    # 全库模式：计算所有物品的得分
                    item_emb_norm = F.normalize(self.item_embedding.weight + 1e-8, p=2, dim=-1)
                    logits = torch.matmul(user_repr_norm, item_emb_norm.T) / self.temperature + self.item_bias
                    logits[:, 0] = -1e9  # 屏蔽padding
                
                return logits

            # 调用SCM（现在支持candidate模式）
            # ⭐ P0修复：提取item embedding（从disentangled_sequence的后128维）
            last_item_embedding = disentangled_sequence[:, -1, -self.item_embed_dim:]  # (batch, 128)
            
            causal_output = self.causal_module.scm(
                z_dict=z_dict,
                mu_dict=mu_dict,
                logvar_dict=logvar_dict,
                quantum_encoder=self.quantum_encoder,
                recommendation_head=recommendation_head,
                target_items=target_items,
                candidate_items=candidate_items,  # ⭐ 传递candidate_items
                item_embedding=last_item_embedding  # ⭐ P0修复：传递item_embedding
            )

            # 添加不确定性量化（向后兼容）
            # ⭐ P0修复：不确定性量化使用完整特征（包括item embedding）
            last_features_full = disentangled_sequence[:, -1, :]  # (batch, 320)
            uncertainty = self.causal_module.uncertainty_quantification(
                last_features_full,
                num_mc_samples=self.num_mc_samples
            )
            causal_output['uncertainty'] = uncertainty
        else:
            # Phase 1: 跳过因果推断，返回空字典
            # ⭐ P0修复：使用完整特征（包括item embedding）
            causal_output = {
                'counterfactuals': {},
                'causal_effects': {},
                'uncertainty': torch.zeros(batch_size, device=item_ids.device),
                'original_features': disentangled_sequence[:, -1, :]  # (batch, 320)
            }

        # ==================== 5. 推荐预测 ====================

        # ⭐⭐⭐ 关键修复: L2归一化 + 温度缩放，防止范数失衡
        # 用户表征归一化（添加epsilon防止零向量导致NaN）
        user_repr_norm = F.normalize(user_interest_representation + 1e-8, p=2, dim=-1)  # (batch, embed_dim)
        
        # 物品embedding归一化（添加epsilon防止零向量导致NaN）
        item_emb_norm = F.normalize(self.item_embedding.weight + 1e-8, p=2, dim=-1)  # (num_items+1, embed_dim)
        
        # 使用固定的温度参数（已在__init__中定义为buffer）
        temperature = self.temperature  # 固定值0.2，不再裁剪
        
        # ⭐⭐⭐ P0核心修复: 统一用点积打分（权重共享）
        if candidate_items is not None:
            # 采样模式：只对候选物品打分（训练时可选）
            candidate_embeddings = self.item_embedding(candidate_items)  # (batch, num_candidates, embed_dim)
            candidate_emb_norm = F.normalize(candidate_embeddings, p=2, dim=-1)
            user_repr_expanded = user_repr_norm.unsqueeze(1)  # (batch, 1, embed_dim)
            recommendation_logits = (user_repr_expanded * candidate_emb_norm).sum(dim=-1) / temperature  # (batch, num_candidates)
            # 添加候选物品的bias
            candidate_bias = self.item_bias[candidate_items]  # (batch, num_candidates)
            recommendation_logits = recommendation_logits + candidate_bias
        else:
            # 全库模式：对所有物品打分（评估/训练默认）
            # ⭐ 归一化点积打分: logits = (user_norm @ item_norm.T) / temperature + bias
            recommendation_logits = torch.matmul(
                user_repr_norm,  # (batch, embed_dim)
                item_emb_norm.T   # (num_items+1, embed_dim).T
            ) / temperature + self.item_bias  # (batch, num_items+1)
            
            # ⭐⭐⭐ P0关键: 屏蔽padding (item_id=0)，防止污染Top-K
            recommendation_logits[:, 0] = -1e9

        # 预测维度重要性（用于可解释性）
        dimension_importance = self.dimension_importance_head(user_representation)
        dimension_importance = F.softmax(dimension_importance, dim=-1)

        # ==================== 结果汇总 ====================

        results = {
            'recommendation_logits': recommendation_logits,
            'user_representation': user_interest_representation,
            'dimension_importance': dimension_importance,
            'quantum_output': quantum_output,
            'causal_output': causal_output,
            'disentangled_sequence': disentangled_sequence
        }

        # ==================== 6. 损失计算 ====================

        if return_loss:
            # ⭐⭐⭐ 使用BPR损失（更适合隐式反馈Top-K推荐）
            if target_items is not None:
                batch_indices = torch.arange(batch_size, device=target_items.device)
                
                # ⭐⭐⭐ 判断是使用dataloader提供的候选物品，还是在forward中采样
                if candidate_items is not None and labels is not None:
                    # 情况1: dataloader已经提供了候选物品（1正+K负）
                    # recommendation_logits的形状: (batch, num_candidates)
                    # candidate_items[i, 0] 是正样本，candidate_items[i, 1:] 是负样本
                    
                    pos_scores = recommendation_logits[:, 0]  # (batch,) 第0个是正样本
                    neg_scores = recommendation_logits[:, 1:]  # (batch, num_negatives) 剩余的是负样本
                    
                    # ⭐ 使用统一的BPR损失函数
                    recommendation_loss = self.bpr_loss_fn(pos_scores, neg_scores)
                    
                else:
                    # 情况2: 全库模式，在forward中采样负样本
                    # recommendation_logits的形状: (batch, num_items+1)
                    
                    # 计算正样本得分
                    pos_scores = recommendation_logits[batch_indices, target_items]  # (batch,)
                    
                    # BPR损失：高效采样负样本
                    # ⭐ 使用self.num_negatives（从config读取）
                    num_negatives = self.num_negatives
                    
                    # ⭐ 硬负样本采样：排除正样本与历史交互，从高分项目中选负样本
                    # 构造用于选择负样本的分数副本（不影响真实打分）
                    scores_for_neg = recommendation_logits.clone()
                    
                    # 排除正样本
                    scores_for_neg[batch_indices, target_items] = -1e9
                    
                    # 排除历史交互（包含padding=0；0列已在上游置为-1e9）
                    batch_idx_exp = batch_indices.unsqueeze(1).expand_as(item_ids)
                    scores_for_neg[batch_idx_exp, item_ids] = -1e9
                    
                    # 从剩余最高分的项目中选取K个负样本（硬负样本）
                    neg_items = torch.topk(scores_for_neg, k=num_negatives, dim=-1).indices  # (batch, num_negatives)
                    
                    # 计算负样本得分（使用原始logits）
                    neg_scores = recommendation_logits[batch_indices.unsqueeze(1), neg_items]  # (batch, num_negatives)
                    
                    # ⭐ 使用统一的BPR损失函数
                    recommendation_loss = self.bpr_loss_fn(pos_scores, neg_scores)
            else:
                recommendation_loss = torch.tensor(0.0, device=item_ids.device)

            # 解耦表征损失
            avg_disentangled_loss = torch.stack(disentangled_losses).mean() if disentangled_losses else 0.0

            # 多样性损失（使用量子输出中的损失）
            diversity_loss = 0.0
            
            # ⭐ 直接使用量子编码器计算的损失
            from models.quantum_inspired_encoder import compute_quantum_losses
            
            if 'metrics' in quantum_output:
                # 使用量子内置的损失函数
                quantum_loss = compute_quantum_losses(quantum_output)
                diversity_loss = quantum_loss

            # ⭐ 因果损失 - 改进版：使用SCM返回的理论严谨损失
            # 如果SCM已经计算了causal_loss，直接使用
            if 'causal_loss' in causal_output and isinstance(causal_output['causal_loss'], torch.Tensor):
                causal_loss = causal_output['causal_loss']
            elif 'counterfactuals' in causal_output and len(causal_output['counterfactuals']) > 0:
                # 收集所有反事实预测
                all_cf_logits = []
                
                for cf_key, cf_features in causal_output['counterfactuals'].items():
                    if cf_key == 'original' or cf_features is None:
                        continue
                    
                    if isinstance(cf_features, torch.Tensor) and cf_features.size(0) > 0:
                        # 获取原始特征并重构反事实特征
                        original_features = causal_output['original_features']  # (batch, 320)
                        
                        # 确定哪个维度被干预
                        if 'function' in cf_key:
                            dim_idx = 0
                        elif 'aesthetics' in cf_key:
                            dim_idx = 1
                        elif 'emotion' in cf_key:
                            dim_idx = 2
                        else:
                            continue
                        
                        # ⭐ P0修复：重构完整特征（前192维是解耦，后128维是item embedding）
                        cf_full_features = original_features.clone()
                        start_idx = dim_idx * self.disentangled_dim
                        end_idx = start_idx + self.disentangled_dim
                        # 只修改解耦部分（前192维），保持item embedding不变
                        cf_full_features[:, start_idx:end_idx] = cf_features
                        
                        # 生成反事实预测
                        cf_quantum_output = self.quantum_encoder(
                            cf_full_features.unsqueeze(1),
                            return_all_interests=False
                        )
                        cf_user_representation = cf_quantum_output['output']
                        
                        # ⭐⭐⭐ P0: 反事实预测也用归一化点积打分（一致性）
                        # 添加epsilon防止零向量导致NaN
                        cf_user_norm = F.normalize(cf_user_representation + 1e-8, p=2, dim=-1)
                        cf_logits = torch.matmul(
                            cf_user_norm,
                            item_emb_norm.T  # 使用前面计算的归一化embedding
                        ) / temperature + self.item_bias
                        cf_logits[:, 0] = -1e9  # 屏蔽padding
                        
                        all_cf_logits.append(cf_logits)
                
                if len(all_cf_logits) > 0:
                    # ⭐ 核心损失1: 预测差异（简单直接，让模型自然学习）
                    pred_diff_loss = 0.0
                    
                    # ⭐⭐⭐ P0: 用归一化点积计算原始logits做比较
                    if labels is not None and target_items is not None:
                        # 混合模式：重新计算全库logits用于因果比较
                        original_all_logits = torch.matmul(
                            user_repr_norm,  # 使用归一化的用户表征
                            item_emb_norm.T
                        ) / temperature + self.item_bias
                        original_all_logits[:, 0] = -1e9
                        for cf_logits in all_cf_logits:
                            pred_diff = (cf_logits - original_all_logits).abs().mean()
                            pred_diff_loss += pred_diff
                    else:
                        # 纯模式：直接比较
                        for cf_logits in all_cf_logits:
                            pred_diff = (cf_logits - recommendation_logits).abs().mean()
                            pred_diff_loss += pred_diff
                    
                    causal_loss += pred_diff_loss / len(all_cf_logits)
                    
                    # ⭐ 核心损失2: ITE幅度（因果效应强度）
                    if 'causal_effects' in causal_output:
                        effects = causal_output['causal_effects']
                        if 'ite' in effects:
                            ite = effects['ite']  # (batch, num_treatments)
                            # 鼓励ITE有合理的幅度
                            ite_magnitude = ite.abs().mean()
                            target_ite = torch.tensor(self.target_ite, device=ite.device)
                            ite_loss = F.smooth_l1_loss(ite_magnitude, target_ite)
                            causal_loss += ite_loss
            else:
                # 没有因果推断输出，损失为0
                causal_loss = torch.tensor(0.0, device=item_ids.device)

            # ⭐ 总损失 - 应用KL退火
            # KL退火：前20个epoch逐渐增加KL权重，避免后验坍塌
            # 因果损失权重 = 模型静态权重 * 训练期运行时权重
            causal_weight = self.alpha_causal * getattr(self, 'alpha_causal_runtime', 1.0)
            total_loss = (
                recommendation_loss +
                self.alpha_recon * self.kl_anneal_factor * avg_disentangled_loss +
                self.alpha_diversity * diversity_loss +
                causal_weight * causal_loss
            )

            results.update({
                'loss': total_loss,
                'recommendation_loss': recommendation_loss,
                'disentangled_loss': avg_disentangled_loss,
                'diversity_loss': diversity_loss,
                'causal_loss': causal_loss
            })

        # ==================== 7. 可解释性信息 ====================

        if return_explanations:
            explanations = {
                'dimension_importance': dimension_importance,
                'uncertainty': causal_output['uncertainty'],
                'interference_strength': quantum_output['interference_strength'],
            }

            if 'dimension_importance' in causal_output:
                explanations['causal_dimension_importance'] = causal_output['dimension_importance']

            results['explanations'] = explanations

        return results

    def predict(
        self,
        item_ids: torch.Tensor,
        multimodal_features: Dict[str, torch.Tensor],
        seq_lengths: Optional[torch.Tensor] = None,
        top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测Top-K推荐

        Args:
            item_ids: 物品ID序列
            multimodal_features: 多模态特征
            seq_lengths: 序列长度
            top_k: 返回Top-K结果

        Returns:
            (top_k_items, top_k_scores)
        """
        with torch.no_grad():
            outputs = self.forward(
                item_ids,
                multimodal_features,
                seq_lengths,
                target_items=None,
                return_loss=False,
                return_explanations=False
            )

            logits = outputs['recommendation_logits']
            # 直接使用原始logits排序，避免softmax压缩分差
            top_k_scores, top_k_items = torch.topk(logits, k=top_k, dim=-1)

            return top_k_items, top_k_scores

    def explain_recommendation(
        self,
        item_ids: torch.Tensor,
        multimodal_features: Dict[str, torch.Tensor],
        seq_lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, any]:
        """
        为推荐结果提供可解释性分析

        Returns:
            包含各种可解释性指标的字典
        """
        with torch.no_grad():
            outputs = self.forward(
                item_ids,
                multimodal_features,
                seq_lengths,
                target_items=None,
                return_loss=False,
                return_explanations=True
            )

            explanations = outputs['explanations']

            # 格式化输出
            dimension_names = ['Function (功能)', 'Aesthetics (美学)', 'Emotion (情感)']

            explanation_text = {
                'dimension_importance': {
                    dimension_names[i]: float(explanations['dimension_importance'][0, i])
                    for i in range(self.num_disentangled_dims)
                },
                'uncertainty': {
                    'aleatoric': float(explanations['uncertainty']['refined_aleatoric'].mean()),
                    'epistemic': float(explanations['uncertainty']['refined_epistemic'].mean()),
                    'confidence': float(explanations['uncertainty']['confidence'].mean())
                },
                'interest_diversity': 'High' if outputs['quantum_output']['interference_strength'].var() > 0.1 else 'Low'
            }

            if 'causal_dimension_importance' in explanations:
                explanation_text['causal_importance'] = {
                    dimension_names[i]: float(explanations['causal_dimension_importance'][i])
                    for i in range(self.num_disentangled_dims)
                }

            return explanation_text

    def get_user_interests(
        self,
        item_ids: torch.Tensor,
        multimodal_features: Dict[str, torch.Tensor],
        seq_lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        提取用户的多个兴趣表示

        Returns:
            包含各个兴趣向量的字典
        """
        with torch.no_grad():
            outputs = self.forward(
                item_ids,
                multimodal_features,
                seq_lengths,
                target_items=None,
                return_loss=False,
                return_explanations=False
            )

            quantum_output = outputs['quantum_output']

            # 返回各个兴趣的复数表示
            interests = {
                f'interest_{i}': {
                    'real': quantum_output['individual_interests_real'][:, i, :],
                    'imag': quantum_output['individual_interests_imag'][:, i, :],
                }
                for i in range(self.num_interests)
            }

            # 也返回叠加后的状态
            interests['superposed'] = {
                'real': quantum_output['superposed_state_real'],
                'imag': quantum_output['superposed_state_imag']
            }

            return interests
