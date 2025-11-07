"""
渐进式多模态推荐器
支持通过配置逐步启用功能：Baseline → +解耦 → +量子 → +因果

设计理念：
- 所有功能关闭时 = Baseline
- 逐步打开功能，观察增益
- 模块化设计，便于消融实验
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

try:
    from .disentangled_representation import DisentangledRepresentation
    from .quantum_inspired_encoder import QuantumInspiredMultiInterestEncoder
    from .causal_inference import CausalInferenceModule
    from utils.losses import BPRLoss
except ImportError:
    # 当作为脚本直接运行时
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.disentangled_representation import DisentangledRepresentation
    from models.quantum_inspired_encoder import QuantumInspiredMultiInterestEncoder
    from models.causal_inference import CausalInferenceModule
    from utils.losses import BPRLoss


class ProgressiveMultimodalRecommender(nn.Module):
    """
    渐进式多模态推荐器
    
    配置示例：
    - Stage 0 (Baseline): enable_disentangled=False, enable_quantum=False, enable_causal=False
    - Stage 1 (+解耦): enable_disentangled=True, enable_quantum=False, enable_causal=False
    - Stage 2 (+量子): enable_disentangled=True, enable_quantum=True, enable_causal=False
    - Stage 3 (完整): enable_disentangled=True, enable_quantum=True, enable_causal=True
    """
    
    def __init__(
        self,
        # 基础配置
        text_dim: int = 768,
        image_dim: int = 2048,
        item_embed_dim: int = 128,
        hidden_dim: int = 256,
        num_items: int = 10000,
        
        # 功能开关 ⭐⭐⭐
        enable_disentangled: bool = False,  # 是否启用解耦模块
        enable_quantum: bool = False,       # 是否启用量子编码器
        enable_causal: bool = False,        # 是否启用因果推断
        
        # 解耦配置（仅当enable_disentangled=True时生效）
        disentangled_dim: int = 64,
        num_disentangled_dims: int = 3,
        
        # 量子配置（仅当enable_quantum=True时生效）
        num_interests: int = 4,
        quantum_state_dim: int = 128,
        
        # 因果配置（仅当enable_causal=True时生效）
        num_ensembles: int = 3,
        num_mc_samples: int = 10,
        target_ite: float = 0.3,
        
        # 损失权重
        alpha_recon: float = 0.01,
        alpha_causal: float = 0.005,
        alpha_diversity: float = 0.001,
        beta: float = 0.05,
        
        # 其他
        dropout: float = 0.2,
        num_negatives: int = 100,
        temperature: float = 0.5,
        use_quantum_computing: bool = False
    ):
        super().__init__()
        
        self.enable_disentangled = enable_disentangled
        self.enable_quantum = enable_quantum
        self.enable_causal = enable_causal
        
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        self.item_embed_dim = item_embed_dim
        self.num_items = num_items
        
        # 损失权重
        self.alpha_recon = alpha_recon if enable_disentangled else 0.0
        self.alpha_causal = alpha_causal if enable_causal else 0.0
        self.alpha_diversity = alpha_diversity if enable_quantum else 0.0
        self.beta = beta if enable_disentangled else 0.0
        
        print("\n" + "="*80)
        print("🎯 渐进式推荐器配置")
        print("="*80)
        print(f"解耦模块: {'✅' if enable_disentangled else '❌'}")
        print(f"量子编码器: {'✅' if enable_quantum else '❌'}")
        print(f"因果推断: {'✅' if enable_causal else '❌'}")
        print("="*80 + "\n")
        
        # ==================== 多模态融合 ====================
        
        if enable_disentangled:
            # 使用解耦模块
            self.disentangled_module = DisentangledRepresentation(
                input_dims={'text': text_dim, 'image': image_dim},
                latent_dim=disentangled_dim,
                beta=beta
            )
            # 解耦后的特征维度
            fusion_dim = disentangled_dim * num_disentangled_dims
        else:
            # Baseline: 简单concat
            self.multimodal_proj = nn.Sequential(
                nn.Linear(text_dim + image_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            fusion_dim = hidden_dim
        
        # ==================== Item Embedding ====================
        
        self.item_embedding = nn.Embedding(
            num_items + 1,
            item_embed_dim,
            padding_idx=0
        )
        
        # ==================== 序列编码器 ====================
        
        # 输入维度 = fusion_dim + item_embed_dim
        sequence_input_dim = fusion_dim + item_embed_dim
        
        self.sequence_encoder = nn.GRU(
            input_size=sequence_input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=True
        )
        
        # GRU输出投影
        self.sequence_proj = nn.Linear(hidden_dim * 2, sequence_input_dim)
        
        # ==================== 量子编码器（可选）====================
        
        if enable_quantum:
            self.quantum_encoder = QuantumInspiredMultiInterestEncoder(
                input_dim=sequence_input_dim,
                num_interests=num_interests,
                qubit_dim=quantum_state_dim // 2,
                output_dim=item_embed_dim,
                hidden_dim=hidden_dim,
                use_quantum_computing=use_quantum_computing
            )
            user_repr_dim = item_embed_dim
        else:
            # 直接投影到item空间
            self.user_proj = nn.Linear(sequence_input_dim, item_embed_dim)
            user_repr_dim = item_embed_dim
        
        # ==================== 因果推断（可选）====================
        
        if enable_causal:
            self.causal_module = CausalInferenceModule(
                disentangled_dim=disentangled_dim if enable_disentangled else hidden_dim,
                num_dimensions=num_disentangled_dims if enable_disentangled else 1,
                hidden_dim=hidden_dim,
                num_ensembles=num_ensembles,
                feature_dim=sequence_input_dim
            )
            self.num_mc_samples = num_mc_samples
            self.target_ite = target_ite
        
        # ==================== 打分 ====================
        
        self.item_bias = nn.Parameter(torch.zeros(num_items + 1))
        self.register_buffer('temperature', torch.tensor(temperature))
        
        # ==================== 损失函数 ====================
        
        self.bpr_loss_fn = BPRLoss()
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
    
    def forward(
        self,
        item_ids: torch.Tensor,
        multimodal_features: Dict[str, torch.Tensor],
        seq_lengths: torch.Tensor,
        target_items: Optional[torch.Tensor] = None,
        candidate_items: Optional[torch.Tensor] = None,
        return_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            item_ids: (batch, seq_len)
            multimodal_features: {'text': (batch, seq_len, 768), 'image': (batch, seq_len, 2048)}
            seq_lengths: (batch,)
            target_items: (batch,)
            candidate_items: (batch, num_cand)
            return_loss: bool
        """
        batch_size, seq_len = item_ids.shape
        device = item_ids.device
        
        # ==================== 1. 多模态融合 ====================
        
        text_seq = multimodal_features['text']  # (batch, seq_len, 768)
        image_seq = multimodal_features['image']  # (batch, seq_len, 2048)
        
        if self.enable_disentangled:
            # 使用解耦模块处理每个时间步
            all_fused = []
            disentangled_losses = []
            
            for t in range(seq_len):
                multimodal_t = {
                    'text': text_seq[:, t, :],
                    'image': image_seq[:, t, :]
                }
                
                disentangled_out = self.disentangled_module(
                    multimodal_t,
                    return_loss=return_loss
                )
                
                all_fused.append(disentangled_out['z_concat'])
                
                if return_loss:
                    disentangled_losses.append(disentangled_out['loss'])
            
            fused_features = torch.stack(all_fused, dim=1)  # (batch, seq_len, fusion_dim)
            
            if return_loss:
                recon_loss = torch.stack(disentangled_losses).mean()
            else:
                recon_loss = None
        else:
            # Baseline: 简单concat
            multimodal_concat = torch.cat([text_seq, image_seq], dim=-1)  # (batch, seq_len, 2816)
            fused_features = self.multimodal_proj(multimodal_concat)  # (batch, seq_len, hidden_dim)
            recon_loss = None
        
        # ==================== 2. 拼接Item Embedding ====================
        
        item_embeds = self.item_embedding(item_ids)  # (batch, seq_len, item_embed_dim)
        
        # 拼接多模态特征和item embedding
        combined_features = torch.cat([fused_features, item_embeds], dim=-1)  # (batch, seq_len, fusion_dim + item_embed_dim)
        
        # ==================== 3. 序列编码 ====================
        
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            combined_features,
            seq_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        output, _ = self.sequence_encoder(packed)
        
        # Unpack
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output,
            batch_first=True,
            total_length=seq_len
        )
        
        output = self.sequence_proj(output)  # (batch, seq_len, sequence_input_dim)
        
        # 取最后时刻
        batch_indices = torch.arange(batch_size, device=device)
        last_indices = (seq_lengths - 1).long()
        user_repr = output[batch_indices, last_indices]  # (batch, sequence_input_dim)
        
        # ==================== 4. 用户表征（量子编码 or 直接投影）====================
        
        if self.enable_quantum:
            quantum_output = self.quantum_encoder(
                user_repr,
                return_all_interests=True
            )
            user_embed = quantum_output['output']  # (batch, item_embed_dim)
            # ⭐ diversity现在是"相似度惩罚"，直接作为损失（越小越好=兴趣越正交）
            if return_loss:
                diversity_loss = quantum_output['metrics']['diversity']
            else:
                diversity_loss = None
        else:
            user_embed = self.user_proj(user_repr)  # (batch, item_embed_dim)
            diversity_loss = None
        
        # 归一化
        user_embed = F.normalize(user_embed, p=2, dim=-1)
        
        # ==================== 5. 因果推断（可选）====================
        
        causal_loss = None
        if self.enable_causal and return_loss and target_items is not None and self.enable_disentangled:
            # 只有开启了解耦模块，才能做因果推断（需要解耦的特征）
            # 需要从解耦模块收集每个时间步的特征
            # 这里简化：使用最后一个时间步的解耦特征
            
            # 获取最后时间步的解耦特征
            last_step_multimodal = {
                'text': text_seq[batch_indices, last_indices, :],
                'image': image_seq[batch_indices, last_indices, :]
            }
            
            # 获取解耦特征（包含完整的VAE输出）
            last_disentangled = self.disentangled_module(
                last_step_multimodal,
                return_loss=False
            )
            
            # 提取需要的特征
            z_dict = {
                'emotion': last_disentangled['z_emotion'],
                'aesthetics': last_disentangled['z_aesthetics'],
                'function': last_disentangled['z_context']  # context作为function
            }
            
            mu_dict = {
                'emotion': last_disentangled['modality_disentangled']['text']['emotion']['mu'],
                'aesthetics': last_disentangled['modality_disentangled']['image']['aesthetics']['mu'],
                'function': last_disentangled['context_full']['mu']
            }
            
            logvar_dict = {
                'emotion': last_disentangled['modality_disentangled']['text']['emotion']['logvar'],
                'aesthetics': last_disentangled['modality_disentangled']['image']['aesthetics']['logvar'],
                'function': last_disentangled['context_full']['logvar']
            }
            
            # 创建推荐打分函数
            def recommendation_head(user_embedding):
                """计算推荐得分"""
                user_embedding_norm = F.normalize(user_embedding, p=2, dim=-1)
                item_emb_norm = F.normalize(self.item_embedding.weight, p=2, dim=-1)
                scores = torch.matmul(user_embedding_norm, item_emb_norm.T)
                scores = scores / self.temperature + self.item_bias
                return scores
            
            # 获取item embedding（用于拼接）
            last_item_embeds = item_embeds[batch_indices, last_indices, :]
            
            # 调用因果推断模块
            if self.enable_quantum:
                # 使用量子编码器
                causal_output = self.causal_module.scm(
                    z_dict=z_dict,
                    mu_dict=mu_dict,
                    logvar_dict=logvar_dict,
                    quantum_encoder=self.quantum_encoder,
                    recommendation_head=recommendation_head,
                    target_items=target_items,
                    candidate_items=candidate_items,
                    item_embedding=last_item_embeds
                )
            else:
                # 不使用量子编码器，用简单的投影
                class SimpleProjector(nn.Module):
                    def __init__(self, proj):
                        super().__init__()
                        self.proj = proj
                    def __call__(self, x):
                        return {'output': self.proj(x)}
                
                simple_encoder = SimpleProjector(self.user_proj)
                
                causal_output = self.causal_module.scm(
                    z_dict=z_dict,
                    mu_dict=mu_dict,
                    logvar_dict=logvar_dict,
                    quantum_encoder=simple_encoder,
                    recommendation_head=recommendation_head,
                    target_items=target_items,
                    candidate_items=candidate_items,
                    item_embedding=last_item_embeds
                )
            
            causal_loss = causal_output['causal_loss']
        elif self.enable_causal and not self.enable_disentangled:
            # 因果推断需要解耦特征，如果没有开启解耦，损失为0
            causal_loss = torch.tensor(0.0, device=device)
        else:
            causal_loss = None
        
        # ==================== 6. 打分 ====================
        
        item_emb_norm = F.normalize(self.item_embedding.weight, p=2, dim=-1)
        
        if candidate_items is not None:
            # 候选模式
            candidate_emb = item_emb_norm[candidate_items]  # (batch, num_cand, dim)
            logits = torch.bmm(
                candidate_emb,
                user_embed.unsqueeze(-1)
            ).squeeze(-1)  # (batch, num_cand)
            logits = logits / self.temperature + self.item_bias[candidate_items]
        else:
            # 全库模式
            logits = torch.matmul(user_embed, item_emb_norm.T)  # (batch, num_items+1)
            logits = logits / self.temperature + self.item_bias
            logits[:, 0] = -1e9  # mask padding
        
        results = {
            'recommendation_logits': logits,
            'user_representation': user_embed
        }
        
        # ==================== 7. 损失计算 ====================
        
        if return_loss and target_items is not None and candidate_items is not None:
            # BPR损失
            pos_scores = logits[:, 0]
            neg_scores = logits[:, 1:]
            bpr_loss = self.bpr_loss_fn(pos_scores, neg_scores)
            
            # 总损失
            total_loss = bpr_loss
            
            if recon_loss is not None:
                total_loss = total_loss + self.alpha_recon * recon_loss
            
            if diversity_loss is not None:
                total_loss = total_loss + self.alpha_diversity * diversity_loss
            
            if causal_loss is not None:
                total_loss = total_loss + self.alpha_causal * causal_loss
            
            results['loss'] = total_loss
            results['bpr_loss'] = bpr_loss
            results['recon_loss'] = recon_loss if recon_loss is not None else torch.tensor(0.0)
            results['diversity_loss'] = diversity_loss if diversity_loss is not None else torch.tensor(0.0)
            results['causal_loss'] = causal_loss if causal_loss is not None else torch.tensor(0.0)
        
        return results


def test_progressive_model():
    """测试不同配置"""
    print("="*80)
    print("测试渐进式模型")
    print("="*80)
    
    batch_size = 4
    seq_len = 10
    num_items = 100
    
    # 模拟数据
    item_ids = torch.randint(1, num_items, (batch_size, seq_len))
    multimodal_features = {
        'text': torch.randn(batch_size, seq_len, 768),
        'image': torch.randn(batch_size, seq_len, 2048)
    }
    seq_lengths = torch.tensor([10, 8, 6, 9])
    target_items = torch.randint(1, num_items, (batch_size,))
    candidate_items = torch.randint(1, num_items, (batch_size, 5))
    candidate_items[:, 0] = target_items  # 第一个是正样本
    
    configs = [
        ("Stage 0: Baseline", False, False, False),
        ("Stage 1: +解耦", True, False, False),
        ("Stage 2: +量子", True, True, False),
        ("Stage 3: 完整", True, True, True),
    ]
    
    for name, dis, quan, cau in configs:
        print(f"\n{'='*80}")
        print(f"测试 {name}")
        print(f"{'='*80}")
        
        model = ProgressiveMultimodalRecommender(
            num_items=num_items,
            enable_disentangled=dis,
            enable_quantum=quan,
            enable_causal=cau
        )
        
        outputs = model(
            item_ids=item_ids,
            multimodal_features=multimodal_features,
            seq_lengths=seq_lengths,
            target_items=target_items,
            candidate_items=candidate_items,
            return_loss=True
        )
        
        print(f"✅ 前向传播成功")
        print(f"  - Logits shape: {outputs['recommendation_logits'].shape}")
        
        def to_float(x):
            return x.item() if hasattr(x, 'item') else x
        
        print(f"  - Loss: {to_float(outputs['loss']):.4f}")
        print(f"    ├─ BPR: {to_float(outputs['bpr_loss']):.4f}")
        print(f"    ├─ Recon: {to_float(outputs['recon_loss']):.4f}")
        print(f"    ├─ Diversity: {to_float(outputs['diversity_loss']):.4f}")
        print(f"    └─ Causal: {to_float(outputs['causal_loss']):.4f}")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  - 参数量: {total_params/1e6:.2f}M")
    
    print("\n" + "="*80)
    print("✅ 所有测试通过！")
    print("="*80)


if __name__ == '__main__':
    test_progressive_model()

