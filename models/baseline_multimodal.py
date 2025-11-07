"""
🎯 Step 1: Baseline多模态序列推荐器
架构：Text(BERT) + Image(ResNet) → Concat → GRU → BPR

设计原则：
- 尽可能简单
- 模块化（方便后续替换）
- 完整（包含训练/评估）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class BaselineMultimodalSeqRec(nn.Module):
    """
    Baseline多模态序列推荐器
    
    流程：
    1. Text + Image → Concat → Projection
    2. Sequence → GRU → User representation
    3. User × Item → Score
    4. BPR Loss
    """
    
    def __init__(
        self,
        text_dim: int = 768,              # BERT特征维度
        image_dim: int = 2048,            # ResNet特征维度
        item_embedding_dim: int = 128,    # 物品嵌入维度
        hidden_dim: int = 256,            # GRU隐藏维度
        num_items: int = 10000,           # 物品总数
        dropout: float = 0.2,             # Dropout率
        num_negatives: int = 100          # 负采样数量
    ):
        super().__init__()
        
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        self.num_items = num_items
        self.num_negatives = num_negatives
        
        # ===Step 1: 多模态特征融合（最简单：concat + projection）===
        multimodal_dim = text_dim + image_dim  # 768 + 2048 = 2816
        self.multimodal_proj = nn.Sequential(
            nn.Linear(multimodal_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # ===Step 2: 序列建模（GRU）===
        self.sequence_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,                 # 单层，简单
            batch_first=True,
            dropout=0.0,                   # 单层不需要dropout
            bidirectional=True             # 双向，效果更好
        )
        
        # ===Step 3: Item Embedding（用于候选打分）===
        self.item_embedding = nn.Embedding(
            num_items + 1,  # +1 for padding
            item_embedding_dim,
            padding_idx=0
        )
        
        # ===Step 4: 输出投影（对齐维度）===
        # 双向GRU输出是 2*hidden_dim
        self.user_proj = nn.Linear(hidden_dim * 2, item_embedding_dim)
        
        # Item bias（推荐系统标配，能提升性能）
        self.item_bias = nn.Parameter(torch.zeros(num_items + 1))
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # Item embedding
        nn.init.xavier_normal_(self.item_embedding.weight[1:])  # 跳过padding
        nn.init.zeros_(self.item_embedding.weight[0])
        
        # Multimodal projection
        for module in self.multimodal_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        # GRU
        for name, param in self.sequence_encoder.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # User projection
        nn.init.xavier_uniform_(self.user_proj.weight)
        nn.init.zeros_(self.user_proj.bias)
    
    def forward(
        self,
        text_seq: torch.Tensor,           # (batch, seq_len, 768)
        image_seq: torch.Tensor,          # (batch, seq_len, 2048)
        seq_lengths: torch.Tensor,        # (batch,)
        target_items: Optional[torch.Tensor] = None,  # (batch,) 训练用
        candidate_items: Optional[torch.Tensor] = None,  # (batch, num_cand) 训练用
        return_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            text_seq: 文本序列特征
            image_seq: 图像序列特征
            seq_lengths: 每个序列的实际长度
            target_items: 目标item ID（训练时用）
            candidate_items: 候选item ID（负采样，训练时用）
            return_loss: 是否返回损失
        
        Returns:
            包含user表示和预测得分的字典
        """
        batch_size, seq_len, _ = text_seq.shape
        device = text_seq.device
        
        # ===Step 1: 多模态融合===
        # Concat text + image
        multimodal_feat = torch.cat([text_seq, image_seq], dim=-1)  # (batch, seq_len, 2816)
        
        # Projection
        multimodal_embed = self.multimodal_proj(multimodal_feat)  # (batch, seq_len, hidden_dim)
        
        # ===Step 2: 序列编码（GRU）===
        # Pack padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(
            multimodal_embed,
            seq_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        packed_output, hidden = self.sequence_encoder(packed_input)
        
        # Unpack
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=seq_len
        )
        
        # 取最后时刻的输出作为用户表示
        device = text_seq.device  # 从输入tensor获取device
        batch_indices = torch.arange(batch_size, device=device)
        last_indices = (seq_lengths - 1).long().to(device)  # 确保在正确的device上
        user_repr = output[batch_indices, last_indices]  # (batch, hidden_dim*2)
        
        # ===Step 3: 投影到item空间===
        user_embed = self.user_proj(user_repr)  # (batch, item_embedding_dim)
        
        # L2归一化（提升稳定性）
        user_embed = F.normalize(user_embed, p=2, dim=-1)
        
        # ===Step 4: 打分===
        # 获取所有item的embedding（归一化）
        all_item_embeds = self.item_embedding.weight  # (num_items+1, item_embedding_dim)
        all_item_embeds_norm = F.normalize(all_item_embeds, p=2, dim=-1)
        
        if candidate_items is not None:
            # 训练模式：计算候选item的得分
            candidate_embeds = all_item_embeds_norm[candidate_items]  # (batch, num_cand, embed_dim)
            logits = torch.bmm(
                candidate_embeds,
                user_embed.unsqueeze(-1)
            ).squeeze(-1)  # (batch, num_cand)
            
            # 加bias
            logits = logits + self.item_bias[candidate_items]
        else:
            # 推理模式：计算所有item的得分
            logits = torch.matmul(user_embed, all_item_embeds_norm.T)  # (batch, num_items+1)
            logits = logits + self.item_bias
            
            # Mask padding item
            logits[:, 0] = -1e9
        
        results = {
            'recommendation_logits': logits,
            'user_representation': user_embed
        }
        
        # ===Step 5: BPR损失===
        if return_loss and target_items is not None and candidate_items is not None:
            # 第一个是正样本，其余是负样本
            pos_scores = logits[:, 0]  # (batch,)
            neg_scores = logits[:, 1:]  # (batch, num_neg)
            
            # BPR损失：-log(σ(pos - neg))
            diff = pos_scores.unsqueeze(1) - neg_scores  # (batch, num_neg)
            loss = -F.logsigmoid(diff).mean()
            
            results['loss'] = loss
            results['bpr_loss'] = loss
        
        return results


# 测试函数
def test_baseline_model():
    """测试模型是否能正常运行"""
    print("=" * 80)
    print("Testing BaselineMultimodalSeqRec")
    print("=" * 80)
    
    # 超参数
    batch_size = 4
    seq_len = 10
    text_dim = 768
    image_dim = 2048
    num_items = 100
    num_negatives = 5
    
    # 创建模型
    model = BaselineMultimodalSeqRec(
        text_dim=text_dim,
        image_dim=image_dim,
        item_embedding_dim=64,
        hidden_dim=128,
        num_items=num_items,
        dropout=0.2,
        num_negatives=num_negatives
    )
    
    print(f"✓ Model created")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {total_params / 1e6:.2f}M")
    
    # 模拟数据
    text_seq = torch.randn(batch_size, seq_len, text_dim)
    image_seq = torch.randn(batch_size, seq_len, image_dim)
    seq_lengths = torch.tensor([10, 8, 6, 9])
    target_items = torch.randint(1, num_items + 1, (batch_size,))
    
    # 构建候选物品（1正 + K负）
    candidate_items = torch.zeros(batch_size, num_negatives + 1, dtype=torch.long)
    candidate_items[:, 0] = target_items  # 第一个是正样本
    for i in range(batch_size):
        # 随机采样负样本
        neg_items = torch.randint(1, num_items + 1, (num_negatives,))
        candidate_items[i, 1:] = neg_items
    
    print(f"✓ Test data created")
    
    # 前向传播（训练模式）
    model.train()
    outputs = model(
        text_seq=text_seq,
        image_seq=image_seq,
        seq_lengths=seq_lengths,
        target_items=target_items,
        candidate_items=candidate_items,
        return_loss=True
    )
    
    print(f"✓ Forward pass (train mode) succeeded")
    print(f"  - Logits shape: {outputs['recommendation_logits'].shape}")
    print(f"  - User repr shape: {outputs['user_representation'].shape}")
    print(f"  - BPR Loss: {outputs['bpr_loss'].item():.4f}")
    
    # 反向传播
    loss = outputs['loss']
    loss.backward()
    print(f"✓ Backward pass succeeded")
    
    # 前向传播（评估模式）
    model.eval()
    with torch.no_grad():
        outputs = model(
            text_seq=text_seq,
            image_seq=image_seq,
            seq_lengths=seq_lengths,
            target_items=None,
            candidate_items=None,
            return_loss=False
        )
    
    print(f"✓ Forward pass (eval mode) succeeded")
    print(f"  - Logits shape: {outputs['recommendation_logits'].shape}")
    print(f"  - Expected: ({batch_size}, {num_items + 1})")
    
    print("=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)


if __name__ == '__main__':
    test_baseline_model()

