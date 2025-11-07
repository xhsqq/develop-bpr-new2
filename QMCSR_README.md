# QMCSR: Quantum-Inspired Multi-Modal Causal Sequential Recommendation

完整的QMCSR框架实现（简化版，避免过拟合）

## 📋 框架概述

QMCSR是一个创新的推荐系统框架，结合了以下三个核心模块：

```
输入: User History [item_1, ..., item_n] + Multi-Modal Features (Text + Image)
         ↓
┌─────────────────────────────────────────────────────────┐
│  Module 1: Multi-Modal Feature Extraction              │
│  ID Embedding + Text (BERT) + Image (ResNet)           │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│  Module 2: Aesthetic-Emotional Disentanglement ⭐⭐⭐  │
│  - Visual → Aesthetic (美学吸引力)                      │
│  - Text → Emotional (情感煽动性)                        │
│  - 正交约束: L_ortho = |cos(h_aes, h_emo)|            │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│  Module 3: Quantum-Inspired Multi-Interest ⭐⭐⭐⭐     │
│  - 幅度+相位编码: ψ_k = A_k * e^(i*φ_k)              │
│  - 干涉效应：相似兴趣增强，相反兴趣抵消                │
│  - 量子测量: |ψ|² = real² + imag²                     │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│  Module 4: Causal Debiasing ⭐⭐⭐⭐⭐                  │
│  - 维度级反事实生成                                     │
│  - Individual Treatment Effect (ITE) 估计              │
│  - 从多兴趣表示中去除bias                               │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│  Module 5: Prediction                                   │
│  score = h_debiased · item_emb                          │
│  Loss = L_bpr + α₁·L_ortho + α₂·L_causal               │
└─────────────────────────────────────────────────────────┘
```

## 🎯 核心创新点

### 1. Aesthetic-Emotional Disentanglement（美学-情感解耦）
- **创新**: 将多模态特征解耦为美学和情感两个独立维度
- **优势**:
  - 正交约束确保两个维度独立
  - 自适应融合权重学习每个维度的重要性
  - 为后续因果推断提供明确的干预目标

### 2. Quantum-Inspired Multi-Interest Encoder（量子启发多兴趣编码器）
- **创新**: 使用复数表示（幅度+相位）建模用户多样化兴趣
- **优势**:
  - 幅度捕捉兴趣重要性，相位捕捉兴趣特性
  - 干涉机制自然建模兴趣相互作用
  - 量子测量提供概率性的兴趣坍缩

### 3. Disentanglement-driven Causal Inference（解耦驱动因果推断）
- **创新**: 在多兴趣表示上进行维度级因果去偏
- **优势**:
  - 解耦特征提供明确的干预目标
  - 多兴趣表示提供丰富的用户信息
  - ITE估计量化每个维度的因果效应

## 📂 文件结构

```
.
├── models/
│   ├── qmcsr_complete.py          # 完整的QMCSR模型实现
│   ├── multimodal_recommender.py  # 原始多模态推荐模型
│   ├── disentangled_representation.py
│   ├── quantum_inspired_encoder.py
│   └── causal_inference.py
├── config_qmcsr.yaml              # QMCSR配置文件
├── train_qmcsr.py                 # QMCSR训练脚本
├── test_qmcsr.py                  # QMCSR测试脚本
└── QMCSR_README.md                # 本文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

```bash
# 下载Amazon数据集
python data/download_amazon.py --category beauty

# 预处理数据
python data/preprocess_amazon.py --category beauty

# 提取多模态特征
python scripts/extract_text_features.py --category beauty
python scripts/extract_image_features.py --category beauty
```

### 3. 测试模型

```bash
# 运行单元测试
python test_qmcsr.py
```

### 4. 训练模型

```bash
# 使用默认配置训练
python train_qmcsr.py --config config_qmcsr.yaml --device cuda

# 使用CPU训练
python train_qmcsr.py --config config_qmcsr.yaml --device cpu
```

## ⚙️ 配置说明

### 模型配置（简化版，避免过拟合）

```yaml
model:
  # 模态维度
  text_dim: 768           # BERT embedding
  image_dim: 2048         # ResNet embedding
  item_embed_dim: 64      # ⭐ 简化：更小的embedding

  # 解耦配置（2维）
  disentangled_dim: 32    # ⭐ 简化：更小的维度
  num_disentangled_dims: 2  # Aesthetic + Emotional

  # 量子编码器
  num_interests: 4        # ⭐ 简化：更少的兴趣

  # 序列编码器
  hidden_dim: 64          # ⭐ 简化：更小的隐藏层
  num_layers: 1           # ⭐ 简化：单层GRU

  # 正则化
  dropout: 0.2
```

### 训练配置

```yaml
training:
  batch_size: 64          # ⭐ 适中的batch size
  epochs: 30              # ⭐ 更少的epochs避免过拟合
  learning_rate: 0.001
  weight_decay: 0.0001    # ⭐ L2正则化

  # 渐进式训练
  progressive:
    phase1_epochs: 10     # Phase 1: 不使用因果
    phase2_epochs: 20     # Phase 2: 启用因果
```

### 损失权重

```yaml
loss:
  alpha_ortho: 0.1        # 正交约束
  alpha_causal: 0.05      # 因果损失
  num_negatives: 5        # ⭐ 更少的负样本
```

## 📊 模型参数量

在Beauty数据集上（~12k items）：

| 模块 | 参数量 |
|------|--------|
| Item Embedding | ~786k |
| Disentanglement | ~355k |
| Sequence Encoder | ~27k |
| Quantum Encoder | ~35k |
| Causal Debiasing | ~24k |
| **Total** | **~1.2M** |

模型大小: ~5 MB (FP32)

## 🎓 渐进式训练策略

QMCSR采用两阶段训练策略避免过拟合：

### Phase 1 (Epochs 1-10): 基础训练
- **目标**: 学习基础的序列表示和多兴趣编码
- **配置**: `use_causal = False`
- **损失**: `L_rec + α₁·L_ortho`
- **优势**: 快速收敛，建立稳定的基础表示

### Phase 2 (Epochs 11-30): 因果增强
- **目标**: 引入因果去偏提升鲁棒性
- **配置**: `use_causal = True`
- **损失**: `L_rec + α₁·L_ortho + α₂·L_causal`
- **优势**: 渐进式引入复杂性，避免训练不稳定

## 📈 评估指标

```python
evaluation:
  metrics: ["ndcg", "recall", "hr"]
  k_list: [5, 10, 20]
  full_library: true           # 全库评估（无负采样）
  filter_train_items: true     # 过滤训练物品
```

## 🔬 模型使用示例

```python
from models.qmcsr_complete import QMCSRRecommender
import torch

# 创建模型
model = QMCSRRecommender(
    text_dim=768,
    image_dim=2048,
    item_embed_dim=64,
    num_items=12000,
    disentangled_dim=32,
    num_interests=4,
    hidden_dim=64,
    alpha_ortho=0.1,
    alpha_causal=0.05
)

# 准备数据
item_ids = torch.randint(1, 12000, (4, 10))  # (batch, seq_len)
text_features = torch.randn(4, 10, 768)
image_features = torch.randn(4, 10, 2048)
seq_lengths = torch.tensor([10, 8, 9, 7])
target_items = torch.randint(1, 12000, (4,))

# Phase 1: 不使用因果
outputs_phase1 = model(
    item_ids=item_ids,
    text_features=text_features,
    image_features=image_features,
    seq_lengths=seq_lengths,
    target_items=target_items,
    use_causal=False,
    return_loss=True
)

print(f"Loss: {outputs_phase1['loss'].item():.4f}")
print(f"Rec Loss: {outputs_phase1['rec_loss'].item():.4f}")
print(f"Ortho Loss: {outputs_phase1['ortho_loss'].item():.4f}")

# Phase 2: 使用因果
outputs_phase2 = model(
    item_ids=item_ids,
    text_features=text_features,
    image_features=image_features,
    seq_lengths=seq_lengths,
    target_items=target_items,
    use_causal=True,
    return_loss=True
)

print(f"Loss: {outputs_phase2['loss'].item():.4f}")
print(f"Causal Loss: {outputs_phase2['causal_loss'].item():.4f}")

# 预测
top_k_items, top_k_scores = model.predict(
    item_ids=item_ids,
    text_features=text_features,
    image_features=image_features,
    seq_lengths=seq_lengths,
    top_k=10,
    use_causal=True
)

print(f"Top-10 items shape: {top_k_items.shape}")  # (4, 10)
```

## 📝 设计原则

### 1. 简化优先，避免过拟合
- 使用更小的embedding维度（64 vs 128）
- 更少的兴趣数量（4 vs 8）
- 单层GRU（1 vs 2）
- 更少的负样本（5 vs 10）

### 2. 渐进式训练
- Phase 1先学习基础表示
- Phase 2再引入因果去偏
- 避免训练初期的不稳定

### 3. 正则化策略
- Dropout: 0.2
- Weight Decay: 0.0001
- 梯度裁剪: max_grad_norm=1.0
- 早停: patience=5

### 4. 损失权重平衡
- 主损失（BPR）为主导
- 辅助损失（正交、因果）权重较小
- 避免辅助损失干扰主任务

## 🔍 关键超参数

| 超参数 | 默认值 | 说明 |
|--------|--------|------|
| `disentangled_dim` | 32 | 每个解耦维度的大小 |
| `num_interests` | 4 | 用户兴趣数量 |
| `hidden_dim` | 64 | 序列编码器隐藏层维度 |
| `item_embed_dim` | 64 | 物品嵌入维度 |
| `alpha_ortho` | 0.1 | 正交约束权重 |
| `alpha_causal` | 0.05 | 因果损失权重 |
| `dropout` | 0.2 | Dropout比率 |

## 📌 注意事项

1. **数据集大小**:
   - 小数据集（<10k items）: 使用默认配置
   - 大数据集（>50k items）: 可适当增加模型容量

2. **训练时间**:
   - Beauty数据集（~12k items）: 约2-3小时（GPU）
   - 建议使用GPU加速

3. **内存占用**:
   - 模型参数: ~5 MB
   - 训练batch: ~2 GB（batch_size=64）

4. **收敛性**:
   - Phase 1通常5-10个epoch收敛
   - Phase 2需要10-20个epoch微调

## 📚 相关论文

如果使用本代码，请考虑引用：

```bibtex
@inproceedings{qmcsr2024,
  title={QMCSR: Quantum-Inspired Multi-Modal Causal Sequential Recommendation},
  author={Your Name},
  booktitle={Conference},
  year={2024}
}
```

## 🤝 贡献

欢迎提Issue和Pull Request！

## 📄 License

MIT License
