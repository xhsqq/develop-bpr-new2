# 🎯 渐进式推荐器使用指南

## 概述

这是一个**渐进式多模态推荐器**，通过配置文件开关来逐步启用功能，方便进行消融实验。

## 核心优势

✅ **一份代码，四种模型**：通过配置开关，无需修改代码  
✅ **真正的渐进式**：每个stage都基于前一个stage构建  
✅ **完整的训练流程**：包含训练、验证、早停、保存最佳模型  
✅ **清晰的指标对比**：自动记录每个stage的效果

## 四个实验Stage

| Stage | 配置 | 说明 | 预期HR@10 |
|-------|------|------|-----------|
| **Stage 0** | 全关闭 | Baseline（简单concat + GRU） | 0.04-0.08 |
| **Stage 1** | `enable_disentangled=true` | + 解耦模块 | > Stage 0 |
| **Stage 2** | + `enable_quantum=true` | + 量子编码器 | > Stage 1 |
| **Stage 3** | + `enable_causal=true` | 完整模型 | 最高 |

## 快速开始

### 1. 运行 Stage 0 (Baseline)

修改 `config_progressive.yaml`：

```yaml
stage:
  enable_disentangled: false
  enable_quantum: false
  enable_causal: false

experiment:
  name: "progressive-stage-0"
  tags:
    - "baseline"
```

运行训练：

```bash
python train_progressive.py --config config_progressive.yaml
```

### 2. 运行 Stage 1 (+解耦)

修改配置：

```yaml
stage:
  enable_disentangled: true
  enable_quantum: false
  enable_causal: false

experiment:
  name: "progressive-stage-1"
  tags:
    - "stage-1-disentangled"
```

运行训练：

```bash
python train_progressive.py --config config_progressive.yaml
python train_progressive.py --config config_progressive_small.yaml
```

### 3. 运行 Stage 2 (+量子)

```yaml
stage:
  enable_disentangled: true
  enable_quantum: true
  enable_causal: false

experiment:
  name: "progressive-stage-2"
```

### 4. 运行 Stage 3 (完整)

```yaml
stage:
  enable_disentangled: true
  enable_quantum: true
  enable_causal: true

experiment:
  name: "progressive-stage-3"
```

## 训练输出示例

```
================================================================================
🎯 渐进式推荐器配置
================================================================================
解耦模块: ✅
量子编码器: ❌
因果推断: ❌
================================================================================

Loading data...
✓ Train: 131413 samples
✓ Valid: 21850 samples
✓ Test: 22156 samples
✓ Items: 12042

🔥 Epoch 1/50: [████████████████] loss: 0.6543, avg: 0.6543
📊 Evaluating: [████████████████] HR@10: 0.0523, NDCG@10: 0.0287

================================================================================
📊 Epoch 1/50 Results
================================================================================
  🔥 Train Loss:    0.6543
     ├─ BPR:        0.6543
     ├─ Recon:      0.0234
     ├─ Diversity:  0.0000
     └─ Causal:     0.0000
  ✅ Valid HR@10:   0.0523
  ✅ Valid NDCG@10: 0.0287
  ✅ Valid MRR:     0.0156
  🌟 Best model saved (NDCG@10: 0.0287)
================================================================================
```

## 结果保存位置

```
checkpoints/
├── progressive-stage-0/
│   ├── best_model.pth          # 最佳模型
│   └── final_results.yaml      # 最终结果
├── progressive-stage-1/
│   ├── best_model.pth
│   └── final_results.yaml
├── progressive-stage-2/
│   └── ...
└── progressive-stage-3/
    └── ...
```

## 对比结果

训练完所有stage后，查看结果：

```bash
cat checkpoints/progressive-stage-0/final_results.yaml
cat checkpoints/progressive-stage-1/final_results.yaml
cat checkpoints/progressive-stage-2/final_results.yaml
cat checkpoints/progressive-stage-3/final_results.yaml
```

## 关键配置项说明

### 损失权重 (loss)

```yaml
loss:
  alpha_recon: 0.01       # 解耦重构损失权重
  alpha_causal: 0.005     # 因果损失权重  
  alpha_diversity: 0.001  # 量子多样性损失权重
  beta: 0.05              # KL散度权重
```

⚠️ **注意**：只有对应功能启用时，这些权重才会生效。

### 训练参数 (training)

```yaml
training:
  batch_size: 64
  num_epochs: 50
  learning_rate: 0.001
  
  early_stopping:
    patience: 10          # 10个epoch没提升就停止
    min_delta: 0.0005     # 最小提升阈值
```

### 模型参数 (model)

```yaml
model:
  item_embed_dim: 128     # Item embedding维度
  hidden_dim: 256         # 隐层维度
  
  # 解耦配置
  disentangled_dim: 64    # 每个解耦维度大小
  num_disentangled_dims: 3  # 解耦维度数量
  
  # 量子配置
  num_interests: 4        # 用户兴趣数量
  quantum_state_dim: 128  # 量子状态维度
  
  # 因果配置
  num_ensembles: 3        # 集成模型数量
  num_mc_samples: 10      # 蒙特卡洛采样数
```

## 常见问题

### Q1: Stage 0 的结果太低怎么办？

**A**: 先确保 Stage 0 (Baseline) 的结果合理（HR@10 ≥ 0.04）：

- 检查数据是否正确加载
- 检查是否过滤了冷启动物品
- 尝试调整学习率 (0.0005 - 0.002)
- 增加训练epoch数

### Q2: Stage 1 比 Stage 0 效果差？

**A**: 可能原因：

1. **损失权重不合适**：尝试降低 `alpha_recon` (0.001 - 0.01)
2. **重构损失太大**：检查训练日志，如果 Recon Loss 很大，说明多模态融合困难
3. **需要更长训练**：解耦模块需要更多epoch才能收敛

### Q3: 如何加速训练？

```yaml
data:
  batch_size: 128        # 增大batch size
  num_workers: 8         # 增加数据加载线程

device:
  use_gpu: true
  mixed_precision: true  # 启用混合精度（需要GPU）
```

### Q4: 如何只评估不训练？

暂时不支持，可以修改 `train_progressive.py`，在开头加载保存的模型后直接evaluate。

## 实验建议

### 推荐的实验顺序

1. **先跑 Stage 0**，确保baseline正常（HR@10 ≥ 0.04）
2. **再跑 Stage 1**，观察解耦是否有增益
3. **如果 Stage 1 有增益，继续 Stage 2**
4. **如果 Stage 2 有增益，继续 Stage 3**

### 如果中间某个stage没有增益

- **不要放弃**！尝试调参：
  - 降低辅助损失权重
  - 增加训练epoch
  - 调整模型维度
- **做消融实验**：
  - 只打开一个功能
  - 观察单独贡献

## 与原始模型的对比

| 方面 | 原始MultimodalRecommender | ProgressiveMultimodalRecommender |
|------|--------------------------|----------------------------------|
| **可配置性** | 固定结构，难以修改 | 完全配置化，改配置即可 |
| **实验友好** | 需要修改代码做消融 | 自动支持消融实验 |
| **代码复杂度** | 843行，功能耦合 | 445行，模块清晰 |
| **训练脚本** | 需要适配 | 配套完整训练脚本 |
| **功能** | 所有功能一次性使用 | 渐进式逐步启用 |

## 代码结构

```
models/
  progressive_recommender.py    # 渐进式模型（445行）
  
config_progressive.yaml         # 配置文件
train_progressive.py           # 训练脚本（380行）

checkpoints/
  progressive-stage-{0,1,2,3}/  # 每个stage的结果
```

## 总结

✅ **简单**：只需要修改配置文件，不用改代码  
✅ **清晰**：每个stage的作用一目了然  
✅ **高效**：自动化训练、评估、保存  
✅ **科学**：渐进式实验，容易找到问题

---

**开始你的渐进式实验之旅吧！** 🚀

