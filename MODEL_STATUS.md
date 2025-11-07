# 模型文件状态总结

## 代码库中的模型版本

### 主要模型（用于训练）

1. **qmcsr_complete.py** ⭐ **最新版本**
   - 用途: QMCSR模型（简化版，避免过拟合）
   - 训练脚本: `train_qmcsr.py`
   - candidate_items支持: ✅ **已修复**（刚刚修复）
   - 状态: **推荐使用**

2. **multimodal_recommender.py**
   - 用途: 完整的多模态推荐模型
   - 训练脚本: `train_amazon.py`, `train_simple_example.py`
   - candidate_items支持: ✅ **已支持**（之前就有）
   - 状态: 可用

### 辅助模块

3. **disentangled_representation.py**
   - 用途: 解耦表征学习模块
   - KL散度bug: ✅ **已修复**

4. **causal_inference.py**
   - 用途: 因果推断模块
   - 状态: 正常

5. **quantum_inspired_encoder.py**
   - 用途: 量子启发编码器
   - 状态: 正常

6. **quantum_encoder_enhanced.py**
   - 用途: 增强版量子编码器
   - 状态: 正常

### 其他模型

7. **baseline_multimodal.py**
   - 用途: 基线模型
   - 状态: 仅供对比

8. **progressive_recommender.py**
   - 用途: 渐进式训练模型
   - 状态: 实验性

---

## 训练脚本对应关系

| 训练脚本 | 使用的模型 | candidate_items支持 | 推荐使用 |
|---------|-----------|-------------------|---------|
| `train_qmcsr.py` | `qmcsr_complete.py` | ✅ 已修复 | ⭐ **推荐** |
| `train_amazon.py` | `multimodal_recommender.py` | ✅ 已支持 | 可用 |
| `train_simple_example.py` | `multimodal_recommender.py` | ✅ 已支持 | 示例 |

---

## 推荐使用的训练流程

### 首选: QMCSR模型 ⭐

```bash
python train_qmcsr.py --config config_qmcsr.yaml
```

**优势：**
- ✅ 简化架构，避免过拟合
- ✅ 参数量适中（1.2M）
- ✅ 已修复所有bug
- ✅ 配置已优化

**配置亮点：**
- num_negatives: 50（数据集提供）
- batch_size: 256
- weight_decay: 0.005
- temperature: 0.2（隐含在模型中）
- KL散度：已修复

### 备选: MultimodalRecommender模型

```bash
python train_amazon.py --config config_recommended.yaml
```

**优势：**
- ✅ 功能更完整
- ✅ 已支持candidate_items
- ⚠️ 参数更多，可能过拟合

---

## 已修复的Bug

### Bug 1: 负样本被浪费 ✅
- **问题**: 数据集准备的50个负样本没被使用
- **影响**: `qmcsr_complete.py` + `train_qmcsr.py`
- **修复**: 已修复（刚刚）

### Bug 2: KL散度计算错误 ✅
- **问题**: 未除以latent_dim
- **影响**: `disentangled_representation.py`
- **修复**: 已修复

### Bug 3: 损失权重失衡 ✅
- **问题**: 辅助损失太小
- **影响**: 所有配置文件
- **修复**: 已修复

### Bug 4: 其他超参数 ✅
- temperature: 0.5 → 0.2
- batch_size: 64 → 256
- weight_decay: 0.0001 → 0.005
- **修复**: 已修复

---

## 注意事项

1. **不要混淆模型版本**
   - 使用qmcsr → 用`train_qmcsr.py`
   - 使用multimodal → 用`train_amazon.py`

2. **确认数据集有candidate_items**
   - 检查: `'candidate_items' in batch`
   - 数量: 50个（config中设置）

3. **验证修复**
   ```bash
   python verify_negative_sampling_fix.py
   ```

---

## 当前推荐配置

**首选配置（已优化）：**
- 模型: `qmcsr_complete.py`
- 配置: `config_qmcsr.yaml`
- 训练脚本: `train_qmcsr.py`
- 数据集: Beauty (12K items)

**关键参数：**
```yaml
num_negatives: 50      # 数据集提供
batch_size: 256        # 稳定训练
weight_decay: 0.005    # 防止过拟合
alpha_ortho: 0.1       # 正交约束
alpha_causal: 0.05     # 因果损失
```

---

## 预期效果

修复后预期指标：
- NDCG@10: 0.04-0.07（vs 之前的0.01-0.02）
- Recall@20: 0.08-0.15
- 训练稳定性: 损失平稳下降

---

最后更新: 2025-11-07 14:15
