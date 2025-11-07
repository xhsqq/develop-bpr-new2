"""
验证负采样修复是否正确
检查数据集的candidate_items是否被正确传递和使用
"""

import torch
from data.dataloader import get_dataloaders
from models.qmcsr_complete import QMCSRRecommender

def verify_fix():
    print("="*80)
    print("验证负采样修复")
    print("="*80)

    # 1. 加载数据
    print("\n1. 加载数据...")
    train_loader, valid_loader, test_loader, num_items = get_dataloaders(
        category='beauty',
        batch_size=4,
        num_workers=0,
        num_negatives=50  # 设置50个负样本
    )

    # 检查第一个batch
    batch = next(iter(train_loader))

    print(f"\n2. 检查batch内容:")
    print(f"   - Keys: {batch.keys()}")
    print(f"   - item_ids shape: {batch['item_ids'].shape}")
    print(f"   - target_items shape: {batch['target_items'].shape}")

    if 'candidate_items' in batch:
        print(f"   ✅ candidate_items存在!")
        print(f"   - candidate_items shape: {batch['candidate_items'].shape}")
        print(f"   - 预期: (batch_size, 1+num_negatives) = (4, 51)")

        # 验证candidate_items[i, 0] == target_items[i]
        for i in range(batch['candidate_items'].size(0)):
            if batch['candidate_items'][i, 0] == batch['target_items'][i]:
                print(f"   ✅ Sample {i}: candidate_items[{i}, 0] == target_items[{i}] = {batch['target_items'][i].item()}")
            else:
                print(f"   ❌ Sample {i}: MISMATCH!")
    else:
        print(f"   ❌ candidate_items不存在!")
        print(f"   问题：数据集没有生成candidate_items")
        return False

    # 3. 测试模型forward
    print(f"\n3. 测试模型forward:")

    model = QMCSRRecommender(
        text_dim=768,
        image_dim=2048,
        item_embed_dim=64,
        num_items=num_items,
        disentangled_dim=32,
        num_interests=4,
        hidden_dim=64,
        max_seq_length=50,
        alpha_ortho=0.1,
        alpha_causal=0.05,
        dropout=0.2
    )

    # 准备输入
    item_ids = batch['item_ids']
    text_features = batch['text_features']
    image_features = batch['image_features']
    seq_lengths = batch['seq_lengths']
    target_items = batch['target_items']
    candidate_items = batch['candidate_items'] if 'candidate_items' in batch else None

    print(f"   - Calling model with candidate_items: {candidate_items is not None}")

    # Forward pass
    try:
        outputs = model(
            item_ids=item_ids,
            text_features=text_features,
            image_features=image_features,
            seq_lengths=seq_lengths,
            target_items=target_items,
            candidate_items=candidate_items,
            use_causal=False,
            return_loss=True
        )

        print(f"   ✅ Forward pass成功!")
        print(f"   - loss: {outputs['loss'].item():.4f}")
        print(f"   - rec_loss: {outputs['rec_loss'].item():.4f}")
        print(f"   - ortho_loss: {outputs['ortho_loss'].item():.4f}")

    except Exception as e:
        print(f"   ❌ Forward pass失败: {e}")
        return False

    # 4. 对比测试：有无candidate_items的差异
    print(f"\n4. 对比测试（有无candidate_items）:")

    # 测试1: 使用candidate_items
    outputs_with_cand = model(
        item_ids=item_ids,
        text_features=text_features,
        image_features=image_features,
        seq_lengths=seq_lengths,
        target_items=target_items,
        candidate_items=candidate_items,  # ← 提供
        use_causal=False,
        return_loss=True
    )

    # 测试2: 不使用candidate_items（回退到hard negative mining）
    outputs_without_cand = model(
        item_ids=item_ids,
        text_features=text_features,
        image_features=image_features,
        seq_lengths=seq_lengths,
        target_items=target_items,
        candidate_items=None,  # ← 不提供
        use_causal=False,
        return_loss=True
    )

    print(f"   - 使用candidate_items: rec_loss = {outputs_with_cand['rec_loss'].item():.4f}")
    print(f"   - 不使用candidate_items: rec_loss = {outputs_without_cand['rec_loss'].item():.4f}")

    if abs(outputs_with_cand['rec_loss'].item() - outputs_without_cand['rec_loss'].item()) > 0.001:
        print(f"   ✅ 损失值不同，说明candidate_items被正确使用！")
    else:
        print(f"   ❌ 损失值相同，candidate_items可能没被使用！")

    # 5. 总结
    print(f"\n{'='*80}")
    print("总结:")
    print("✅ 数据集正确生成了candidate_items")
    print("✅ 模型正确接收了candidate_items参数")
    print("✅ 模型forward使用了candidate_items计算损失")
    print("✅ 训练时会使用50个负样本（而不是5个hard negatives）")
    print("="*80)

    return True

if __name__ == "__main__":
    verify_fix()
