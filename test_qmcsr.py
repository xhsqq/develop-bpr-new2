"""
Simple test script for QMCSR Model
验证模型是否可以正常前向传播
"""

import torch
import yaml
from models.qmcsr_complete import (
    QMCSRRecommender,
    AestheticEmotionalDisentanglement,
    QuantumMultiInterestEncoder,
    CausalDebiasing
)


def test_disentanglement():
    """测试解耦模块"""
    print("\n" + "=" * 50)
    print("Testing Aesthetic-Emotional Disentanglement")
    print("=" * 50)

    batch_size = 4
    seq_len = 10
    text_dim = 768
    image_dim = 2048
    latent_dim = 32

    model = AestheticEmotionalDisentanglement(
        text_dim=text_dim,
        image_dim=image_dim,
        latent_dim=latent_dim
    )

    # 创建假数据
    text_features = torch.randn(batch_size, seq_len, text_dim)
    image_features = torch.randn(batch_size, seq_len, image_dim)

    # 前向传播
    outputs = model(text_features, image_features, return_loss=True)

    print(f"✓ h_disentangled shape: {outputs['h_disentangled'].shape}")
    print(f"✓ h_aes shape: {outputs['h_aes'].shape}")
    print(f"✓ h_emo shape: {outputs['h_emo'].shape}")
    print(f"✓ ortho_loss: {outputs['ortho_loss'].item():.4f}")

    assert outputs['h_disentangled'].shape == (batch_size, seq_len, latent_dim * 2)
    print("✓ Disentanglement module test passed!")


def test_quantum_encoder():
    """测试量子编码器"""
    print("\n" + "=" * 50)
    print("Testing Quantum Multi-Interest Encoder")
    print("=" * 50)

    batch_size = 4
    input_dim = 64
    num_interests = 4
    output_dim = 128

    model = QuantumMultiInterestEncoder(
        input_dim=input_dim,
        num_interests=num_interests,
        output_dim=output_dim
    )

    # 创建假数据
    x = torch.randn(batch_size, input_dim)

    # 前向传播
    outputs = model(x, return_all_interests=True)

    print(f"✓ output shape: {outputs['output'].shape}")
    print(f"✓ amplitudes shape: {outputs['amplitudes'].shape}")
    print(f"✓ phases shape: {outputs['phases'].shape}")
    print(f"✓ real_parts shape: {outputs['real_parts'].shape}")
    print(f"✓ imag_parts shape: {outputs['imag_parts'].shape}")

    assert outputs['output'].shape == (batch_size, output_dim)
    assert outputs['amplitudes'].shape == (batch_size, num_interests)
    print("✓ Quantum encoder test passed!")


def test_causal_debiasing():
    """测试因果去偏模块"""
    print("\n" + "=" * 50)
    print("Testing Causal Debiasing")
    print("=" * 50)

    batch_size = 4
    seq_repr_dim = 128
    disentangled_dim = 32

    model = CausalDebiasing(
        seq_repr_dim=seq_repr_dim,
        disentangled_dim=disentangled_dim,
        num_dimensions=2
    )

    # 创建假数据
    h_seq = torch.randn(batch_size, seq_repr_dim)
    h_aes = torch.randn(batch_size, disentangled_dim)
    h_emo = torch.randn(batch_size, disentangled_dim)

    # 前向传播
    outputs = model(h_seq, h_aes, h_emo)

    print(f"✓ h_debiased shape: {outputs['h_debiased'].shape}")
    print(f"✓ h_aes_cf shape: {outputs['h_aes_cf'].shape}")
    print(f"✓ h_emo_cf shape: {outputs['h_emo_cf'].shape}")
    print(f"✓ ite shape: {outputs['ite'].shape}")

    assert outputs['h_debiased'].shape == (batch_size, seq_repr_dim)
    assert outputs['ite'].shape == (batch_size, 2)
    print("✓ Causal debiasing test passed!")


def test_complete_model():
    """测试完整模型"""
    print("\n" + "=" * 50)
    print("Testing Complete QMCSR Model")
    print("=" * 50)

    batch_size = 4
    seq_len = 10
    text_dim = 768
    image_dim = 2048
    item_embed_dim = 64
    num_items = 1000

    model = QMCSRRecommender(
        text_dim=text_dim,
        image_dim=image_dim,
        item_embed_dim=item_embed_dim,
        num_items=num_items,
        disentangled_dim=32,
        num_interests=4,
        hidden_dim=64
    )

    # 创建假数据
    item_ids = torch.randint(1, num_items, (batch_size, seq_len))
    text_features = torch.randn(batch_size, seq_len, text_dim)
    image_features = torch.randn(batch_size, seq_len, image_dim)
    seq_lengths = torch.randint(5, seq_len + 1, (batch_size,))
    target_items = torch.randint(1, num_items, (batch_size,))

    # Phase 1: 不使用因果
    print("\nPhase 1: Without Causal Debiasing")
    outputs_phase1 = model(
        item_ids=item_ids,
        text_features=text_features,
        image_features=image_features,
        seq_lengths=seq_lengths,
        target_items=target_items,
        use_causal=False,
        return_loss=True
    )

    print(f"✓ logits shape: {outputs_phase1['logits'].shape}")
    print(f"✓ loss: {outputs_phase1['loss'].item():.4f}")
    print(f"✓ rec_loss: {outputs_phase1['rec_loss'].item():.4f}")
    print(f"✓ ortho_loss: {outputs_phase1['ortho_loss'].item():.4f}")

    # Phase 2: 使用因果
    print("\nPhase 2: With Causal Debiasing")
    outputs_phase2 = model(
        item_ids=item_ids,
        text_features=text_features,
        image_features=image_features,
        seq_lengths=seq_lengths,
        target_items=target_items,
        use_causal=True,
        return_loss=True
    )

    print(f"✓ logits shape: {outputs_phase2['logits'].shape}")
    print(f"✓ loss: {outputs_phase2['loss'].item():.4f}")
    print(f"✓ rec_loss: {outputs_phase2['rec_loss'].item():.4f}")
    print(f"✓ ortho_loss: {outputs_phase2['ortho_loss'].item():.4f}")
    print(f"✓ causal_loss: {outputs_phase2['causal_loss'].item():.4f}")

    # 测试预测
    print("\nTesting Prediction")
    top_k_items, top_k_scores = model.predict(
        item_ids=item_ids,
        text_features=text_features,
        image_features=image_features,
        seq_lengths=seq_lengths,
        top_k=10,
        use_causal=True
    )

    print(f"✓ top_k_items shape: {top_k_items.shape}")
    print(f"✓ top_k_scores shape: {top_k_scores.shape}")

    assert outputs_phase1['logits'].shape == (batch_size, num_items + 1)
    assert top_k_items.shape == (batch_size, 10)
    print("✓ Complete model test passed!")


def test_parameter_count():
    """测试参数量"""
    print("\n" + "=" * 50)
    print("Testing Parameter Count")
    print("=" * 50)

    model = QMCSRRecommender(
        text_dim=768,
        image_dim=2048,
        item_embed_dim=64,
        num_items=10000,
        disentangled_dim=32,
        num_interests=4,
        hidden_dim=64
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    print(f"✓ Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")

    # 打印各个模块的参数量
    print("\nParameter breakdown:")
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {module_params:,} parameters")


def test_with_config():
    """使用配置文件测试"""
    print("\n" + "=" * 50)
    print("Testing with Config File")
    print("=" * 50)

    try:
        with open('config_qmcsr.yaml', 'r') as f:
            config = yaml.safe_load(f)

        model = QMCSRRecommender(
            text_dim=config['model']['text_dim'],
            image_dim=config['model']['image_dim'],
            item_embed_dim=config['model']['item_embed_dim'],
            num_items=10000,
            disentangled_dim=config['model']['disentangled_dim'],
            num_interests=config['model']['num_interests'],
            hidden_dim=config['model']['hidden_dim'],
            max_seq_length=config['data']['max_seq_length'],
            alpha_ortho=config['loss']['alpha_ortho'],
            alpha_causal=config['loss']['alpha_causal'],
            dropout=config['model']['dropout']
        )

        print(f"✓ Model created successfully with config")
        print(f"✓ Config loaded from: config_qmcsr.yaml")

        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Total parameters: {total_params:,}")

    except FileNotFoundError:
        print("⚠ Config file not found, skipping config test")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" " * 15 + "QMCSR Model Test Suite")
    print("=" * 60)

    try:
        # 测试各个模块
        test_disentanglement()
        test_quantum_encoder()
        test_causal_debiasing()

        # 测试完整模型
        test_complete_model()

        # 测试参数量
        test_parameter_count()

        # 测试配置文件
        test_with_config()

        print("\n" + "=" * 60)
        print(" " * 20 + "All Tests Passed! ✓")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(" " * 20 + "Test Failed! ✗")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
