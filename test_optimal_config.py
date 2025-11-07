"""
测试最优配置的参数量
验证针对13万样本的参数设计是否合理
"""
import torch
import yaml
from models.multimodal_recommender import MultimodalRecommender


def count_parameters(model):
    """统计模型参数"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 按模块统计
    module_params = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        module_params[name] = params
    
    return total, trainable, module_params


def test_config(config_path, config_name):
    """测试指定配置"""
    print("\n" + "=" * 80)
    print(f"📊 测试配置: {config_name}")
    print("=" * 80)
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 模拟数据规模 (Beauty数据集)
    num_items = 12042
    num_train_samples = 131413
    
    print(f"数据规模: {num_train_samples:,} 训练样本, {num_items:,} 物品\n")
    
    # 读取模型配置
    model_config = config['model']
    loss_config = config['loss']
    advanced_config = config.get('advanced', {})
    
    # 模态维度
    modality_dims = model_config['modality_dims']
    
    # 投影层维度（如果有）
    modality_proj_dims = model_config.get('modality_proj_dims', None)
    
    # 因果损失权重
    causal_loss_weights = {'magnitude': loss_config.get('causal_weights', {}).get('magnitude', 1.0)}
    
    # 创建模型
    print("创建模型...")
    model = MultimodalRecommender(
        modality_dims=modality_dims,
        modality_proj_dims=modality_proj_dims,
        disentangled_dim=model_config['disentangled_dim'],
        num_disentangled_dims=model_config['num_disentangled_dims'],
        num_interests=model_config['num_interests'],
        quantum_state_dim=model_config['quantum_state_dim'],
        hidden_dim=model_config['hidden_dim'],
        item_embed_dim=model_config['item_embed_dim'],
        num_items=num_items,
        max_seq_length=config['data']['max_seq_length'],
        alpha_recon=loss_config['alpha_recon'],
        alpha_causal=loss_config['alpha_causal'],
        alpha_diversity=loss_config['alpha_diversity'],
        causal_loss_weights=causal_loss_weights,
        num_negatives=advanced_config.get('num_negatives', 100),
        use_quantum_computing=False,
        beta=loss_config.get('beta', 0.05),
        temperature=advanced_config.get('temperature', 0.5),
        num_mc_samples=advanced_config.get('num_mc_samples', 10),
        num_ensembles=advanced_config.get('num_ensembles', 3),
        target_ite=advanced_config.get('target_ite', 0.3),
        dropout=model_config.get('dropout', 0.2)
    )
    
    # 统计参数
    total_params, trainable_params, module_params = count_parameters(model)
    
    print(f"\n✅ 模型创建成功")
    print(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # 参数/样本比（修正计算）
    ratio = total_params / num_train_samples  # 参数/样本
    samples_per_param = num_train_samples / total_params  # 样本/参数
    
    print(f"\n📊 参数效率分析:")
    print(f"  参数/样本比: {ratio:.4f} (参数量 / 样本数)")
    print(f"  样本/参数比: {samples_per_param:.2f} (每个参数有多少样本)")
    
    # 评估（基于样本/参数比）
    if samples_per_param >= 100:
        status = "✅ 优秀"
        explanation = f"每个参数有{samples_per_param:.0f}个样本，充分训练"
    elif samples_per_param >= 50:
        status = "✅ 良好"
        explanation = f"每个参数有{samples_per_param:.0f}个样本，训练充足"
    elif samples_per_param >= 20:
        status = "⚠️  可接受"
        explanation = f"每个参数有{samples_per_param:.0f}个样本，需注意正则化"
    elif samples_per_param >= 10:
        status = "⚠️  偏小"
        explanation = f"每个参数有{samples_per_param:.0f}个样本，需强正则化"
    else:
        status = "🔴 过拟合风险"
        explanation = f"每个参数只有{samples_per_param:.1f}个样本，容易过拟合"
    
    print(f"  状态: {status} - {explanation}")
    
    # 模块参数分布
    print(f"\n📦 模块参数分布:")
    sorted_modules = sorted(module_params.items(), key=lambda x: x[1], reverse=True)
    for name, params in sorted_modules[:10]:  # 只显示前10个最大的模块
        percentage = params / total_params * 100
        print(f"  {name:30s}: {params:>10,} ({percentage:>5.1f}%)")
    
    # 显示关键配置
    print(f"\n🔧 关键配置:")
    if modality_proj_dims:
        text_proj = modality_proj_dims.get('text', modality_dims['text'])
        image_proj = modality_proj_dims.get('image', modality_dims['image'])
        print(f"  投影层: Text {modality_dims['text']}→{text_proj}, "
              f"Image {modality_dims['image']}→{image_proj}")
        
        # 计算投影层参数
        proj_params = modality_dims['text'] * text_proj + modality_dims['image'] * image_proj
        proj_percentage = (module_params.get('modality_projections', 0) / total_params * 100) if total_params > 0 else 0
        print(f"  投影层参数: {proj_params:,} ({proj_percentage:.1f}%)")
    else:
        print(f"  投影层: 未启用")
    
    print(f"  Item嵌入: {model_config['item_embed_dim']}维 "
          f"({num_items * model_config['item_embed_dim']:,}参数)")
    print(f"  解耦维度: {model_config['disentangled_dim']}维 × {model_config['num_disentangled_dims']}")
    print(f"  量子兴趣: {model_config['num_interests']}个 × {model_config['quantum_state_dim']}维")
    print(f"  隐藏层: {model_config['hidden_dim']}维")
    print(f"  Dropout: {model_config.get('dropout', 0.2)}")
    
    print("=" * 80)
    
    return {
        'total_params': total_params,
        'ratio': ratio,
        'samples_per_param': samples_per_param,
        'status': status
    }


def main():
    """测试所有配置"""
    print("\n" + "=" * 80)
    print("🎯 多模态推荐系统 - 参数配置对比分析")
    print("=" * 80)
    
    configs = [
        ('config_optimal.yaml', '最优配置 (新设计)'),
        ('config_balanced.yaml', '平衡配置'),
        ('config.yaml', '当前配置'),
    ]
    
    results = {}
    for config_path, config_name in configs:
        try:
            result = test_config(config_path, config_name)
            results[config_name] = result
        except FileNotFoundError:
            print(f"\n⚠️  配置文件未找到: {config_path}")
        except Exception as e:
            print(f"\n❌ 测试失败 ({config_path}): {e}")
            # import traceback
            # traceback.print_exc()
    
    # 对比总结
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("📊 配置对比总结")
        print("=" * 80)
        print(f"{'配置名称':<25} {'参数量(M)':<12} {'样本/参数':<12} {'状态':<20}")
        print("-" * 80)
        for name, result in results.items():
            print(f"{name:<25} {result['total_params']/1e6:<12.2f} "
                  f"{result['samples_per_param']:<12.1f} {result['status']:<20}")
        
        # 推荐
        print("\n💡 推荐配置:")
        best_config = max(results.items(), key=lambda x: x[1]['samples_per_param'])
        print(f"  → {best_config[0]}")
        print(f"    理由: 样本/参数比最高 ({best_config[1]['samples_per_param']:.1f})，训练最充分")
    
    print("\n" + "=" * 80)
    print("✅ 测试完成")
    print("=" * 80)


if __name__ == '__main__':
    main()

