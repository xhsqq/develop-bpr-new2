"""
测试模型参数量和样本比例
"""
import torch
import yaml
from models.multimodal_recommender import MultimodalRecommender
from data.dataloader import get_dataloaders

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

def main():
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print("📊 加载数据统计训练样本数")
    print("=" * 80)
    
    # 加载数据
    train_loader, valid_loader, test_loader, dataset_info = get_dataloaders(
        category=config['data']['category'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        max_seq_length=config['data']['max_seq_length'],
        num_negatives=config['advanced']['num_negatives']
    )
    
    num_users = dataset_info['num_users']
    num_items = dataset_info['num_items']
    
    num_train_samples = len(train_loader.dataset)
    num_valid_samples = len(valid_loader.dataset)
    num_test_samples = len(test_loader.dataset)
    
    print(f"✅ 训练样本数: {num_train_samples:,}")
    print(f"✅ 验证样本数: {num_valid_samples:,}")
    print(f"✅ 测试样本数: {num_test_samples:,}")
    print(f"✅ 用户数: {num_users:,}")
    print(f"✅ 物品数: {num_items:,}")
    print()
    
    print("=" * 80)
    print("🔧 创建模型并统计参数量")
    print("=" * 80)
    
    # 创建模型（简化版，仅用于统计参数）
    model_config = config['model']
    loss_config = config['loss']
    
    # 因果损失权重
    causal_loss_weights = {
        'magnitude': loss_config['causal_weights']['magnitude']
    }
    
    model = MultimodalRecommender(
        modality_dims=model_config['modality_dims'],
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
        num_negatives=config['advanced']['num_negatives'],
        use_quantum_computing=False,
        beta=loss_config['beta'],
        temperature=config['advanced']['temperature'],
        num_mc_samples=config['advanced']['num_mc_samples'],
        num_ensembles=config['advanced']['num_ensembles'],
        target_ite=config['advanced']['target_ite']
    )
    
    total_params, trainable_params, module_params = count_parameters(model)
    
    print(f"\n📈 总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"📈 可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print()
    
    print("=" * 80)
    print("📦 各模块参数量分布")
    print("=" * 80)
    sorted_modules = sorted(module_params.items(), key=lambda x: x[1], reverse=True)
    for name, params in sorted_modules:
        percentage = params / total_params * 100
        print(f"  {name:30s}: {params:>10,} ({percentage:>5.1f}%)")
    print()
    
    print("=" * 80)
    print("⚖️ 参数量 vs 样本数 比例分析")
    print("=" * 80)
    ratio = total_params / num_train_samples
    print(f"  参数量/训练样本 = {total_params:,} / {num_train_samples:,} = {ratio:.2f}")
    print()
    
    # 参考标准
    print("  📚 业界参考标准：")
    print("  ✅ 优秀：每个参数 >= 10个样本 (ratio <= 0.1)")
    print("  ⚠️  可接受：每个参数 5-10个样本 (ratio = 0.1-0.2)")
    print("  🔴 过拟合风险：每个参数 < 5个样本 (ratio > 0.2)")
    print()
    
    if ratio <= 0.1:
        status = "✅ 优秀"
    elif ratio <= 0.2:
        status = "⚠️ 可接受"
    else:
        status = "🔴 过拟合风险"
    
    print(f"  当前状态: {status}")
    print()
    
    # 建议
    if ratio > 0.2:
        print("=" * 80)
        print("💡 优化建议")
        print("=" * 80)
        target_params = int(num_train_samples * 0.1)
        reduction_needed = total_params - target_params
        reduction_pct = reduction_needed / total_params * 100
        
        print(f"  🎯 目标参数量: {target_params:,} ({target_params/1e6:.2f}M)")
        print(f"  📉 需要减少: {reduction_needed:,} ({reduction_pct:.1f}%)")
        print()
        
        # 分析最大的模块
        top_module, top_params = sorted_modules[0]
        print(f"  ⭐ 最大模块: {top_module} ({top_params:,}, {top_params/total_params*100:.1f}%)")
        
        # 如果是item_embedding
        if top_module == 'item_embedding':
            current_dim = config['model']['item_embed_dim']
            target_dim = int(current_dim * (target_params / total_params) ** 0.5)
            target_dim = max(32, (target_dim // 16) * 16)  # 向下取整到16的倍数
            print(f"  💡 建议: item_embed_dim: {current_dim} → {target_dim}")
        
        print()
        print("  🔧 其他优化方向：")
        print("  1. 添加BERT/ResNet投影降维层（768→128, 2048→256）")
        print("  2. 进一步减小 hidden_dim（64→32）")
        print("  3. 减少 num_interests（3→2）")
        print()
    
    print("=" * 80)
    print("🔍 多模态特征维度分析")
    print("=" * 80)
    text_dim = config['model']['modality_dims']['text']
    image_dim = config['model']['modality_dims']['image']
    print(f"  Text (BERT):   {text_dim} 维")
    print(f"  Image (ResNet): {image_dim} 维")
    print(f"  总计:          {text_dim + image_dim} 维")
    print()
    
    # 计算多模态编码器的参数
    disentangled_params = module_params.get('disentangled_module', 0)
    print(f"  解耦模块参数量: {disentangled_params:,} ({disentangled_params/total_params*100:.1f}%)")
    
    # 估算如果加投影层能减少多少参数
    text_proj_dim = 128
    image_proj_dim = 256
    print()
    print("  💡 如果添加投影降维：")
    print(f"     Text:  {text_dim} → {text_proj_dim}")
    print(f"     Image: {image_dim} → {image_proj_dim}")
    print(f"     总维度: {text_dim + image_dim} → {text_proj_dim + image_proj_dim}")
    print(f"     维度减少: {((text_dim + image_dim) - (text_proj_dim + image_proj_dim)) / (text_dim + image_dim) * 100:.1f}%")
    
    # 投影层新增参数
    proj_params = text_dim * text_proj_dim + image_dim * image_proj_dim
    print(f"     新增投影层参数: {proj_params:,}")
    
    # 解耦模块节省参数（粗略估算：输入维度减少，编码器参数大约减少相同比例）
    dim_ratio = (text_proj_dim + image_proj_dim) / (text_dim + image_dim)
    estimated_savings = disentangled_params * (1 - dim_ratio) - proj_params
    
    if estimated_savings > 0:
        print(f"     预计净节省参数: {estimated_savings:,} ({estimated_savings/total_params*100:.1f}%)")
        new_total = total_params - estimated_savings
        new_ratio = new_total / num_train_samples
        print(f"     优化后参数量: {new_total:,} ({new_total/1e6:.2f}M)")
        print(f"     优化后比例: {new_ratio:.2f}")
    else:
        print(f"     ⚠️ 投影层可能不会减少参数（新增 > 节省）")
    
    print("=" * 80)

if __name__ == '__main__':
    main()

