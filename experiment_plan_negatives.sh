#!/bin/bash
# 实验计划：测试不同负采样数量的影响
# 目标：找到训练效率和性能的最佳平衡点

echo "================================================================"
echo "负采样数量实验计划"
echo "================================================================"
echo ""
echo "实验设计："
echo "1. Phase 1: num_negatives=5   (快速验证，约30分钟)"
echo "2. Phase 2: num_negatives=50  (推荐baseline，约5小时)"
echo "3. Phase 3: num_negatives=100 (性能调优，约10小时)"
echo "4. Phase 4: num_negatives=200 (可选，探索上限，约20小时)"
echo ""
echo "================================================================"
echo ""

# 设置变量
CONFIG="config_qmcsr.yaml"
LOG_DIR="logs/negative_sampling_experiments"
mkdir -p $LOG_DIR

# ===== Experiment 1: 5 negative samples =====
echo "▶ Experiment 1: num_negatives=5 (快速验证)"
echo "  目的: 验证模型基本架构是否work"
echo "  预期时间: 30分钟"
echo ""
read -p "是否运行实验1？(y/n): " run_exp1
if [ "$run_exp1" = "y" ]; then
    # 修改配置
    sed -i 's/num_negatives: [0-9]*/num_negatives: 5/' $CONFIG

    # 训练
    echo "开始训练..."
    python train_qmcsr.py --config $CONFIG 2>&1 | tee $LOG_DIR/exp1_neg5.log

    echo "✓ 实验1完成，日志保存在: $LOG_DIR/exp1_neg5.log"
    echo ""
fi

# ===== Experiment 2: 50 negative samples =====
echo "▶ Experiment 2: num_negatives=50 (推荐baseline)"
echo "  目的: 平衡训练效率和性能"
echo "  预期时间: 5小时"
echo "  预期提升: 相比实验1，NDCG@10提升20-50%"
echo ""
read -p "是否运行实验2？(y/n): " run_exp2
if [ "$run_exp2" = "y" ]; then
    # 修改配置
    sed -i 's/num_negatives: [0-9]*/num_negatives: 50/' $CONFIG

    # 训练
    echo "开始训练..."
    python train_qmcsr.py --config $CONFIG 2>&1 | tee $LOG_DIR/exp2_neg50.log

    echo "✓ 实验2完成，日志保存在: $LOG_DIR/exp2_neg50.log"
    echo ""
fi

# ===== Experiment 3: 100 negative samples =====
echo "▶ Experiment 3: num_negatives=100 (性能调优)"
echo "  目的: 探索更多负样本是否进一步提升"
echo "  预期时间: 10小时"
echo "  决策依据: 如果实验2 NDCG@10 > 0.04，值得尝试"
echo ""
read -p "是否运行实验3？(y/n): " run_exp3
if [ "$run_exp3" = "y" ]; then
    # 修改配置
    sed -i 's/num_negatives: [0-9]*/num_negatives: 100/' $CONFIG

    # 训练
    echo "开始训练..."
    python train_qmcsr.py --config $CONFIG 2>&1 | tee $LOG_DIR/exp3_neg100.log

    echo "✓ 实验3完成，日志保存在: $LOG_DIR/exp3_neg100.log"
    echo ""
fi

# ===== Experiment 4: 200 negative samples (可选) =====
echo "▶ Experiment 4: num_negatives=200 (可选，探索上限)"
echo "  目的: 探索性能上限"
echo "  预期时间: 20小时"
echo "  决策依据: 如果实验3提升>5%，值得尝试"
echo ""
read -p "是否运行实验4？(y/n): " run_exp4
if [ "$run_exp4" = "y" ]; then
    # 修改配置
    sed -i 's/num_negatives: [0-9]*/num_negatives: 200/' $CONFIG

    # 训练
    echo "开始训练..."
    python train_qmcsr.py --config $CONFIG 2>&1 | tee $LOG_DIR/exp4_neg200.log

    echo "✓ 实验4完成，日志保存在: $LOG_DIR/exp4_neg200.log"
    echo ""
fi

# ===== 结果总结 =====
echo "================================================================"
echo "实验完成！"
echo "================================================================"
echo ""
echo "查看结果："
echo "  grep 'NDCG@10' $LOG_DIR/*.log"
echo ""
echo "决策建议："
echo "  1. 如果 neg5 vs neg50 提升 > 30%  → 使用50作为baseline"
echo "  2. 如果 neg50 vs neg100 提升 > 10% → 使用100"
echo "  3. 如果 neg100 vs neg200 提升 < 5% → 边际收益递减，停在100"
echo ""
echo "记得查看tensorboard："
echo "  tensorboard --logdir=logs/qmcsr"
echo ""
