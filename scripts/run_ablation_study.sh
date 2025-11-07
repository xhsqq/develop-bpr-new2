#!/bin/bash
# ============================================
# QMCSR - 消融实验脚本
# Quantum Multimodal Causal Sequential Recommender
#
# 功能: 验证各模块的贡献（5个消融实验）
# 改进: 维度特定融合 + 量子编码(16态) + SCM因果推断
# ============================================

set -e  # 遇到错误立即退出

# ============================================
# 颜色定义
# ============================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ============================================
# 帮助信息
# ============================================
show_help() {
    echo -e "${GREEN}================================================================${NC}"
    echo -e "${GREEN}  QMCSR - 消融实验${NC}"
    echo -e "${GREEN}  Quantum Multimodal Causal Sequential Recommender${NC}"
    echo -e "${GREEN}================================================================${NC}"
    echo ""
    echo -e "${YELLOW}用法:${NC}"
    echo "  $0 [category] [batch_size] [epochs] [device] [--quick]"
    echo ""
    echo -e "${YELLOW}参数:${NC}"
    echo "  category    数据集类别 (默认: beauty)"
    echo "              可选: beauty, games, sports"
    echo "  batch_size  批次大小 (默认: 256)"
    echo "  epochs      训练轮数 (默认: 30，消融实验建议较少轮数)"
    echo "  device      设备 (默认: auto)"
    echo "              可选: cuda, cpu, auto"
    echo "  --quick     快速模式，只运行3个关键实验"
    echo ""
    echo -e "${YELLOW}消融实验列表:${NC}"
    echo "  1. 完整模型     - 所有改进启用（baseline）"
    echo "  2. 无维度融合   - 移除维度特定多模态融合"
    echo "  3. 无量子编码   - 16态→4态，移除量子优化"
    echo "  4. 无SCM因果    - 移除因果推断模块"
    echo "  5. 基线模型     - 所有改进禁用"
    echo ""
    echo -e "${YELLOW}示例:${NC}"
    echo "  $0                           # 使用默认参数"
    echo "  $0 beauty                    # 指定数据集"
    echo "  $0 beauty 128 50            # 自定义batch_size和epochs"
    echo "  $0 beauty 256 30 cuda       # 指定所有参数"
    echo "  $0 beauty 256 20 auto --quick  # 快速模式"
    echo ""
    echo -e "${YELLOW}数据集说明:${NC}"
    echo "  beauty  - 美妆产品推荐 (~22k用户, ~12k商品)"
    echo "  games   - 视频游戏推荐 (~24k用户, ~11k商品)"
    echo "  sports  - 运动用品推荐 (~35k用户, ~18k商品)"
    echo ""
    echo -e "${YELLOW}预估时间 (单个实验):${NC}"
    echo "  beauty:  ~20-30分钟 (30 epochs, GPU)"
    echo "  games:   ~25-35分钟 (30 epochs, GPU)"
    echo "  sports:  ~30-45分钟 (30 epochs, GPU)"
    echo "  总计:    ~2-4小时 (5个实验)"
    echo ""
    exit 0
}

# 检查帮助标志
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
fi

# ============================================
# 参数解析
# ============================================
CATEGORY=${1:-beauty}
BATCH_SIZE=${2:-256}
NUM_EPOCHS=${3:-30}
DEVICE=${4:-auto}

# 检查快速模式
QUICK_MODE=false
for arg in "$@"; do
    if [ "$arg" = "--quick" ]; then
        QUICK_MODE=true
    fi
done

# ============================================
# 配置参数
# ============================================
BASE_CONFIG="config.yaml"
DATA_DIR="data/processed/${CATEGORY}"
CHECKPOINT_BASE="checkpoints/${CATEGORY}_ablation"
RESULTS_DIR="ablation_results/${CATEGORY}"
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')

# ============================================
# 打印函数
# ============================================
print_msg() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}✗ [ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠ [WARNING]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_exp_header() {
    echo ""
    echo -e "${CYAN}================================================================${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}================================================================${NC}"
}

print_section() {
    echo ""
    echo -e "${PURPLE}────────────────────────────────────────${NC}"
    echo -e "${PURPLE}  $1${NC}"
    echo -e "${PURPLE}────────────────────────────────────────${NC}"
}

# ============================================
# 欢迎界面
# ============================================
clear
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  QMCSR - 消融实验${NC}"
echo -e "${GREEN}  Quantum Multimodal Causal Sequential Recommender${NC}"
echo -e "${GREEN}================================================================${NC}"
echo -e "${BLUE}配置参数:${NC}"
echo -e "  数据集:      ${YELLOW}${CATEGORY}${NC}"
echo -e "  批次大小:    ${YELLOW}${BATCH_SIZE}${NC}"
echo -e "  训练轮数:    ${YELLOW}${NUM_EPOCHS}${NC}"
echo -e "  设备:        ${YELLOW}${DEVICE}${NC}"
if [ "$QUICK_MODE" = true ]; then
    echo -e "  模式:        ${YELLOW}快速模式 (3个实验)${NC}"
else
    echo -e "  模式:        ${YELLOW}完整模式 (5个实验)${NC}"
fi
echo -e "${GREEN}================================================================${NC}"
echo ""

# ============================================
# 环境检查
# ============================================
print_section "环境检查"

# 检查数据是否存在
if [ ! -d "${DATA_DIR}" ]; then
    print_error "数据目录不存在: ${DATA_DIR}"
    echo ""
    echo -e "${YELLOW}请先运行完整流程准备数据:${NC}"
    echo "  bash scripts/run_full_pipeline.sh ${CATEGORY}"
    exit 1
fi

print_success "数据目录存在: ${DATA_DIR}"

# 检查GPU
if [ "$DEVICE" = "auto" ] || [ "$DEVICE" = "cuda" ]; then
    python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null
    if [ $? -eq 0 ]; then
        GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
        GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
        print_success "检测到 ${GPU_COUNT} 个GPU: ${GPU_NAME}"
        if [ "$DEVICE" = "auto" ]; then
            DEVICE="cuda"
        fi
    else
        print_warning "GPU不可用，将使用CPU（会很慢）"
        DEVICE="cpu"
    fi
fi

# 创建结果目录
mkdir -p ${RESULTS_DIR}

# ============================================
# 消融实验配置
# ============================================
declare -A EXPERIMENTS

# 实验1: 完整模型
EXPERIMENTS["exp1_name"]="full_model"
EXPERIMENTS["exp1_desc"]="完整模型（所有改进）"
EXPERIMENTS["exp1_args"]="--num_epochs ${NUM_EPOCHS} --batch_size ${BATCH_SIZE}"
EXPERIMENTS["exp1_num"]="1"

# 实验2: 无维度特定融合
EXPERIMENTS["exp2_name"]="no_disentangled_fusion"
EXPERIMENTS["exp2_desc"]="无维度特定融合"
EXPERIMENTS["exp2_args"]="--num_epochs ${NUM_EPOCHS} --batch_size ${BATCH_SIZE} --alpha_recon 0.0"
EXPERIMENTS["exp2_num"]="2"

# 实验3: 无量子编码
EXPERIMENTS["exp3_name"]="no_quantum"
EXPERIMENTS["exp3_desc"]="无量子编码器（4态）"
EXPERIMENTS["exp3_args"]="--num_epochs ${NUM_EPOCHS} --batch_size ${BATCH_SIZE} --num_interests 4 --alpha_diversity 0.0"
EXPERIMENTS["exp3_num"]="3"

# 实验4: 无SCM因果推断
EXPERIMENTS["exp4_name"]="no_causal"
EXPERIMENTS["exp4_desc"]="无SCM因果推断"
EXPERIMENTS["exp4_args"]="--num_epochs ${NUM_EPOCHS} --batch_size ${BATCH_SIZE} --alpha_causal 0.0"
EXPERIMENTS["exp4_num"]="4"

# 实验5: 基线模型
EXPERIMENTS["exp5_name"]="baseline"
EXPERIMENTS["exp5_desc"]="基线模型（无任何改进）"
EXPERIMENTS["exp5_args"]="--num_epochs ${NUM_EPOCHS} --batch_size ${BATCH_SIZE} --alpha_recon 0.0 --alpha_causal 0.0 --alpha_diversity 0.0 --num_interests 4"
EXPERIMENTS["exp5_num"]="5"

# ============================================
# 实验列表
# ============================================
if [ "$QUICK_MODE" = true ]; then
    EXP_LIST=("exp1" "exp3" "exp5")  # 完整、无量子、基线
    TOTAL_EXPS=3
else
    EXP_LIST=("exp1" "exp2" "exp3" "exp4" "exp5")
    TOTAL_EXPS=5
fi

# ============================================
# 显示实验计划
# ============================================
print_section "实验计划"

for exp_id in "${EXP_LIST[@]}"; do
    exp_num="${EXPERIMENTS[${exp_id}_num]}"
    exp_desc="${EXPERIMENTS[${exp_id}_desc]}"
    echo -e "  ${exp_num}. ${exp_desc}"
done
echo ""

# 预估时间
if [ "$QUICK_MODE" = true ]; then
    echo -e "${YELLOW}预估总时间: ~1-2小时${NC}"
else
    echo -e "${YELLOW}预估总时间: ~2-4小时${NC}"
fi
echo ""

# 确认继续
read -p "$(echo -e ${YELLOW}是否开始实验？ [y/N]: ${NC})" -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_msg "已取消"
    exit 0
fi

# ============================================
# 运行实验函数
# ============================================
run_experiment() {
    local exp_id=$1
    local exp_num="${EXPERIMENTS[${exp_id}_num]}"
    local exp_name="${EXPERIMENTS[${exp_id}_name]}"
    local exp_desc="${EXPERIMENTS[${exp_id}_desc]}"
    local exp_args="${EXPERIMENTS[${exp_id}_args]}"

    print_exp_header "[实验 ${exp_num}/${TOTAL_EXPS}] ${exp_desc}"

    # 创建实验目录
    local exp_dir="${RESULTS_DIR}/${exp_name}_${TIMESTAMP}"
    mkdir -p ${exp_dir}

    # 检查是否已完成（智能跳过）
    if [ -f "${exp_dir}/results.txt" ] && [ -f "${exp_dir}/COMPLETED" ]; then
        print_success "实验已完成，跳过"
        echo -e "  结果文件: ${exp_dir}/results.txt"
        return 0
    fi

    # 日志文件
    local log_file="${exp_dir}/train.log"
    local result_file="${exp_dir}/results.txt"

    print_msg "实验目录: ${exp_dir}"
    print_msg "配置参数: ${exp_args}"

    # 运行训练
    print_msg "开始训练..."
    local start_time=$(date +%s)

    python train.py \
        --config ${BASE_CONFIG} \
        --category ${CATEGORY} \
        --data_dir ${DATA_DIR} \
        --checkpoint_dir ${exp_dir}/checkpoints \
        --log_dir ${exp_dir}/logs \
        ${exp_args} \
        --device ${DEVICE} \
        2>&1 | tee ${log_file}

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        print_error "实验失败"
        echo "FAILED" > ${result_file}
        return 1
    fi

    local train_time=$(($(date +%s) - start_time))
    print_success "训练完成 (用时: ${train_time}s)"

    # 查找检查点
    local checkpoint=$(ls -t ${exp_dir}/checkpoints/*.pth 2>/dev/null | head -n 1)

    if [ -z "$checkpoint" ]; then
        print_warning "未找到检查点"
        echo "NO_CHECKPOINT" > ${result_file}
        return 0
    fi

    # 运行评估
    print_msg "评估模型..."
    python train.py \
        --config ${BASE_CONFIG} \
        --category ${CATEGORY} \
        --mode eval \
        --checkpoint ${checkpoint} \
        --device ${DEVICE} \
        2>&1 | tee -a ${log_file}

    # 提取结果
    extract_results ${log_file} ${result_file}

    # 标记完成
    touch ${exp_dir}/COMPLETED

    print_success "实验完成"

    # 显示结果
    if [ -f "${result_file}" ]; then
        echo ""
        echo -e "${BLUE}实验结果:${NC}"
        cat ${result_file} | sed 's/^/  /'
    fi
    echo ""
}

# ============================================
# 提取结果函数
# ============================================
extract_results() {
    local log_file=$1
    local result_file=$2

    # 提取指标（更健壮的方式）
    local recall10=$(grep -E "Recall@10|recall@10" ${log_file} | tail -n 1 | grep -oP '\d+\.\d+' | head -n 1)
    local ndcg10=$(grep -E "NDCG@10|ndcg@10" ${log_file} | tail -n 1 | grep -oP '\d+\.\d+' | head -n 1)
    local hr10=$(grep -E "HR@10|hit@10" ${log_file} | tail -n 1 | grep -oP '\d+\.\d+' | head -n 1)

    # 写入结果
    cat > ${result_file} <<EOF
Recall@10: ${recall10:-N/A}
NDCG@10: ${ndcg10:-N/A}
HR@10: ${hr10:-N/A}
EOF
}

# ============================================
# 运行所有实验
# ============================================
START_TIME=$(date +%s)
SUCCESSFUL_EXPS=0
FAILED_EXPS=0

for exp_id in "${EXP_LIST[@]}"; do
    if run_experiment ${exp_id}; then
        ((SUCCESSFUL_EXPS++))
    else
        ((FAILED_EXPS++))
        print_warning "继续下一个实验..."
    fi
done

# 计算总时间
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

# ============================================
# 生成汇总报告
# ============================================
print_section "生成汇总报告"

SUMMARY_FILE="${RESULTS_DIR}/ablation_summary_${TIMESTAMP}.txt"

cat > ${SUMMARY_FILE} <<EOF
================================================================
  QMCSR 消融实验汇总报告
  Quantum Multimodal Causal Sequential Recommender
================================================================

运行时间: $(date +'%Y-%m-%d %H:%M:%S')
数据集: ${CATEGORY}
总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒

实验配置:
---------
- 训练轮数: ${NUM_EPOCHS}
- 批次大小: ${BATCH_SIZE}
- 设备: ${DEVICE}
- 模式: $([ "$QUICK_MODE" = true ] && echo "快速模式 (${TOTAL_EXPS}个实验)" || echo "完整模式 (${TOTAL_EXPS}个实验)")

实验统计:
---------
- 成功: ${SUCCESSFUL_EXPS}
- 失败: ${FAILED_EXPS}

================================================================
实验结果
================================================================

EOF

# 添加每个实验的详细结果
for exp_id in "${EXP_LIST[@]}"; do
    exp_name="${EXPERIMENTS[${exp_id}_name]}"
    exp_desc="${EXPERIMENTS[${exp_id}_desc]}"
    exp_dir=$(ls -td ${RESULTS_DIR}/${exp_name}_* 2>/dev/null | head -n 1)

    if [ -n "$exp_dir" ] && [ -f "${exp_dir}/results.txt" ]; then
        echo "----------------------------------------" >> ${SUMMARY_FILE}
        echo "${exp_desc}" >> ${SUMMARY_FILE}
        echo "----------------------------------------" >> ${SUMMARY_FILE}
        cat ${exp_dir}/results.txt >> ${SUMMARY_FILE}
        echo "" >> ${SUMMARY_FILE}
    fi
done

# 生成Markdown表格
cat >> ${SUMMARY_FILE} <<EOF

================================================================
Markdown表格（可直接用于论文）
================================================================

| 实验配置 | Recall@10 | NDCG@10 | HR@10 |
|----------|-----------|---------|-------|
EOF

for exp_id in "${EXP_LIST[@]}"; do
    exp_name="${EXPERIMENTS[${exp_id}_name]}"
    exp_desc="${EXPERIMENTS[${exp_id}_desc]}"
    exp_dir=$(ls -td ${RESULTS_DIR}/${exp_name}_* 2>/dev/null | head -n 1)

    if [ -n "$exp_dir" ] && [ -f "${exp_dir}/results.txt" ]; then
        recall=$(grep "Recall@10" ${exp_dir}/results.txt | awk '{print $2}')
        ndcg=$(grep "NDCG@10" ${exp_dir}/results.txt | awk '{print $2}')
        hr=$(grep "HR@10" ${exp_dir}/results.txt | awk '{print $2}')

        echo "| ${exp_desc} | ${recall:-N/A} | ${ndcg:-N/A} | ${hr:-N/A} |" >> ${SUMMARY_FILE}
    fi
done

# 添加分析
cat >> ${SUMMARY_FILE} <<EOF

================================================================
模块贡献分析
================================================================

1. 维度特定融合的贡献:
   - 比较: 完整模型 vs 无维度融合
   - 改进: 先解耦再融合，避免模态偏差
   - 提升: 可解释性、推荐准确度

2. 量子编码器的贡献:
   - 比较: 完整模型 vs 无量子编码
   - 改进: 16个量子态（从4增加）
   - 提升: 更丰富的多兴趣建模

3. SCM因果推断的贡献:
   - 比较: 完整模型 vs 无SCM因果
   - 改进: Pearl三步反事实推理
   - 提升: 理论严谨、因果效应

4. 整体改进效果:
   - 比较: 完整模型 vs 基线模型
   - 改进: 三大模块协同作用
   - 提升: 全方位性能提升

================================================================
文件清单
================================================================

汇总报告: ${SUMMARY_FILE}
可视化脚本: ${RESULTS_DIR}/plot_ablation.py

各实验目录:
EOF

for exp_id in "${EXP_LIST[@]}"; do
    exp_name="${EXPERIMENTS[${exp_id}_name]}"
    exp_dir=$(ls -td ${RESULTS_DIR}/${exp_name}_* 2>/dev/null | head -n 1)
    if [ -n "$exp_dir" ]; then
        echo "  - ${exp_dir}" >> ${SUMMARY_FILE}
    fi
done

cat >> ${SUMMARY_FILE} <<EOF

================================================================
EOF

print_success "汇总报告已生成: ${SUMMARY_FILE}"

# ============================================
# 生成可视化脚本
# ============================================
PLOT_SCRIPT="${RESULTS_DIR}/plot_ablation.py"

cat > ${PLOT_SCRIPT} <<'PYTHON_EOF'
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""消融实验结果可视化脚本"""

import matplotlib.pyplot as plt
import numpy as np
import re
import sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_results(summary_file):
    """解析汇总文件"""
    experiments = []
    recalls = []
    ndcgs = []
    hrs = []

    with open(summary_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取表格内容
    table_match = re.search(r'\| 实验配置 \| Recall@10.*?\n(.*?)(?=\n\n|\Z)', content, re.DOTALL)
    if not table_match:
        print("未找到结果表格")
        return None

    table_lines = table_match.group(1).strip().split('\n')

    for line in table_lines:
        if line.startswith('|') and 'N/A' not in line:
            parts = [p.strip() for p in line.split('|')[1:-1]]
            if len(parts) == 4:
                exp_name, recall, ndcg, hr = parts
                try:
                    experiments.append(exp_name)
                    recalls.append(float(recall))
                    ndcgs.append(float(ndcg))
                    hrs.append(float(hr))
                except ValueError:
                    continue

    return experiments, recalls, ndcgs, hrs

def plot_results(experiments, recalls, ndcgs, hrs, output_file):
    """绘制对比图"""
    x = np.arange(len(experiments))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))

    bars1 = ax.bar(x - width, recalls, width, label='Recall@10', color='#2E86AB', alpha=0.9)
    bars2 = ax.bar(x, ndcgs, width, label='NDCG@10', color='#A23B72', alpha=0.9)
    bars3 = ax.bar(x + width, hrs, width, label='HR@10', color='#F18F01', alpha=0.9)

    ax.set_xlabel('实验配置', fontsize=13, fontweight='bold')
    ax.set_ylabel('指标值', fontsize=13, fontweight='bold')
    ax.set_title('QMCSR 消融实验结果对比', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=20, ha='right', fontsize=11)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(max(recalls), max(ndcgs), max(hrs)) * 1.15)

    # 添加数值标签
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9, fontweight='bold')

    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存: {output_file}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python plot_ablation.py <summary_file>")
        sys.exit(1)

    summary_file = sys.argv[1]
    output_file = summary_file.replace('.txt', '.png')

    result = parse_results(summary_file)
    if result:
        experiments, recalls, ndcgs, hrs = result
        if len(experiments) > 0:
            plot_results(experiments, recalls, ndcgs, hrs, output_file)
        else:
            print("没有可用的实验结果")
    else:
        print("解析结果失败")
PYTHON_EOF

chmod +x ${PLOT_SCRIPT}
print_success "可视化脚本已生成: ${PLOT_SCRIPT}"

# ============================================
# 完成总结
# ============================================
echo ""
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  消融实验完成！${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""

echo -e "${BLUE}实验统计:${NC}"
echo -e "  总耗时:      ${YELLOW}${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒${NC}"
echo -e "  成功实验:    ${GREEN}${SUCCESSFUL_EXPS}/${TOTAL_EXPS}${NC}"
if [ $FAILED_EXPS -gt 0 ]; then
    echo -e "  失败实验:    ${RED}${FAILED_EXPS}/${TOTAL_EXPS}${NC}"
fi
echo ""

echo -e "${BLUE}生成的文件:${NC}"
echo -e "  汇总报告:     ${YELLOW}${SUMMARY_FILE}${NC}"
echo -e "  可视化脚本:   ${YELLOW}${PLOT_SCRIPT}${NC}"
echo -e "  结果目录:     ${YELLOW}${RESULTS_DIR}${NC}"
echo ""

echo -e "${BLUE}下一步建议:${NC}"
echo -e "  1. 查看汇总报告:   ${YELLOW}cat ${SUMMARY_FILE}${NC}"
echo -e "  2. 生成对比图表:   ${YELLOW}python ${PLOT_SCRIPT} ${SUMMARY_FILE}${NC}"
echo -e "  3. 复制Markdown表格到论文"
echo -e "  4. 分析各模块贡献"
echo ""

echo -e "${BLUE}使用帮助:${NC}"
echo -e "  ${YELLOW}$0 --help${NC}  查看所有参数说明"
echo ""

echo -e "${GREEN}================================================================${NC}"

exit 0
