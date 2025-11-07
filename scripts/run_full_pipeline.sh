#!/bin/bash
# ============================================
# QMCSR - 完整训练流程
# Quantum Multimodal Causal Sequential Recommender
#
# 功能: 下载数据 -> 预处理 -> 提取特征 -> 训练 -> 评估
# 改进: 维度特定融合 + 量子编码(16态) + SCM因果推断
# ============================================

set -e  # 遇到错误立即退出

# 检查并使用 demo 环境的 Python
if [ -f "/opt/anaconda3/envs/demo/bin/python" ]; then
    PYTHON_CMD="/opt/anaconda3/envs/demo/bin/python"
    echo "使用 demo 环境: ${PYTHON_CMD}"
else
    PYTHON_CMD="python"
    echo "使用系统 Python: ${PYTHON_CMD}"
fi

# ============================================
# 颜色定义
# ============================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# ============================================
# 帮助信息
# ============================================
show_help() {
    echo -e "${GREEN}================================================================${NC}"
    echo -e "${GREEN}  QMCSR - 完整训练流程${NC}"
    echo -e "${GREEN}  Quantum Multimodal Causal Sequential Recommender${NC}"
    echo -e "${GREEN}================================================================${NC}"
    echo ""
    echo -e "${YELLOW}用法:${NC}"
    echo "  $0 [category] [batch_size] [epochs] [device]"
    echo ""
    echo -e "${YELLOW}参数:${NC}"
    echo "  category    数据集类别 (默认: beauty)"
    echo "              可选: beauty, games, sports, all"
    echo "  batch_size  批次大小 (默认: 256)"
    echo "  epochs      训练轮数 (默认: 50)"
    echo "  device      设备 (默认: auto)"
    echo "              可选: cuda, cpu, auto"
    echo ""
    echo -e "${YELLOW}示例:${NC}"
    echo "  $0                           # 使用默认参数"
    echo "  $0 beauty                    # 指定数据集"
    echo "  $0 beauty 128 100           # 指定category, batch_size, epochs"
    echo "  $0 all 256 50 cuda          # 处理所有数据集"
    echo ""
    echo -e "${YELLOW}数据集说明:${NC}"
    echo "  beauty  - 美妆产品推荐 (~22k用户, ~12k商品)"
    echo "  games   - 视频游戏推荐 (~24k用户, ~11k商品)"
    echo "  sports  - 运动用品推荐 (~35k用户, ~18k商品)"
    echo "  all     - 依次处理以上三个数据集"
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
EPOCHS=${3:-50}
DEVICE=${4:-auto}

# ============================================
# 配置参数
# ============================================
DATA_DIR="data"
RAW_DIR="${DATA_DIR}/raw"
PROCESSED_DIR="${DATA_DIR}/processed"
CONFIG_FILE="config.yaml"
CHECKPOINT_BASE="checkpoints"
REPORT_DIR="reports"

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

print_step() {
    echo ""
    echo -e "${PURPLE}================================================================${NC}"
    echo -e "${PURPLE}  $1${NC}"
    echo -e "${PURPLE}================================================================${NC}"
}

# ============================================
# 欢迎界面
# ============================================
clear
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  QMCSR - 完整训练流程${NC}"
echo -e "${GREEN}  Quantum Multimodal Causal Sequential Recommender${NC}"
echo -e "${GREEN}================================================================${NC}"
echo -e "${BLUE}配置参数:${NC}"
echo -e "  数据集:      ${YELLOW}${CATEGORY}${NC}"
echo -e "  批次大小:    ${YELLOW}${BATCH_SIZE}${NC}"
echo -e "  训练轮数:    ${YELLOW}${EPOCHS}${NC}"
echo -e "  设备:        ${YELLOW}${DEVICE}${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""

# ============================================
# 确定要处理的数据集列表
# ============================================
if [ "${CATEGORY}" = "all" ]; then
    CATEGORIES=("beauty" "games" "sports")
    print_msg "将依次处理 3 个数据集"
else
    CATEGORIES=("${CATEGORY}")
    print_msg "将处理 1 个数据集"
fi

# ============================================
# Step 1: 数据下载
# ============================================
print_step "[Step 1/5] 数据下载"

mkdir -p ${RAW_DIR}
mkdir -p ${PROCESSED_DIR}

for CAT in "${CATEGORIES[@]}"; do
    print_msg "检查数据集: ${CAT}"

    REVIEWS_FILE="${RAW_DIR}/${CAT}_reviews.json"
    META_FILE="${RAW_DIR}/${CAT}_meta.json"

    if [ -f "$REVIEWS_FILE" ] && [ -f "$META_FILE" ]; then
        print_success "${CAT} 原始数据已存在，跳过下载"
    else
        print_msg "下载 ${CAT} 数据集..."
        ${PYTHON_CMD} data/download_amazon.py --category ${CAT} --data_dir ${RAW_DIR}

        if [ $? -ne 0 ]; then
            print_error "${CAT} 数据下载失败"
            exit 1
        fi
        print_success "${CAT} 数据下载完成"
    fi
done

# ============================================
# Step 2: 数据预处理
# ============================================
print_step "[Step 2/5] 数据预处理"

for CAT in "${CATEGORIES[@]}"; do
    print_msg "检查预处理: ${CAT}"

    TRAIN_SEQ="${PROCESSED_DIR}/${CAT}/train_sequences.pkl"

    if [ -f "$TRAIN_SEQ" ]; then
        print_success "${CAT} 预处理数据已存在，跳过此步骤"
    else
        print_msg "预处理 ${CAT} 数据集..."
        ${PYTHON_CMD} data/preprocess_amazon.py \
            --category ${CAT} \
            --raw_dir ${RAW_DIR} \
            --processed_dir ${PROCESSED_DIR} \
            --min_interactions 5 \
            --max_seq_length 50

        if [ $? -ne 0 ]; then
            print_error "${CAT} 数据预处理失败"
            exit 1
        fi
        print_success "${CAT} 数据预处理完成"
    fi
done

# ============================================
# Step 3: 特征提取
# ============================================
print_step "[Step 3/5] 多模态特征提取"

for CAT in "${CATEGORIES[@]}"; do
    print_msg "处理数据集: ${CAT}"

    # 文本特征（BERT）
    TEXT_FEAT="${PROCESSED_DIR}/${CAT}/text_features.pkl"
    if [ -f "$TEXT_FEAT" ]; then
        print_success "${CAT} 文本特征已存在，跳过提取"
    else
        print_msg "提取 ${CAT} BERT文本特征 (768维)..."

        if [ "$DEVICE" = "cuda" ]; then
            ${PYTHON_CMD} scripts/extract_text_features.py \
                --category ${CAT} \
                --data_dir ${PROCESSED_DIR} \
                --model_name "bert-base-uncased" \
                --batch_size 64 \
                --device cuda
        else
            ${PYTHON_CMD} scripts/extract_text_features.py \
                --category ${CAT} \
                --data_dir ${PROCESSED_DIR} \
                --model_name "bert-base-uncased" \
                --batch_size 32 \
                --device cpu
        fi

        if [ $? -ne 0 ]; then
            print_error "${CAT} 文本特征提取失败"
            echo "请检查是否已安装 transformers: pip install transformers"
            exit 1
        fi
        print_success "${CAT} 文本特征提取完成"
    fi

    # 图像特征（ResNet）
    IMAGE_FEAT="${PROCESSED_DIR}/${CAT}/image_features.pkl"
    if [ -f "$IMAGE_FEAT" ]; then
        print_success "${CAT} 图像特征已存在，跳过提取"
        print_msg "  如需重新提取真实特征，请删除: ${IMAGE_FEAT}"
    else
        print_msg "提取 ${CAT} ResNet图像特征 (2048维)..."
        print_warning "将从Amazon URL下载真实图片（可能需要5-30分钟）"

        if [ "$DEVICE" = "cuda" ]; then
            ${PYTHON_CMD} scripts/extract_image_features.py \
                --category ${CAT} \
                --data_dir ${PROCESSED_DIR} \
                --model_name "resnet50" \
                --batch_size 32 \
                --device cuda \
                --timeout 10 \
                --num_workers 16
        else
            ${PYTHON_CMD} scripts/extract_image_features.py \
                --category ${CAT} \
                --data_dir ${PROCESSED_DIR} \
                --model_name "resnet50" \
                --batch_size 16 \
                --device cpu \
                --timeout 10 \
                --num_workers 8
        fi

        if [ $? -ne 0 ]; then
            print_error "${CAT} 图像特征提取失败"
            echo "请检查是否已安装: pip install torchvision pillow requests"
            exit 1
        fi
        print_success "${CAT} 图像特征提取完成"
    fi
done

print_success "所有特征提取完成"

# ============================================
# Step 4: 快速测试（可选）
# ============================================
print_step "[Step 4/5] 快速测试"

if [ -f "scripts/quick_test.py" ]; then
    print_msg "运行快速测试验证数据和模型..."
    ${PYTHON_CMD} scripts/quick_test.py
    if [ $? -eq 0 ]; then
        print_success "快速测试通过"
    else
        print_warning "快速测试失败，但将继续训练"
    fi
else
    print_warning "未找到 scripts/quick_test.py，跳过测试"
fi

# ============================================
# Step 5: 模型训练
# ============================================
print_step "[Step 5/5] 模型训练"

print_msg "模型架构:"
echo -e "  ${GREEN}✓${NC} 维度特定多模态融合 (先解耦再融合)"
echo -e "  ${GREEN}✓${NC} 量子编码器 (16个量子态)"
echo -e "  ${GREEN}✓${NC} SCM因果推断 (Pearl三步法)"
echo ""

for CAT in "${CATEGORIES[@]}"; do
    print_msg "训练数据集: ${CAT}"

    CHECKPOINT_DIR="${CHECKPOINT_BASE}/${CAT}"
    mkdir -p ${CHECKPOINT_DIR}

    # 构建训练命令
    TRAIN_CMD="${PYTHON_CMD} train.py --config ${CONFIG_FILE} --category ${CAT}"

    # 只在用户明确指定时才覆盖配置
    if [ $# -ge 2 ]; then
        TRAIN_CMD="${TRAIN_CMD} --batch_size ${BATCH_SIZE}"
    fi
    if [ $# -ge 3 ]; then
        TRAIN_CMD="${TRAIN_CMD} --epochs ${EPOCHS}"
    fi

    TRAIN_CMD="${TRAIN_CMD} --device ${DEVICE}"
    TRAIN_CMD="${TRAIN_CMD} --save_dir ${CHECKPOINT_DIR}"

    print_msg "执行命令: ${TRAIN_CMD}"
    eval ${TRAIN_CMD}

    if [ $? -ne 0 ]; then
        print_error "${CAT} 训练失败"
        if [ "${CATEGORY}" = "all" ]; then
            print_warning "继续处理下一个数据集..."
            continue
        else
            exit 1
        fi
    fi

    print_success "${CAT} 训练完成"

    # 显示结果（如果存在）
    if [ -f "${CHECKPOINT_DIR}/results.json" ]; then
        echo ""
        echo -e "${BLUE}${CAT} 训练结果:${NC}"
        cat ${CHECKPOINT_DIR}/results.json | python -m json.tool 2>/dev/null || cat ${CHECKPOINT_DIR}/results.json
        echo ""
    fi
done

# ============================================
# 生成报告
# ============================================
mkdir -p ${REPORT_DIR}
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
REPORT_FILE="${REPORT_DIR}/experiment_${CATEGORY}_${TIMESTAMP}.txt"

cat > ${REPORT_FILE} <<EOF
================================================================
  QMCSR 实验报告
  Quantum Multimodal Causal Sequential Recommender
================================================================

运行时间: $(date +'%Y-%m-%d %H:%M:%S')
数据集: ${CATEGORY}
处理的数据集: ${CATEGORIES[@]}

配置参数:
---------
- 批次大小: ${BATCH_SIZE}
- 训练轮数: ${EPOCHS}
- 设备: ${DEVICE}
- 配置文件: ${CONFIG_FILE}

架构改进:
---------
1. ✓ 维度特定多模态融合
   - 先解耦再融合
   - 每个模态独立解耦为功能/美学/情感三维度
   - 维度内跨模态注意力融合
   - 避免模态偏差，可解释性提升

2. ✓ 量子编码器优化 (16个量子态)
   - 相位编码: |ψ⟩ = A * e^{iφ}
   - 幺正干涉: U = (I+iA)(I-iA)^{-1}
   - 严格的量子测量 (Born规则)
   - 量子度量: Purity, Entanglement, Fidelity

3. ✓ SCM因果推断 (Pearl三步法)
   - Abduction: 推断外生变量 ε = (z-μ)/σ
   - Action: 干预操作
   - Prediction: 反事实预测与ITE计算

数据集统计:
---------
EOF

for CAT in "${CATEGORIES[@]}"; do
    cat >> ${REPORT_FILE} <<EOF

### ${CAT} 数据集
- 处理目录: ${PROCESSED_DIR}/${CAT}
- 文本特征: 768维 (BERT-base-uncased)
- 图像特征: 2048维 (ResNet50)
- 检查点目录: ${CHECKPOINT_BASE}/${CAT}
EOF

    STATS_FILE="${PROCESSED_DIR}/${CAT}/statistics.json"
    if [ -f "${STATS_FILE}" ]; then
        echo "" >> ${REPORT_FILE}
        cat ${STATS_FILE} >> ${REPORT_FILE}
    fi

    RESULT_FILE="${CHECKPOINT_BASE}/${CAT}/results.json"
    if [ -f "${RESULT_FILE}" ]; then
        echo "" >> ${REPORT_FILE}
        echo "训练结果:" >> ${REPORT_FILE}
        cat ${RESULT_FILE} >> ${REPORT_FILE}
    fi
done

cat >> ${REPORT_FILE} <<EOF

================================================================
EOF

# ============================================
# 完成总结
# ============================================
echo ""
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  训练完成！${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""

echo -e "${BLUE}处理摘要:${NC}"
echo -e "  ${GREEN}✓${NC} 环境检查"
echo -e "  ${GREEN}✓${NC} 数据下载"
echo -e "  ${GREEN}✓${NC} 数据预处理"
echo -e "  ${GREEN}✓${NC} 特征提取 (BERT 768维 + ResNet 2048维)"
echo -e "  ${GREEN}✓${NC} 模型训练"
echo ""

echo -e "${BLUE}处理的数据集:${NC}"
for CAT in "${CATEGORIES[@]}"; do
    echo -e "  - ${YELLOW}${CAT}${NC}"
done
echo ""

echo -e "${BLUE}生成的文件:${NC}"
for CAT in "${CATEGORIES[@]}"; do
    echo -e "  ${CAT}:"
    echo -e "    数据: ${PROCESSED_DIR}/${CAT}/"
    echo -e "      - train_sequences.pkl, val_sequences.pkl, test_sequences.pkl"
    echo -e "      - text_features.pkl (BERT 768维)"
    echo -e "      - image_features.pkl (ResNet 2048维)"
    echo -e "      - statistics.json"
    echo -e "    模型: ${CHECKPOINT_BASE}/${CAT}/"
    echo -e "      - best_model.pt"
    echo -e "      - results.json"
done
echo ""

echo -e "${BLUE}实验报告:${NC}"
echo -e "  ${REPORT_FILE}"
echo ""

echo -e "${BLUE}下一步建议:${NC}"
echo -e "  - 查看TensorBoard:  ${YELLOW}tensorboard --logdir runs${NC}"
echo -e "  - 运行消融实验:     ${YELLOW}bash scripts/run_ablation_study.sh ${CATEGORY}${NC}"
echo -e "  - 查看数据统计:     ${YELLOW}cat ${PROCESSED_DIR}/${CATEGORIES[0]}/statistics.json${NC}"
if [ -f "${CHECKPOINT_BASE}/${CATEGORIES[0]}/results.json" ]; then
    echo -e "  - 查看训练结果:     ${YELLOW}cat ${CHECKPOINT_BASE}/${CATEGORIES[0]}/results.json${NC}"
fi
echo ""

echo -e "${BLUE}使用帮助:${NC}"
echo -e "  ${YELLOW}$0 --help${NC}  查看所有参数说明"
echo ""

echo -e "${GREEN}================================================================${NC}"

exit 0
