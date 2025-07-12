#!/bin/bash
#SBATCH --job-name=CTS_Dynamic_ep2
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G          
#SBATCH --array=1    
#SBATCH --time=72:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=2151102@tongji.edu.cn
#SBATCH --output=log/%j.out    # 输出到 log 目录
#SBATCH --error=log/%j.err     # 错误日志到 log 目录

# 加载必要的模块
module load cuda/11.8

# 创建必要的目录
mkdir -p models
mkdir -p output
mkdir -p log                  

# 设置环境变量控制 PyTorch 内存使用
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 运行训练命令
# 基础配置：使用自动时间检测和默认参数
srun python main.py --mode train --auto-detect-time

# 可选的高级配置示例（注释掉，需要时可以启用）
# srun python main.py \
#     --mode train \
#     --episodes 1000 \
#     --lr 0.0003 \
#     --batch-size 128 \
#     --replay-size 50000 \
#     --exploration-ratio 0.4 \
#     --auto-detect-time

# 指定特定数据集和时间范围的示例（注释掉）
# srun python main.py \
#     --mode train \
#     --data-path ./custom_data \
#     --start-date 2025-06-01 \
#     --end-date 2025-06-30 \
#     --episodes 500

# 仅评估模式示例（注释掉）
# srun python main.py --mode evaluate --auto-detect-time