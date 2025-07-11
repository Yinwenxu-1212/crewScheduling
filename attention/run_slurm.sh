#!/bin/bash
#SBATCH --job-name=CTS_optimized_ep2
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
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
srun python run_training.py