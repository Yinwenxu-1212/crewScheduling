#!/bin/bash
#SBATCH --job-name=CTS_Fast5_MultiGPU
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:a800:2
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=2151102@tongji.edu.cn
#SBATCH --output=log/fast5_%j.out
#SBATCH --error=log/fast5_%j.err

# 加载必要的模块
# module load python/3.8  # 注释掉不存在的模块
module load cuda/11.8

# 设置多GPU CUDA环境变量
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.0;8.6"
export FORCE_CUDA="1"

echo "=== GPU和CUDA环境信息 ==="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA_HOME: $CUDA_HOME"
nvidia-smi
echo "=============================="

# 创建必要的目录
mkdir -p models_fast5
mkdir -p output
mkdir -p log
mkdir -p plots
mkdir -p debug

# 检查PyTorch CUDA支持
echo "检查PyTorch CUDA支持..."
python check_cuda.py

if [ $? -ne 0 ]; then
    echo "PyTorch CUDA问题检测到，执行紧急修复..."
    chmod +x emergency_fix.sh
    ./emergency_fix.sh
    exit 0  # 紧急修复脚本会自动开始训练
else
    echo "PyTorch CUDA支持正常，开始快速5轮训练..."
fi

# 运行训练（直接使用修改后的config.py进行5轮训练）
echo "开始快速5轮多GPU训练..."
srun python run_training.py --episodes 5

# 训练完成后的清理工作
echo "快速5轮多GPU训练完成！"
echo "模型保存路径: models_fast5/"
echo "日志文件: log/fast5_${SLURM_JOB_ID}.out"
echo "错误日志: log/fast5_${SLURM_JOB_ID}.err"

# 显示GPU使用情况
echo "=== 最终GPU状态 ==="
nvidia-smi
echo "====================="

# 可选：压缩日志文件
# tar -czf logs_fast5_${SLURM_JOB_ID}.tar.gz log/

echo "快速5轮多GPU训练作业完成！"