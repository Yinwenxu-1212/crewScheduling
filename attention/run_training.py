#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服务器训练启动脚本
用于在服务器上启动大规模训练
"""

import os
import sys
import time
import argparse
from datetime import datetime

def setup_environment():
    """设置训练环境"""
    # 设置CUDA相关环境变量
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 用于调试CUDA错误
    os.environ['TORCH_USE_CUDA_DSA'] = '1'    # 启用CUDA设备端断言
    
    # 创建必要的目录
    directories = ['models', 'log', 'plots', 'debug']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"确保目录存在: {directory}")

def check_system_requirements():
    """检查系统要求"""
    import torch
    
    print("=== 系统检查 ===")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu_props.name}, 内存: {gpu_props.total_memory / 1024**3:.2f}GB")
    else:
        print("警告: CUDA不可用，将使用CPU训练（速度较慢）")
    
    print("==================\n")

def run_training_with_monitoring():
    """运行训练并监控"""
    start_time = datetime.now()
    print(f"训练开始时间: {start_time}")
    
    try:
        # 导入并运行训练
        from main import train
        
        print("开始大规模训练...")
        model, monitor, analyzer = train()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n=== 训练完成 ===")
        print(f"训练结束时间: {end_time}")
        print(f"总训练时长: {duration}")
        
        # 保存训练摘要
        training_summary = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'final_stats': monitor.get_current_stats() if monitor else None
        }
        
        import json
        with open('log/training_summary.json', 'w', encoding='utf-8') as f:
            json.dump(training_summary, f, indent=2, ensure_ascii=False)
        
        print("训练摘要已保存到 log/training_summary.json")
        
        return True
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        return False
    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='服务器训练启动脚本')
    parser.add_argument('--check-only', action='store_true', help='只检查系统要求，不开始训练')
    parser.add_argument('--episodes', type=int, help='覆盖配置中的训练轮数')
    parser.add_argument('--lr', type=float, help='覆盖配置中的学习率')
    
    args = parser.parse_args()
    
    print("=== 机组排班强化学习 - 服务器训练 ===")
    print("双头架构 + 注意力机制 + 对偶价格特征")
    print("================================================\n")
    
    # 设置环境
    setup_environment()
    
    # 检查系统要求
    check_system_requirements()
    
    if args.check_only:
        print("系统检查完成，退出。")
        return
    
    # 更新配置（如果提供了参数）
    if args.episodes or args.lr:
        import config
        if args.episodes:
            config.NUM_EPISODES = args.episodes
            print(f"训练轮数设置为: {args.episodes}")
        if args.lr:
            config.LEARNING_RATE = args.lr
            print(f"学习率设置为: {args.lr}")
    
    # 运行训练
    success = run_training_with_monitoring()
    
    if success:
        print("\n训练成功完成！")
        print("可以查看以下文件:")
        print("- models/best_model.pth: 最佳模型")
        print("- log/final_training_log.json: 训练日志")
        print("- plots/final_training_progress.png: 训练进度图")
        print("- rosterResult.csv: 最终排班结果")
    else:
        print("\n训练未能正常完成，请检查错误信息。")
        sys.exit(1)

if __name__ == '__main__':
    main()