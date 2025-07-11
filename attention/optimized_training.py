# optimized_run_training.py
"""
优化的训练脚本，集成所有性能优化
"""
import os
import sys
import time
import torch
import numpy as np
from datetime import datetime
import multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.cuda.amp import autocast, GradScaler

# 导入优化模块
from precompute_features import PrecomputeManager
from optimized_environment import OptimizedCrewRosteringEnv
from optimized_reward_function import OptimizedRewardFunction
from optimized_unified_config import OptimizedUnifiedConfig
from flight_cycle_constraints import FlightCycleConstraints

def setup_optimized_environment():
    """设置优化环境"""
    # 设置多线程
    torch.set_num_threads(mp.cpu_count())
    
    # 启用cuDNN基准测试
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
    # 创建缓存目录
    os.makedirs('cache', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('log', exist_ok=True)
    
def run_optimized_training():
    """运行优化的训练"""
    print("=== 优化的机组排班训练 ===")
    print(f"开始时间: {datetime.now()}")
    
    # 1. 数据预处理和预计算
    print("\n1. 数据预处理...")
    from utils import DataHandler
    data_handler = DataHandler()
    
    # 预计算特征
    print("2. 预计算特征...")
    precompute_manager = PrecomputeManager(data_handler)
    if not precompute_manager.load_cache():
        precompute_manager.precompute_all_features()
        
    # 3. 创建优化环境
    print("3. 创建优化环境...")
    env = OptimizedCrewRosteringEnv(data_handler)
    reward_function = OptimizedRewardFunction(data_handler, precompute_manager)
    cycle_constraints = FlightCycleConstraints(data_handler)
    
    # 4. 初始化模型（使用混合精度训练）
    print("4. 初始化模型...")
    from model import ActorCritic
    import config
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ActorCritic(config.STATE_DIM, config.ACTION_DIM, config.HIDDEN_DIM).to(device)
    
    # 使用DataParallel如果有多GPU
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU")
        model = DataParallel(model)
        
    # 5. 优化器设置
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    # 混合精度训练
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    # 6. 训练循环优化
    print("\n开始训练...")
    best_coverage = 0.0
    episodes_without_improvement = 0
    
    for episode in range(config.NUM_EPISODES):
        episode_start = time.time()
        
        # 重置环境
        obs, info = env.reset()
        done = False
        
        # Episode统计
        total_reward = 0
        covered_flights = set()
        cycle_violations = 0
        steps = 0
        
        # 批量收集经验
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_values = []
        
        while not done and steps < config.MAX_TOTAL_STEPS:
            # 批量处理多个机组决策
            batch_size = min(32, len(info['pending_crews']))
            
            if batch_size > 0:
                # 批量获取状态和动作
                states, action_features, action_masks = env.get_batch_observations(batch_size)
                
                # 批量前向传播（使用混合精度）
                with autocast(enabled=(scaler is not None)):
                    with torch.no_grad():
                        action_probs, values = model(
                            torch.FloatTensor(states).to(device),
                            torch.FloatTensor(action_features).to(device),
                            torch.BoolTensor(action_masks).to(device)
                        )
                        
                # 批量选择动作
                actions = []
                for i in range(batch_size):
                    if info['valid_actions'][i]:
                        probs = action_probs[i].cpu().numpy()
                        action_idx = np.random.choice(len(probs), p=probs)
                        actions.append(action_idx)
                    else:
                        actions.append(-1)
                        
                # 批量执行动作
                next_states, rewards, dones, infos = env.step_batch(actions)
                
                # 收集经验
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_rewards.extend(rewards)
                batch_values.extend(values.cpu().numpy())
                
                # 更新统计
                total_reward += sum(rewards)
                for i, info in enumerate(infos):
                    if info.get('flight_covered'):
                        covered_flights.add(info['flight_covered'])
                        # 检查飞行周期约束
                        crew_id = info.get('crew_id')
                        flight_id = info['flight_covered']
                        if crew_id and not cycle_constraints.check_flight_feasibility(crew_id, flight_id):
                            cycle_violations += 1
                        
                steps += batch_size
                
        # 计算覆盖率
        total_flights = len(data_handler.data['flights'])
        coverage_rate = len(covered_flights) / total_flights if total_flights > 0 else 0
        
        # 更新最佳覆盖率
        if coverage_rate > best_coverage:
            best_coverage = coverage_rate
            episodes_without_improvement = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'models/best_model.pth')
        else:
            episodes_without_improvement += 1
            
        # 训练模型（每个episode结束后）
        if len(batch_states) > 0:
            train_loss = train_on_batch(
                model, optimizer, scaler,
                batch_states, batch_actions, batch_rewards, batch_values,
                device
            )
            
        # 打印进度
        episode_time = time.time() - episode_start
        print(f"Episode {episode+1}/{config.NUM_EPISODES} | "
              f"覆盖率: {coverage_rate:.2%} | "
              f"最佳覆盖率: {best_coverage:.2%} | "
              f"奖励: {total_reward:.1f} | "
              f"周期违规: {cycle_violations} | "
              f"时间: {episode_time:.1f}s")
              
        # 早停检查
        if coverage_rate >= 0.85:  # 达到85%覆盖率即可
            print(f"\n达到目标覆盖率 {coverage_rate:.2%}，提前结束训练！")
            break
            
        if episodes_without_improvement >= 10:
            print(f"\n{episodes_without_improvement} 轮未改进，提前结束训练！")
            break
            
    print(f"\n训练完成！最佳覆盖率: {best_coverage:.2%}")
    print(f"结束时间: {datetime.now()}")
    
def train_on_batch(model, optimizer, scaler, states, actions, rewards, values, device):
    """批量训练"""
    # 转换为张量
    states_tensor = torch.FloatTensor(states).to(device)
    actions_tensor = torch.LongTensor(actions).to(device)
    rewards_tensor = torch.FloatTensor(rewards).to(device)
    values_tensor = torch.FloatTensor(values).to(device)
    
    # 计算优势函数
    advantages = rewards_tensor - values_tensor
    
    # 标准化优势函数
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # PPO训练循环
    for _ in range(config.PPO_EPOCHS):
        # 使用混合精度
        with autocast(enabled=(scaler is not None)):
            # 前向传播
            action_probs, new_values = model(states_tensor, None, None)
            
            # 计算损失
            # ... PPO损失计算 ...
            
        # 反向传播
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        optimizer.zero_grad()
        
    return loss.item()

if __name__ == '__main__':
    setup_optimized_environment()
    run_optimized_training()