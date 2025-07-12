# main.py

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm, trange
import numpy as np
import os
import sys
import collections
import random
from collections import deque
import argparse
from datetime import datetime
import json
import matplotlib.pyplot as plt

# Import our custom modules
try:
    # 相对导入（当作为模块导入时）
    from . import config
    from .utils import DataHandler, calculate_final_score
    from .environment import CrewRosteringEnv
    from .model import ActorCritic
    from .training_monitor import TrainingMonitor, FeatureAnalyzer
except ImportError:
    # 绝对导入（当直接运行时）
    import config
    from utils import DataHandler, calculate_final_score
    from environment import CrewRosteringEnv
    from model import ActorCritic
    from training_monitor import TrainingMonitor, FeatureAnalyzer

class ExperienceReplay:
    """经验回放缓冲区"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)

class CurriculumLearning:
    """课程学习管理器"""
    def __init__(self):
        self.current_difficulty = 0.3  # 初始难度
        self.max_difficulty = 1.0
        self.difficulty_increment = 0.1
        self.episodes_per_level = 50
        self.episode_count = 0
    
    def update_difficulty(self, success_rate):
        self.episode_count += 1
        if self.episode_count % self.episodes_per_level == 0:
            if success_rate > 0.7 and self.current_difficulty < self.max_difficulty:
                self.current_difficulty = min(self.current_difficulty + self.difficulty_increment, self.max_difficulty)
                print(f"Curriculum: Difficulty increased to {self.current_difficulty:.2f}")
    
    def get_flight_subset_ratio(self):
        """返回当前难度下应该使用的航班比例"""
        return self.current_difficulty

def train():
    """改进的主训练函数，集成双头架构监控、经验回放、课程学习和多项优化策略"""
    print(f"Using device: {config.DEVICE}")
    
    # 创建必要的目录
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs("log", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # 初始化环境和模型
    data_handler = DataHandler()
    
    # 大规模机组适配信息
    crew_count = len(data_handler.data['crews'])
    max_steps_per_episode = min(
        crew_count * config.MAX_STEPS_PER_CREW * 1.5,
        config.MAX_TOTAL_STEPS
    )
    env = CrewRosteringEnv(data_handler)
    model = ActorCritic(config.STATE_DIM, config.ACTION_DIM, config.HIDDEN_DIM).to(config.DEVICE)
    
    # 优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.LR_DECAY_STEP, gamma=config.LR_DECAY_GAMMA)
    
    # 初始化训练组件
    experience_replay = ExperienceReplay(capacity=config.REPLAY_BUFFER_SIZE)
    curriculum = CurriculumLearning()
    monitor = TrainingMonitor(window_size=100)
    analyzer = FeatureAnalyzer()
    
    best_score = -np.inf
    recent_scores = deque(maxlen=100)
    success_count = 0
    

    episode_pbar = trange(config.NUM_EPISODES, desc="Episodes")
    
    for episode in episode_pbar:
        # 课程学习：动态调整训练难度
        flight_ratio = curriculum.get_flight_subset_ratio()
        
        # 每个episode都输出进度
        print(f"\n开始Episode {episode+1}/{config.NUM_EPISODES}")
        sys.stdout.flush()
        
        observation, info = env.reset()
        terminated, truncated = False, False
        
        episode_memory = collections.defaultdict(list)
        episode_reward = 0
        step_count = 0
        
        # 双头架构统计
        episode_alpha_weights = []
        episode_beta_weights = []
        episode_dual_contributions = []
        episode_base_contributions = []
        constraint_violations = 0
        
        print_details = (episode + 1) % config.PRINT_DETAILS_INTERVAL == 0 or episode == 0
        if print_details:
            tqdm.write(f"\n===== Episode {episode+1} (Difficulty: {flight_ratio:.2f}) =====")

        while not (terminated or truncated):
            current_crew_id = env.crews[env.current_crew_idx]['crewId']
            
            if not info['valid_actions']:
                action_idx = -1
                if print_details:
                    tqdm.write(f"  Step {step_count}: Crew {current_crew_id} -> No valid actions")
                
                state = observation['state']
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(config.DEVICE)
                dummy_actions = torch.zeros(1, config.MAX_CANDIDATE_ACTIONS, config.ACTION_DIM).to(config.DEVICE)
                dummy_mask = torch.zeros(1, config.MAX_CANDIDATE_ACTIONS, dtype=torch.bool).to(config.DEVICE)
                
                with torch.no_grad(): 
                    _, value = model(state_tensor, dummy_actions, dummy_mask)
                
                episode_memory['states'].append(state_tensor)
                episode_memory['actions'].append(torch.tensor([-1]).to(config.DEVICE))
                episode_memory['log_probs'].append(torch.tensor([0.0]).to(config.DEVICE))
                episode_memory['values'].append(value)
                episode_memory['candidate_actions'].append(dummy_actions)
                episode_memory['action_masks'].append(dummy_mask)
            else:
                obs_dict = observation
                state_tensor = torch.FloatTensor(obs_dict['state']).unsqueeze(0).to(config.DEVICE)
                candidate_action_tensor = torch.FloatTensor(obs_dict['action_features']).unsqueeze(0).to(config.DEVICE)
                action_mask_tensor = torch.BoolTensor(obs_dict['action_mask'] == 1).unsqueeze(0).to(config.DEVICE)

                with torch.no_grad():
                    dist, value = model(state_tensor, candidate_action_tensor, action_mask_tensor)
                    
                    # 获取双头架构的内部信息
                    model.eval()
                    
                    # 分离特征
                    base_features = candidate_action_tensor[:, :, :13]
                    dual_features = candidate_action_tensor[:, :, 13:]
                    
                    # 计算各部分贡献
                    base_encoded = model.base_feature_extractor(base_features)
                    dual_encoded = model.dual_price_extractor(dual_features)
                    combined_features = torch.cat([base_encoded, dual_encoded], dim=-1)
                    actions_encoded = model.feature_fusion(combined_features)
                    
                    # 计算双头评分
                    action_scores = model.action_scorer(actions_encoded).squeeze(-1)
                    dual_values = model.dual_value_head(actions_encoded).squeeze(-1)
                    
                    # 获取自适应权重
                    alpha, beta = model._compute_adaptive_weights(action_scores, dual_values, state_tensor)
                    
                    # 记录双头架构统计
                    episode_alpha_weights.append(alpha.mean().item())
                    episode_beta_weights.append(beta.mean().item())
                    episode_dual_contributions.append(torch.abs(dual_values).mean().item())
                    episode_base_contributions.append(torch.abs(action_scores).mean().item())
                    
                    model.train()
                
                # 改进的动作选择策略（结合探索和利用）
                if episode < config.NUM_EPISODES * config.EXPLORATION_PHASE_RATIO:  # 前期增加探索
                    action = dist.sample()
                else:  # 后期更多利用
                    if random.random() < config.LATE_EXPLORATION_PROB:  # 后期探索概率
                        action = dist.sample()
                    else:
                        action = torch.argmax(dist.probs, dim=1)
                
                log_prob = dist.log_prob(action)
                action_idx = action.item()
                
                if print_details and len(info['valid_actions']) > action_idx and step_count % config.STEP_PRINT_INTERVAL == 0:
                    chosen_task = info['valid_actions'][action_idx]
                    if chosen_task['type'] == 'ground_duty':
                        tqdm.write(
                            f"  Step {step_count}: Crew {current_crew_id} -> ground_duty {chosen_task['taskId']}"
                        )
                    else:
                        tqdm.write(
                            f"  Step {step_count}: Crew {current_crew_id} -> {chosen_task['type']} {chosen_task['taskId']}"
                        )
                
                episode_memory['states'].append(state_tensor)
                episode_memory['actions'].append(action)
                episode_memory['log_probs'].append(log_prob)
                episode_memory['values'].append(value)
                episode_memory['candidate_actions'].append(candidate_action_tensor)
                episode_memory['action_masks'].append(action_mask_tensor)

            next_observation, reward, terminated, truncated, next_info = env.step(action_idx)
            
            episode_memory['rewards'].append(torch.FloatTensor([reward]).to(config.DEVICE))
            episode_memory['dones'].append(torch.FloatTensor([1.0 if (terminated or truncated) else 0.0]).to(config.DEVICE))
            
            episode_reward += reward
            step_count += 1
            observation, info = next_observation, next_info
            
            if terminated or truncated:
                if print_details:
                    termination_reason = "所有航班已分配" if terminated else "达到步数限制"
                    tqdm.write(f"===== Episode {episode+1} Finished ({termination_reason}, Steps: {step_count}, Reward: {episode_reward:.2f}) =====")
                # 每个episode都输出完成信息
                termination_reason = "完成" if terminated else "截断"
                print(f"Episode {episode+1} {termination_reason}，步数: {step_count}, 奖励: {episode_reward:.2f}")
                sys.stdout.flush()
                break
        
        # 记录episode结果
        recent_scores.append(episode_reward)
        if episode_reward > config.SUCCESS_THRESHOLD:  # 定义成功的阈值
            success_count += 1
        
        # 计算解决方案质量指标
        coverage_rate = getattr(env, 'get_coverage_rate', lambda: 0.0)()
        solution_quality = episode_reward / max(step_count, 1)
        
        # 记录episode数据到监控器
        episode_data = {
            'total_reward': episode_reward,
            'episode_length': step_count,
            'avg_alpha': np.mean(episode_alpha_weights) if episode_alpha_weights else 0.7,
            'avg_beta': np.mean(episode_beta_weights) if episode_beta_weights else 0.3,
            'avg_dual_contribution': np.mean(episode_dual_contributions) if episode_dual_contributions else 0,
            'avg_base_contribution': np.mean(episode_base_contributions) if episode_base_contributions else 0,
            'constraint_violations': constraint_violations,
            'coverage_rate': coverage_rate,
            'solution_quality': solution_quality
        }
        monitor.log_episode(episode_data)
        
        # 存储经验到回放缓冲区
        if len(episode_memory['rewards']) > 0:
            experience_replay.push({
                'states': episode_memory['states'],
                'actions': episode_memory['actions'],
                'log_probs': episode_memory['log_probs'],
                'values': episode_memory['values'],
                'rewards': episode_memory['rewards'],
                'dones': episode_memory['dones'],
                'candidate_actions': episode_memory['candidate_actions'],
                'action_masks': episode_memory['action_masks']
            })
        
        # 更新模型（使用经验回放）
        if len(experience_replay) >= config.MIN_REPLAY_SIZE:  # 有足够经验时开始训练
            actor_loss, critic_loss = train_model_with_replay(model, optimizer, experience_replay, episode)
            episode_data['actor_loss'] = actor_loss
            episode_data['critic_loss'] = critic_loss
        
        # 特征重要性分析（每5个episode）
        if episode % 5 == 0 and episode > 0:
            sample_batch = experience_replay.sample(min(32, len(experience_replay)))
            if sample_batch:
                # 从经验回放中提取样本进行特征分析
                sample_states = []
                sample_actions = []
                sample_masks = []
                
                for exp in sample_batch[:5]:  # 取前5个经验
                    if len(exp['states']) > 0:
                        sample_states.extend(exp['states'][:3])  # 每个经验取前3个状态
                        sample_actions.extend(exp['candidate_actions'][:3])
                        sample_masks.extend(exp['action_masks'][:3])
                
                if sample_states:
                    states = torch.cat(sample_states).to(config.DEVICE)
                    actions = torch.cat(sample_actions).to(config.DEVICE)
                    masks = torch.cat(sample_masks).to(config.DEVICE)
                    
                    feature_importance = analyzer.analyze_feature_importance(model, (states, actions, masks))
                    monitor.log_dual_price_features({
                        'feature_importance': feature_importance,
                        'avg_alpha': np.mean(episode_alpha_weights) if episode_alpha_weights else 0.7,
                        'avg_beta': np.mean(episode_beta_weights) if episode_beta_weights else 0.3
                    })
        
        # 更新最佳模型
        if episode_reward > best_score:
            best_score = episode_reward
            # 生成带日期时间的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"best_model_{timestamp}.pth"
            
            # 保存模型和双头架构统计
            model_save_data = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode': episode,
                'best_score': best_score,
                'dual_head_stats': {
                    'avg_alpha': np.mean(episode_alpha_weights) if episode_alpha_weights else 0.7,
                    'avg_beta': np.mean(episode_beta_weights) if episode_beta_weights else 0.3,
                    'dual_contribution': np.mean(episode_dual_contributions) if episode_dual_contributions else 0,
                    'base_contribution': np.mean(episode_base_contributions) if episode_base_contributions else 0
                }
            }
            
            torch.save(model_save_data, os.path.join(config.MODEL_SAVE_PATH, model_filename))
            torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_PATH, "best_model.pth"))
            
            tqdm.write(f"** New best model saved with score: {best_score:.2f} at episode {episode+1} **")
            tqdm.write(f"** Model saved as: {model_filename} **")
            if episode_alpha_weights:
                tqdm.write(f"** Dual-head stats - Alpha: {np.mean(episode_alpha_weights):.3f}, Beta: {np.mean(episode_beta_weights):.3f} **")
        
        # 课程学习更新
        if episode > 0 and episode % config.CURRICULUM_UPDATE_INTERVAL == 0:
            success_rate = success_count / config.CURRICULUM_UPDATE_INTERVAL
            curriculum.update_difficulty(success_rate)
            success_count = 0
        
        # 学习率调度
        scheduler.step()
        
        # 更新进度条（包含双头架构信息）
        avg_score = np.mean(recent_scores) if recent_scores else episode_reward
        
        progress_desc = f"Episodes (Avg: {avg_score:.2f}, Best: {best_score:.2f}, LR: {scheduler.get_last_lr()[0]:.6f}"
        if episode_alpha_weights:
            progress_desc += f", α: {np.mean(episode_alpha_weights):.2f}, β: {np.mean(episode_beta_weights):.2f}"
        progress_desc += ")"
        
        episode_pbar.set_description(progress_desc)
        
        # 保存训练进度（每500个episode）
        if episode % 500 == 0 and episode > 0:
            monitor.save_training_log(f'log/training_log_episode_{episode}.json')
            monitor.plot_training_progress(f'plots/training_progress_episode_{episode}.png')
            
            # 打印双头架构详细统计
            if episode_alpha_weights:
                tqdm.write(f"\n--- Episode {episode} 双头架构统计 ---")
                tqdm.write(f"平均Alpha权重: {np.mean(episode_alpha_weights):.3f}")
                tqdm.write(f"平均Beta权重: {np.mean(episode_beta_weights):.3f}")
                tqdm.write(f"对偶价格平均贡献: {np.mean(episode_dual_contributions):.3f}")
                tqdm.write(f"基础评分平均贡献: {np.mean(episode_base_contributions):.3f}")
                tqdm.write(f"约束违反次数: {constraint_violations}")

    episode_pbar.close()
    
    print("\n=== 双头架构训练完成！ ===")
    
    # 最终分析和报告
    final_stats = monitor.get_current_stats()
    print(f"\n最终训练统计:")
    print(f"总Episode数: {final_stats.get('episode_count', config.NUM_EPISODES)}")
    print(f"最佳奖励: {final_stats.get('best_reward', best_score):.4f}")
    print(f"最终Alpha权重: {final_stats.get('avg_alpha_weight', 0.7):.3f}")
    print(f"最终Beta权重: {final_stats.get('avg_beta_weight', 0.3):.3f}")
    print(f"对偶价格特征贡献: {final_stats.get('dual_price_contribution', 0):.3f}")
    
    # 保存最终结果
    monitor.save_training_log('log/final_training_log.json')
    monitor.plot_training_progress('plots/final_training_progress.png')
    
    print("--- Enhanced Training with Dual-Head Architecture Finished ---")
    
    return model, monitor, analyzer

def train_model_with_replay(model, optimizer, experience_replay, episode):
    """改进的PPO训练函数，返回损失值用于监控"""
    batch_size = min(config.BATCH_SIZE, len(experience_replay))
    experiences = experience_replay.sample(batch_size)
    
    all_states, all_actions, all_log_probs = [], [], []
    all_values, all_returns, all_candidate_actions, all_action_masks = [], [], [], []
    
    for exp in experiences:
        if len(exp['rewards']) == 0:
            continue
            
        # 计算GAE和returns
        rewards, values, dones = exp['rewards'], exp['values'], exp['dones']
        returns = compute_gae_returns(rewards, values, dones)
        
        # 过滤有效的动作
        valid_indices = [i for i, act in enumerate(torch.cat(exp['actions'])) if act.item() != -1]
        if not valid_indices:
            continue
            
        all_states.extend([exp['states'][i] for i in valid_indices])
        all_actions.extend([exp['actions'][i] for i in valid_indices])
        all_log_probs.extend([exp['log_probs'][i] for i in valid_indices])
        all_values.extend([exp['values'][i] for i in valid_indices])
        all_returns.extend([returns[i] for i in valid_indices])
        all_candidate_actions.extend([exp['candidate_actions'][i] for i in valid_indices])
        all_action_masks.extend([exp['action_masks'][i] for i in valid_indices])
    
    if not all_states:
        return 0.0, 0.0
    
    # 转换为tensor
    old_states = torch.cat(all_states)
    old_actions = torch.cat(all_actions)
    old_log_probs = torch.cat(all_log_probs)
    old_values = torch.cat(all_values)
    returns = torch.cat(all_returns).detach()
    old_candidate_actions = torch.cat(all_candidate_actions)
    old_action_masks = torch.cat(all_action_masks)
    
    # 计算优势
    advantages = (returns - old_values.detach()).squeeze()
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    elif advantages.numel() == 1:
        advantages = advantages.unsqueeze(0)
    
    total_actor_loss = 0.0
    total_critic_loss = 0.0
    
    # PPO更新时添加梯度裁剪和正则化
    for ppo_epoch in range(config.PPO_EPOCHS):
        dist, value = model(old_states, old_candidate_actions, old_action_masks)
        
        # 添加分布的熵正则化以避免过早收敛
        entropy = dist.entropy().mean()
        
        # 确保有效动作的处理
        valid_mask = old_actions >= 0
        if not valid_mask.any():
            continue
            
        new_log_probs = dist.log_prob(old_actions)
        
        # KL散度约束，防止策略更新过大
        kl_div = (old_log_probs - new_log_probs).mean()
        if kl_div > getattr(config, 'KL_TARGET', 0.02) * 1.5:
            print(f"Early stopping due to KL divergence: {kl_div:.4f}")
            break  # 早停机制
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # 添加ratio的额外裁剪，防止极端值
        ratio = torch.clamp(ratio, 0.5, 2.0)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - config.PPO_EPSILON, 1 + config.PPO_EPSILON) * advantages
        
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # 改进的价值损失，使用Huber损失减少异常值影响
        critic_loss = nn.SmoothL1Loss()(value.squeeze(), returns.squeeze())
        
        total_actor_loss += actor_loss.item()
        total_critic_loss += critic_loss.item()
        
        # 动态熵系数
        entropy_coef = config.ENTROPY_COEF * max(0.1, 1.0 - episode / config.NUM_EPISODES)
        
        # 添加L2正则化
        l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
        
        loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy + 1e-4 * l2_reg
        
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        # 检查梯度是否正常
        grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        if grad_norm > 100:
            print(f"Warning: Large gradient norm: {grad_norm:.2f}")
        
        optimizer.step()
    
    return total_actor_loss / config.PPO_EPOCHS, total_critic_loss / config.PPO_EPOCHS

def compute_gae_returns(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """计算GAE returns"""
    returns = []
    gae = 0
    next_value = torch.zeros(1, 1).to(config.DEVICE)
    
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        gae = delta + gamma * gae_lambda * gae * (1 - dones[i])
        returns.insert(0, gae + values[i])
        next_value = values[i]
    
    return returns

# ... (evaluate_and_save_roster 函数和 if __name__ == '__main__' 部分保持不变) ...

def evaluate_and_save_roster(is_final=True):
    if is_final:
        print("\n--- Starting Final Evaluation and Roster Generation ---")
    
    data_handler = DataHandler()
    env = CrewRosteringEnv(data_handler)
    
    model = ActorCritic(config.STATE_DIM, config.ACTION_DIM, config.HIDDEN_DIM).to(config.DEVICE)
    model_path = os.path.join(config.MODEL_SAVE_PATH, "best_model.pth")
    
    if not os.path.exists(model_path):
        print("Error: No trained model found. Please run training first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()

    observation, info = env.reset()
    done = False
    with torch.no_grad():
        while not done:
            if not info['valid_actions']:
                action_idx = -1
            else:
                obs_dict = observation
                state_tensor = torch.FloatTensor(obs_dict['state']).unsqueeze(0)
                candidate_action_tensor = torch.FloatTensor(obs_dict['action_features']).unsqueeze(0)
                action_mask_tensor = torch.BoolTensor(obs_dict['action_mask'] == 1).unsqueeze(0)
                dist, _ = model(state_tensor, candidate_action_tensor, action_mask_tensor)
                action_idx = torch.argmax(dist.probs, dim=1).item()
            
            observation, _, terminated, truncated, info = env.step(action_idx)
            done = terminated or truncated
    
    final_roster_plan = env.roster_plan
    final_score = calculate_final_score(final_roster_plan, data_handler)
    tqdm.write(f"Generated roster score: {final_score:.2f}")

    output_data = []
    for crew_id, tasks in final_roster_plan.items():
        assignable_tasks = [t for t in tasks if t.get('type') != 'ground_duty']
        assignable_tasks.sort(key=lambda x: x['startTime'])
        for task in assignable_tasks:
            is_ddh = "1" if 'positioning' in task.get('type', '') else "0"
            output_data.append({"crewId": crew_id, "taskId": task['taskId'], "isDDH": is_ddh})
    
    output_df = pd.DataFrame(output_data) if output_data else pd.DataFrame(columns=["crewId", "taskId", "isDDH"])
    
    # 生成带日期时间的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if is_final:
        output_filename = f"rosterResult_{timestamp}.csv"
    else:
        output_filename = f"rosterResult_best_at_episode_{timestamp}.csv"
    
    # 同时保存一个通用的最新结果文件
    output_df.to_csv("rosterResult.csv", index=False)
    output_df.to_csv(output_filename, index=False)
    tqdm.write(f"Roster saved to {output_filename}")
    tqdm.write(f"Latest roster also saved to rosterResult.csv")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='机组排班强化学习训练')
    
    # 数据配置参数
    parser.add_argument('--data-path', type=str, default=config.DATA_PATH,
                        help=f'数据文件路径 (默认: {config.DATA_PATH})')
    parser.add_argument('--start-date', type=str, default=None,
                        help='规划开始日期 (格式: YYYY-MM-DD，如不指定则自动从数据中检测)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='规划结束日期 (格式: YYYY-MM-DD，如不指定则自动从数据中检测)')
    parser.add_argument('--auto-detect-time', action='store_true', default=True,
                        help='自动从数据文件检测时间范围 (默认: True)')
    
    # 训练参数
    parser.add_argument('--episodes', type=int, default=config.NUM_EPISODES,
                        help=f'训练轮数 (默认: {config.NUM_EPISODES})')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE,
                        help=f'学习率 (默认: {config.LEARNING_RATE})')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                        help=f'批次大小 (默认: {config.BATCH_SIZE})')
    parser.add_argument('--replay-size', type=int, default=config.REPLAY_BUFFER_SIZE,
                        help=f'经验回放缓冲区大小 (默认: {config.REPLAY_BUFFER_SIZE})')
    
    # 探索参数
    parser.add_argument('--exploration-ratio', type=float, default=config.EXPLORATION_PHASE_RATIO,
                        help=f'探索阶段比例 (默认: {config.EXPLORATION_PHASE_RATIO})')
    parser.add_argument('--late-exploration', type=float, default=config.LATE_EXPLORATION_PROB,
                        help=f'后期探索概率 (默认: {config.LATE_EXPLORATION_PROB})')
    
    # 课程学习参数
    parser.add_argument('--curriculum-interval', type=int, default=config.CURRICULUM_UPDATE_INTERVAL,
                        help=f'课程学习更新间隔 (默认: {config.CURRICULUM_UPDATE_INTERVAL})')
    parser.add_argument('--success-threshold', type=float, default=config.SUCCESS_THRESHOLD,
                        help=f'成功阈值 (默认: {config.SUCCESS_THRESHOLD})')
    
    # 输出参数
    parser.add_argument('--print-interval', type=int, default=config.PRINT_DETAILS_INTERVAL,
                        help=f'打印详细信息间隔 (默认: {config.PRINT_DETAILS_INTERVAL})')
    parser.add_argument('--model-save-path', type=str, default=config.MODEL_SAVE_PATH,
                        help=f'模型保存路径 (默认: {config.MODEL_SAVE_PATH})')
    
    # 模式选择
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], default='train',
                        help='运行模式: train(训练) 或 evaluate(评估)')
    
    return parser.parse_args()

def update_config_from_args(args):
    """根据命令行参数更新配置"""
    # 数据配置更新
    if args.data_path != config.DATA_PATH:
        config.update_data_path(args.data_path)
        print(f"数据路径已更新为: {args.data_path}")
    
    # 时间配置更新
    if args.start_date or args.end_date:
        if args.start_date and args.end_date:
            config.set_planning_dates_manual(args.start_date, args.end_date)
            print(f"手动设置规划时间: {args.start_date} 到 {args.end_date}")
        else:
            print("警告: 开始日期和结束日期必须同时指定，将使用自动检测")
            config.get_planning_dates_from_data()
    elif args.auto_detect_time:
        config.get_planning_dates_from_data()
        print(f"自动检测到规划时间: {config.PLANNING_START_DATE} 到 {config.PLANNING_END_DATE}")
    
    # 训练参数更新
    config.NUM_EPISODES = args.episodes
    config.LEARNING_RATE = args.lr
    config.BATCH_SIZE = args.batch_size
    config.REPLAY_BUFFER_SIZE = args.replay_size
    config.EXPLORATION_PHASE_RATIO = args.exploration_ratio
    config.LATE_EXPLORATION_PROB = args.late_exploration
    config.CURRICULUM_UPDATE_INTERVAL = args.curriculum_interval
    config.SUCCESS_THRESHOLD = args.success_threshold
    config.PRINT_DETAILS_INTERVAL = args.print_interval
    config.MODEL_SAVE_PATH = args.model_save_path

if __name__ == "__main__":
    args = parse_args()
    update_config_from_args(args)
    
    print("=== 训练配置 ===")
    print(f"数据路径: {config.DATA_PATH}")
    print(f"规划开始时间: {config.PLANNING_START_DATE}")
    print(f"规划结束时间: {config.PLANNING_END_DATE}")
    print(f"训练轮数: {config.NUM_EPISODES}")
    print(f"学习率: {config.LEARNING_RATE}")
    print(f"批次大小: {config.BATCH_SIZE}")
    print(f"经验回放缓冲区大小: {config.REPLAY_BUFFER_SIZE}")
    print(f"探索阶段比例: {config.EXPLORATION_PHASE_RATIO}")
    print(f"后期探索概率: {config.LATE_EXPLORATION_PROB}")
    print(f"课程学习更新间隔: {config.CURRICULUM_UPDATE_INTERVAL}")
    print(f"成功阈值: {config.SUCCESS_THRESHOLD}")
    print(f"模型保存路径: {config.MODEL_SAVE_PATH}")
    print("==================\n")
    if args.mode == 'train':
        train()
    elif args.mode == 'evaluate':
        evaluate_and_save_roster()