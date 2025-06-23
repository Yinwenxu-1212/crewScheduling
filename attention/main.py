# main.py
import torch
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import collections
import torch.nn as nn  # <--- 在这里添加这一行
# Import our custom modules
from . import config
from .utils import DataHandler, calculate_final_score
from .environment import CrewRosteringEnv
from .model import ActorCritic

def train():
    """
    主训练函数，负责模型的整个学习过程。
    """
    print(f"Using device: {config.DEVICE}")
    if not os.path.exists(config.MODEL_SAVE_PATH):
        os.makedirs(config.MODEL_SAVE_PATH)

    # 1. 初始化数据处理器、环境和模型
    data_handler = DataHandler()
    env = CrewRosteringEnv(data_handler)
    
    model = ActorCritic(config.STATE_DIM, config.ACTION_DIM, config.HIDDEN_DIM).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    best_score = -np.inf
    print("--- Starting Training ---")

    # 2. 主训练循环
    for episode in tqdm(range(config.NUM_EPISODES), desc="Training Episodes"):
        observation, info = env.reset()
        done = False
        
        # 存储一个回合的经验数据
        memory = collections.defaultdict(list)

        while not done:
            # 从观测中提取数据
            state = observation['state']
            action_features = observation['action_features']
            action_mask = observation['action_mask']
            
            # 转换为Tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(config.DEVICE)
            candidate_action_tensor = torch.FloatTensor(action_features).unsqueeze(0).to(config.DEVICE)
            action_mask_tensor = torch.BoolTensor(action_mask == 1).unsqueeze(0).to(config.DEVICE) # Mask should be boolean

            # --- 动作选择 ---
            with torch.no_grad():
                dist, value = model(state_tensor, candidate_action_tensor, action_mask_tensor)
            
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # --- 与环境交互 ---
            next_observation, reward, terminated, truncated, next_info = env.step(action.item(), info['valid_actions'])
            done = terminated or truncated

            # 存储经验
            memory['states'].append(state_tensor)
            memory['actions'].append(action)
            memory['log_probs'].append(log_prob)
            memory['rewards'].append(torch.FloatTensor([reward]).to(config.DEVICE))
            memory['dones'].append(torch.FloatTensor([1.0 if done else 0.0]).to(config.DEVICE))
            memory['values'].append(value)
            memory['candidate_actions'].append(candidate_action_tensor)
            memory['action_masks'].append(action_mask_tensor)
            
            # 更新观测
            observation = next_observation
            info = next_info

        final_reward = memory['rewards'][-1].item()
        print(f"Episode {episode+1}/{config.NUM_EPISODES}, Final Reward: {final_reward:.2f}, Best Score: {best_score:.2f}")

        # --- PPO 更新逻辑 ---
        if len(memory['rewards']) > 0:
            # 计算 GAE (Generalized Advantage Estimation)
            rewards = memory['rewards']
            values = memory['values']
            dones = memory['dones']
            gae = 0
            returns = []
            for i in reversed(range(len(rewards))):
                # 如果是最后一个状态，没有下一个value，所以用0
                next_value = values[i+1] if i + 1 < len(values) else torch.zeros(1,1).to(config.DEVICE)
                delta = rewards[i] + config.GAMMA * next_value * (1 - dones[i]) - values[i]
                gae = delta + config.GAMMA * 0.95 * gae * (1 - dones[i])
                returns.insert(0, gae + values[i])
            
            # 转换为tensor
            old_states = torch.cat(memory['states']).detach()
            old_actions = torch.cat(memory['actions']).detach()
            old_log_probs = torch.cat(memory['log_probs']).detach()
            returns = torch.cat(returns).detach()
            advantages = (returns - torch.cat(values).detach()).squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # 标准化优势函数
            old_candidate_actions = torch.cat(memory['candidate_actions']).detach()
            old_action_masks = torch.cat(memory['action_masks']).detach()

            # 多轮次优化
            for _ in tqdm(range(config.PPO_EPOCHS), desc="PPO Updates", leave=False):

                dist, value = model(old_states, old_candidate_actions, old_action_masks)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(old_actions)

                ratio = torch.exp(new_log_probs - old_log_probs)
                
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - config.PPO_EPSILON, 1 + config.PPO_EPSILON) * advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(value.squeeze(), returns.squeeze())
                
                loss = actor_loss + 0.5 * critic_loss - config.ENTROPY_COEF * entropy
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        # 保存最佳模型
        if final_reward > best_score:
            best_score = final_reward
            torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_PATH, "best_model.pth"))
            print(f"** New best model saved with score: {best_score:.2f} **")

def evaluate_and_save_roster():
    """
    加载最佳模型，运行一次评估，并生成最终的排班结果CSV文件。
    """
    print("\n--- Starting Evaluation and Roster Generation ---")
    
    # 1. 初始化环境和模型
    data_handler = DataHandler()
    env = CrewRosteringEnv(data_handler)
    
    model = ActorCritic(config.STATE_DIM, config.ACTION_DIM, config.HIDDEN_DIM).to(config.DEVICE)
    model_path = os.path.join(config.MODEL_SAVE_PATH, "best_model.pth")
    
    if not os.path.exists(model_path):
        print("Error: No trained model found. Please run the training first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval() # 设置为评估模式

    # 2. 运行一个评估回合
    observation, info = env.reset()
    done = False
    
    with torch.no_grad():
        while not done:
            state = observation['state']
            action_features = observation['action_features']
            action_mask = observation['action_mask']
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(config.DEVICE)
            candidate_action_tensor = torch.FloatTensor(action_features).unsqueeze(0).to(config.DEVICE)
            action_mask_tensor = torch.BoolTensor(action_mask == 1).unsqueeze(0).to(config.DEVICE)

            dist, _ = model(state_tensor, candidate_action_tensor, action_mask_tensor)
            
            # 在评估时，选择概率最高的动作（贪婪策略）
            action = torch.argmax(dist.probs, dim=1)
            
            observation, _, terminated, truncated, info = env.step(action.item(), info['valid_actions'])
            done = terminated or truncated

    final_roster_plan = env.roster_plan
    final_score = calculate_final_score(final_roster_plan, data_handler)
    print(f"Generated roster score: {final_score:.2f}")

    # 3. 格式化并保存结果
    output_data = []
    for crew_id, tasks in final_roster_plan.items():
        # 仅保留非占位任务
        assignable_tasks = [t for t in tasks if 'groundDuty' not in t.get('type', '')]
        
        # 按开始时间排序
        assignable_tasks.sort(key=lambda x: x['startTime'])
        
        for task in assignable_tasks:
            is_ddh = "1" if 'positioning' in task.get('type', '') else "0"
            output_data.append({
                "crewId": crew_id,
                "taskId": task['taskId'],
                "isDDH": is_ddh
            })
            
    if not output_data:
        print("Warning: The generated roster is empty.")
        # 创建一个空的DataFrame以符合格式要求
        output_df = pd.DataFrame(columns=["crewId", "taskId", "isDDH"])
    else:
        output_df = pd.DataFrame(output_data)

    output_filename = "rosterResult.csv"
    output_df.to_csv(output_filename, index=False)
    print(f"Final roster saved to {output_filename}")


if __name__ == '__main__':
    # 运行训练过程
    train()
    
    # 训练结束后，自动进行评估并生成结果文件
    evaluate_and_save_roster()