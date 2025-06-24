# main.py

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm, trange
import numpy as np
import os
import collections

# Import our custom modules
import config
from utils import DataHandler, calculate_final_score
from environment import CrewRosteringEnv
from model import ActorCritic

def train():
    """
    主训练函数，负责模型的整个学习过程。
    """
    print(f"Using device: {config.DEVICE}")
    if not os.path.exists(config.MODEL_SAVE_PATH):
        os.makedirs(config.MODEL_SAVE_PATH)

    data_handler = DataHandler()
    env = CrewRosteringEnv(data_handler)
    
    model = ActorCritic(config.STATE_DIM, config.ACTION_DIM, config.HIDDEN_DIM).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    best_score = -np.inf
    print("--- Starting Training ---")

    episode_pbar = trange(config.NUM_EPISODES, desc="Episodes")
    
    for episode in episode_pbar:
        observation, info = env.reset()
        terminated, truncated = False, False
        
        memory = collections.defaultdict(list)
        
        # --- 核心修改：设置一个标志位，决定是否打印详细信息 ---
        print_details = (episode + 1) % 100 == 0 or episode == 0

        if print_details:
            tqdm.write(f"\n===== Episode {episode+1} Detailed Log =====")

        while not (terminated or truncated):
            current_crew_id = env.crews[env.current_crew_idx]['crewId']
            
            if not info['valid_actions']:
                action_idx = -1
                
                if print_details:
                    tqdm.write(f"  Step {env.total_steps}: Crew {current_crew_id} -> No valid actions. Passing.")
                
                # ... (占位符添加逻辑不变) ...
                state = observation['state']
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(config.DEVICE)
                dummy_actions = torch.zeros(1, config.MAX_CANDIDATE_ACTIONS, config.ACTION_DIM).to(config.DEVICE)
                dummy_mask = torch.zeros(1, config.MAX_CANDIDATE_ACTIONS, dtype=torch.bool).to(config.DEVICE)
                with torch.no_grad(): _, value = model(state_tensor, dummy_actions, dummy_mask)
                memory['states'].append(state_tensor); memory['actions'].append(torch.tensor([-1]).to(config.DEVICE)); memory['log_probs'].append(torch.tensor([0.0]).to(config.DEVICE)); memory['values'].append(value); memory['candidate_actions'].append(dummy_actions); memory['action_masks'].append(dummy_mask)
            else:
                obs_dict = observation
                state_tensor = torch.FloatTensor(obs_dict['state']).unsqueeze(0)
                candidate_action_tensor = torch.FloatTensor(obs_dict['action_features']).unsqueeze(0)
                action_mask_tensor = torch.BoolTensor(obs_dict['action_mask'] == 1).unsqueeze(0)

                with torch.no_grad():
                    dist, value = model(state_tensor, candidate_action_tensor, action_mask_tensor)
                
                action = dist.sample()
                log_prob = dist.log_prob(action)
                action_idx = action.item()
                
                # --- 核心修改：打印决策信息 ---
                if print_details:
                    chosen_task = info['valid_actions'][action_idx]
                    tqdm.write(
                        f"  Step {env.total_steps}: Crew {current_crew_id} -> Chose '{chosen_task['type']}' "
                        f"task {chosen_task['taskId']} ({chosen_task['depaAirport']}->{chosen_task['arriAirport']}). "
                        f"Options: {len(info['valid_actions'])}."
                    )
                
                # ... (存储经验逻辑不变) ...
                memory['states'].append(state_tensor); memory['actions'].append(action); memory['log_probs'].append(log_prob); memory['values'].append(value); memory['candidate_actions'].append(candidate_action_tensor); memory['action_masks'].append(action_mask_tensor)

            next_observation, reward, terminated, truncated, next_info = env.step(action_idx)
            
            memory['rewards'].append(torch.FloatTensor([reward]).to(config.DEVICE))
            done = terminated or truncated
            memory['dones'].append(torch.FloatTensor([1.0 if done else 0.0]).to(config.DEVICE))
            
            observation, info = next_observation, next_info
            if done: 
                if print_details:
                    tqdm.write(f"===== Episode {episode+1} Finished =====")
                break
        
        final_reward = memory['rewards'][-1].item()
        episode_pbar.set_description(f"Episodes (Last Reward: {final_reward:.2f}, Best: {best_score:.2f})")

        # --- PPO 更新逻辑 (保持不变) ---
        if len(memory['rewards']) > 0:
            rewards, values, dones = memory['rewards'], memory['values'], memory['dones']
            gae, returns = 0, []
            next_value = torch.zeros(1,1).to(config.DEVICE)
            if not (terminated or truncated):
                 with torch.no_grad():
                    obs_dict = observation
                    state_tensor = torch.FloatTensor(obs_dict['state']).unsqueeze(0)
                    dummy_actions = torch.zeros(1, config.MAX_CANDIDATE_ACTIONS, config.ACTION_DIM)
                    dummy_mask = torch.zeros(1, config.MAX_CANDIDATE_ACTIONS, dtype=torch.bool)
                    _, next_value = model(state_tensor.to(config.DEVICE), dummy_actions.to(config.DEVICE), dummy_mask.to(config.DEVICE))
            for i in reversed(range(len(rewards))):
                delta = rewards[i] + config.GAMMA * next_value * (1 - dones[i]) - values[i]
                gae = delta + config.GAMMA * 0.95 * gae * (1 - dones[i])
                returns.insert(0, gae + values[i])
                next_value = values[i]
            
            valid_indices = [i for i, act in enumerate(torch.cat(memory['actions'])) if act.item() != -1]
            if not valid_indices: continue
            old_states, old_actions, old_log_probs = torch.cat(memory['states'])[valid_indices], torch.cat(memory['actions'])[valid_indices], torch.cat(memory['log_probs'])[valid_indices]
            returns = torch.cat(returns)[valid_indices].detach()
            advantages = (returns - torch.cat(values)[valid_indices].detach()).squeeze()
            if advantages.numel() > 1: advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            elif advantages.numel() == 1: advantages = advantages.unsqueeze(0)
            old_candidate_actions, old_action_masks = torch.cat(memory['candidate_actions'])[valid_indices], torch.cat(memory['action_masks'])[valid_indices]

            for _ in range(config.PPO_EPOCHS):
                dist, value = model(old_states, old_candidate_actions, old_action_masks)
                entropy, new_log_probs = dist.entropy().mean(), dist.log_prob(old_actions)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1, surr2 = ratio * advantages, torch.clamp(ratio, 1 - config.PPO_EPSILON, 1 + config.PPO_EPSILON) * advantages
                actor_loss, critic_loss = -torch.min(surr1, surr2).mean(), nn.MSELoss()(value.squeeze(), returns.squeeze())
                loss = actor_loss + 0.5 * critic_loss - config.ENTROPY_COEF * entropy
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5); optimizer.step()

        if final_reward > best_score:
            best_score = final_reward
            torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_PATH, "best_model.pth"))
            tqdm.write(f"** New best model saved with score: {best_score:.2f} at episode {episode+1} **")

    episode_pbar.close()
    print("--- Training Finished ---")

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
        assignable_tasks = [t for t in tasks if t.get('type') != 'groundDuty']
        assignable_tasks.sort(key=lambda x: x['startTime'])
        for task in assignable_tasks:
            is_ddh = "1" if 'positioning' in task.get('type', '') else "0"
            output_data.append({"crewId": crew_id, "taskId": task['taskId'], "isDDH": is_ddh})
    
    output_df = pd.DataFrame(output_data) if output_data else pd.DataFrame(columns=["crewId", "taskId", "isDDH"])
    
    output_filename = "rosterResult.csv"
    if not is_final:
        output_filename = f"rosterResult_best_at_episode.csv"

    output_df.to_csv(output_filename, index=False)
    tqdm.write(f"Roster saved to {output_filename}")


if __name__ == '__main__':
    train()
    evaluate_and_save_roster(is_final=True) 