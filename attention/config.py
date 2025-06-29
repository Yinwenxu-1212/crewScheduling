# config.py

import torch

# --- 竞赛评价指标权重 ---
SCORE_FLY_TIME_MULTIPLIER = 100
PENALTY_UNCOVERED_FLIGHT = -50
PENALTY_OVERNIGHT_STAY_AWAY_FROM_BASE = -1.0
PENALTY_POSITIONING = -0.5
PENALTY_RULE_VIOLATION = -10.0

# --- 奖励塑造参数 ---
IMMEDIATE_COVERAGE_REWARD = 15.0
PENALTY_PASS_ACTION = -2.0

# --- 强化学习超参数 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GAMMA = 0.998
LEARNING_RATE = 3e-5
PPO_EPSILON = 0.2
PPO_EPOCHS = 5
ENTROPY_COEF = 0.01  # 熵系数，用于鼓励探索

# --- 新增：Epsilon-Greedy 探索参数 ---
EPSILON_START = 0.3  # 初始探索率 (30%的概率随机选择)
EPSILON_END = 0.01   # 最终探索率 (1%的概率随机选择)
EPSILON_DECAY_EPISODES = 5000 # 经过多少个episodes衰减到最终值

# --- 模型架构参数 ---
STATE_DIM = 14              # 状态特征维度，与_extract_state_features保持一致 (索引0-13)
ACTION_DIM = 13             # 动作特征维度，与_extract_task_features保持一致 (索引0-12)
HIDDEN_DIM = 256

# --- 训练参数 ---
NUM_EPISODES = 1
MAX_STEPS_PER_CREW = 20
MAX_CANDIDATE_ACTIONS = 250

# --- 路径 ---
DATA_PATH = "data/"
MODEL_SAVE_PATH = "models/"

# --- 时间设定 ---
PLANNING_START_DATE = "2025-04-29 00:00:00"
PLANNING_END_DATE = "2025-05-07 23:59:59"