# config.py

import torch

# --- 竞赛评价指标权重 ---
SCORE_FLY_TIME_MULTIPLIER = 1000
PENALTY_UNCOVERED_FLIGHT = -5
PENALTY_OVERNIGHT_STAY = -1.0  # 每晚
PENALTY_POSITIONING = -0.5
PENALTY_RULE_VIOLATION = -10

# --- 强化学习超参数 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GAMMA = 0.99                # 奖励折扣因子
LEARNING_RATE = 3e-4
PPO_EPSILON = 0.2           # PPO裁剪参数
PPO_EPOCHS = 4              # 每次更新策略的轮数
PPO_BATCH_SIZE = 128
ENTROPY_COEF = 0.01

# --- 模型架构参数 ---
STATE_DIM = 24              # 状态向量维度 (需要仔细设计)
ACTION_DIM = 12             # 动作特征维度 (需要仔细设计)
EMBEDDING_DIM = 128
HIDDEN_DIM = 256

# --- 训练参数 ---
NUM_EPISODES = 100
MAX_STEPS_PER_CREW = 16     # 每个机组人员最多安排的任务数
MAX_CANDIDATE_ACTIONS = 200 # 每次决策的最大候选动作数

# --- 路径 ---
DATA_PATH = "data/"
MODEL_SAVE_PATH = "models/"

# --- 时间设定 ---
PLANNING_START_DATE = "2025-05-29 00:00:00"
PLANNING_END_DATE = "2025-06-05 00:00:00" # 结束于6月4日24:00