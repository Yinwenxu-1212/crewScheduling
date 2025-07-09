# config.py

import torch

# --- 竞赛评价指标权重 ---
SCORE_FLY_TIME_MULTIPLIER = 20  # 大幅降低飞行时间权重，避免过度偏向飞行任务
PENALTY_UNCOVERED_FLIGHT = -100
PENALTY_OVERNIGHT_STAY_AWAY_FROM_BASE = -1.0
PENALTY_POSITIONING = -3.0  # 基础置位惩罚（从-8降到-3，进一步鼓励置位覆盖占位任务）
PENALTY_POSITIONING_MIDDLE = -10.0  # 值勤日中间置位的惩罚（从-20降到-10）
PENALTY_RULE_VIOLATION = -50.0  # 增加规则违规惩罚

# --- 置位任务限制参数（放宽约束以提高占位任务覆盖率）---
MAX_POSITIONING_PER_DUTY = 2  # 每个值勤最多置位次数（从1增加到2）
MAX_POSITIONING_TOTAL = 8     # 整个规划期最多置位次数
PENALTY_EXCESS_DUTY_POSITIONING = -100.0   # 值勤内超额置位惩罚（从-200减轻到-100）
PENALTY_EXCESS_TOTAL_POSITIONING = -100.0  # 总超额置位惩罚（从-200减轻到-100）
PENALTY_SECOND_POSITIONING = -30.0         # 第二个置位的额外惩罚（从-80减轻到-30）
PENALTY_NO_FLIGHT_POSITIONING = -5.0      # 有航班可走时进行置位的惩罚（从-25降到-5）

# --- 置位战略奖励参数 ---
POSITIONING_STRATEGIC_BONUS = 20.0  # 新增：如果置位能连接高价值航班
POSITIONING_CHAIN_BONUS = 30.0  # 新增：置位-执行链奖励
POSITIONING_COVERAGE_BONUS = 8.0  # 新增：置位覆盖难点区域奖励

# --- 动态优先级权重配置 ---
PRIORITY_WEIGHTS = {
    'flight_base': 100.0,                    # 航班任务基础分数
    'ground_duty_base': 150.0,               # 占位任务基础分数
    'ground_duty_exclusive': 80.0,           # 专属占位任务额外分数
    'ground_duty_risk_bonus': 20.0,          # 高风险占位任务额外分数
    'ground_duty_workload_balance': 15.0,    # 工作负载平衡奖励
    'positioning_base': 53.5,                # 置位任务基础分数
    'positioning_high_value': 50.0,          # 高价值置位额外分数（从35.0提高到50.0）
    'positioning_medium_value': 30.0,        # 中等价值置位额外分数（从20.0提高到30.0）
    'positioning_low_value': 15.0,           # 低价值置位额外分数（从10.0提高到15.0）
    'time_urgency_multiplier': 30.0,         # 时间紧迫性乘数
    'connection_bonus': 15.0,                # 连接性奖励
    'scarcity_multiplier': 40.0              # 稀缺性乘数
}

# --- 奖励塑造参数 ---
IMMEDIATE_COVERAGE_REWARD = 15.0  # 进一步降低飞行任务奖励
PENALTY_PASS_ACTION = -2.0
GROUND_DUTY_COVERAGE_REWARD = 200.0  # 进一步提高占位任务覆盖奖励，确保优先级
GROUND_DUTY_PRIORITY_BONUS = 30.0  # 增加占位任务优先级奖励
CRITICAL_GROUND_DUTY_BONUS = 25.0   # 增加关键时段占位任务额外奖励

# --- 强化学习超参数（快速训练优化版）---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GAMMA = 0.995  # 稍微增加折扣因子，更重视长期奖励
LEARNING_RATE = 0.0005  # 进一步提高学习率，加快2轮训练收敛
PPO_EPSILON = 0.15  # 稍微增加PPO裁剪参数
PPO_EPOCHS = 3  # 适度减少PPO轮数，提高训练速度
KL_TARGET = 0.015  # 稍微降低KL散度约束
ENTROPY_COEF = 0.015  # 适度增加熵系数，平衡探索和利用
VALUE_COEF = 0.6  # 增加价值损失系数
MAX_GRAD_NORM = 1.0  # 增加梯度裁剪阈值
EARLY_STOP_PATIENCE = 3  # 早停耐心，适合5轮训练
VALIDATION_INTERVAL = 1  # 验证间隔，适合5轮训练

# --- 新增：Epsilon-Greedy 探索参数 ---
EPSILON_START = 0.3  # 初始探索率 (30%的概率随机选择)
EPSILON_END = 0.01   # 最终探索率 (1%的概率随机选择)
EPSILON_DECAY_EPISODES = 3 # 经过3个episodes衰减到最终值（适合5轮训练）

# --- 模型架构参数 ---
STATE_DIM = 17              # 状态特征维度（增加了全局特征）
ACTION_DIM = 20             # 动作特征维度，扩展支持伪对偶价格特征 (索引0-19)
HIDDEN_DIM = 128
NUM_HEADS = 8               # 多头注意力头数
NUM_LAYERS = 2              # Transformer层数
DROPOUT = 0.1               # Dropout率

# --- 训练参数（2轮训练版本）---
NUM_EPISODES = 2  # 2轮训练
MAX_STEPS_PER_CREW = 12  # 适度增加每个机组的步数
MAX_CANDIDATE_ACTIONS = 60  # 平衡候选动作数量和计算效率
UPDATE_INTERVAL = 2  # 更频繁的更新
MAX_TOTAL_STEPS = 8000  # 大规模训练：总步数8000步
STEP_PRINT_INTERVAL = 20  # 每20步打印一次，减少输出提高效率

# --- 经验回放参数（2轮训练优化）---
REPLAY_BUFFER_SIZE = 1500  # 适中缓冲区大小，平衡内存和效果
MIN_REPLAY_SIZE = 50  # 积累一定经验后开始学习
BATCH_SIZE = 96  # 优化批次大小，提高训练效率

# --- 课程学习参数（2轮训练优化）---
CURRICULUM_UPDATE_INTERVAL = 1  # 每个episode更新课程难度
SUCCESS_THRESHOLD = -300  # 适中的成功阈值

# --- 学习率调度参数 ---
LR_SCHEDULER_STEP_SIZE = 2  # 学习率调度步长（适合5轮训练）
LR_SCHEDULER_GAMMA = 0.9  # 调整衰减因子
LR_DECAY_STEP = 2  # 学习率衰减步长（适合5轮训练）
LR_DECAY_GAMMA = 0.9  # 学习率衰减因子

# --- 模型保存参数 ---
MODEL_SAVE_INTERVAL = 1  # 每1个episode保存一次模型（适合5轮训练）
PRINT_DETAILS_INTERVAL = 1  # 每1个episode打印详细信息（适合5轮训练）

# --- 探索策略参数 ---
EXPLORATION_PHASE_RATIO = 0.4  # 前40%的episode增加探索，适合长期训练
LATE_EXPLORATION_PROB = 0.02  # 后期的探索概率，更保守

# --- 动态熵系数参数 ---
ENTROPY_COEF_START = 0.015  # 稍微增加初始熵系数
ENTROPY_COEF_END = 0.0005  # 降低最终熵系数
ENTROPY_DECAY_EPISODES = 3  # 衰减周期，适合5轮训练

# --- 路径 ---
import os
# 动态获取数据路径，确保路径正确
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(current_dir, "..", "data")  # 数据路径，指向上级目录的data文件夹
MODEL_SAVE_PATH = "models/"

# --- 时间设定 ---
# 使用数据配置模块动态获取时间范围
try:
    from data_config import get_planning_dates_from_data, get_data_config
    
    # 动态获取规划时间范围
    PLANNING_START_DATE, PLANNING_END_DATE = get_planning_dates_from_data()
    
    # 打印数据摘要（可选）
    data_config = get_data_config()
    data_config.print_data_summary()
    
except ImportError as e:
    print(f"Warning: Could not import data_config module: {e}")
    # 使用默认值
    PLANNING_START_DATE = "2025-04-29 00:00:00"
    PLANNING_END_DATE = "2025-05-07 23:59:59"
    print(f"Using default planning period: {PLANNING_START_DATE} to {PLANNING_END_DATE}")
except Exception as e:
    print(f"Warning: Error getting planning dates: {e}")
    # 使用默认值
    PLANNING_START_DATE = "2025-04-29 00:00:00"
    PLANNING_END_DATE = "2025-05-07 23:59:59"
    print(f"Using default planning period: {PLANNING_START_DATE} to {PLANNING_END_DATE}")