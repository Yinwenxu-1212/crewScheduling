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
MAX_POSITIONING_TOTAL = 6     # 整个规划期最多置位次数
PENALTY_EXCESS_DUTY_POSITIONING = -100.0   # 值勤内超额置位惩罚（从-200减轻到-100）
PENALTY_EXCESS_TOTAL_POSITIONING = -100.0  # 总超额置位惩罚（从-200减轻到-100）
PENALTY_SECOND_POSITIONING = -30.0         # 第二个置位的额外惩罚（从-80减轻到-30）
PENALTY_NO_FLIGHT_POSITIONING = -5.0      # 有航班可走时进行置位的惩罚（从-25降到-5）

# --- 置位战略奖励参数 ---
POSITIONING_STRATEGIC_BONUS = 10.0  # 新增：如果置位能连接高价值航班
POSITIONING_CHAIN_BONUS = 20.0  # 新增：置位-执行链奖励
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

# --- 强化学习超参数（优化版 - 快速训练）---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GAMMA = 0.99
LEARNING_RATE = 0.0003  # 提高学习率加速收敛
PPO_EPSILON = 0.1  # PPO裁剪参数
PPO_EPOCHS = 3  # 减少PPO训练轮数
KL_TARGET = 0.02  # KL散度约束目标值
ENTROPY_COEF = 0.008  # 熵系数，用于鼓励探索
VALUE_COEF = 0.5  # 价值损失系数
MAX_GRAD_NORM = 0.5  # 梯度裁剪
EARLY_STOP_PATIENCE = 10  # 早停机制
VALIDATION_INTERVAL = 5  # 验证间隔

# --- 新增：Epsilon-Greedy 探索参数 ---
EPSILON_START = 0.2  # 初始探索率 (20%的概率随机选择)
EPSILON_END = 0.005   # 最终探索率 (0.5%的概率随机选择)
EPSILON_DECAY_EPISODES = 5000 # 经过多少个episodes衰减到最终值

# --- 模型架构参数 ---
STATE_DIM = 17              # 状态特征维度（增加了全局特征）
ACTION_DIM = 20             # 动作特征维度，扩展支持伪对偶价格特征 (索引0-19)
HIDDEN_DIM = 256
NUM_HEADS = 8               # 多头注意力头数
NUM_LAYERS = 3              # Transformer层数
DROPOUT = 0.1               # Dropout率

# --- 训练参数（平衡速度与质量版本）---
NUM_EPISODES = 2  # 快速训练：只训练2轮
MAX_STEPS_PER_CREW = 10  # 平衡训练：每个机组10步，保证覆盖度
MAX_CANDIDATE_ACTIONS = 100  # 适中的候选动作数量，保证选择质量
UPDATE_INTERVAL = 3  # 适中的更新频率
MAX_TOTAL_STEPS = 8000  # 平衡训练：总步数6000步
STEP_PRINT_INTERVAL = 1  # 每1步打印一次

# --- 经验回放参数（平衡速度与质量版本）---
REPLAY_BUFFER_SIZE = 1000  # 平衡训练：适中的缓冲区大小
MIN_REPLAY_SIZE = 16  # 平衡训练：保证最小经验质量
BATCH_SIZE = 24  # 平衡训练：适中批次大小

# --- 课程学习参数 ---
CURRICULUM_UPDATE_INTERVAL = 1  # 每多少episode更新一次课程难度
SUCCESS_THRESHOLD = -1000  # 定义成功的奖励阈值

# --- 学习率调度参数 ---
LR_SCHEDULER_STEP_SIZE = 200
LR_SCHEDULER_GAMMA = 0.9
LR_DECAY_STEP = 200  # 学习率衰减步长
LR_DECAY_GAMMA = 0.9  # 学习率衰减因子

# --- 模型保存参数 ---
MODEL_SAVE_INTERVAL = 1  # 每多少episode保存一次模型
PRINT_DETAILS_INTERVAL = 1  # 每多少episode打印详细信息

# --- 探索策略参数 ---
EXPLORATION_PHASE_RATIO = 0.3  # 前30%的episode增加探索
LATE_EXPLORATION_PROB = 0.05  # 后期的探索概率

# --- 动态熵系数参数 ---
ENTROPY_COEF_START = 0.01
ENTROPY_COEF_END = 0.001
ENTROPY_DECAY_EPISODES = 1000

# --- 路径 ---
DATA_PATH = "data/"  # 数据路径，指向当前目录的data文件夹
MODEL_SAVE_PATH = "models/"

# --- 动态时间配置功能 ---
import pandas as pd
import os

def get_planning_dates_from_data(data_path=None):
    """从数据文件中自动获取规划时间范围"""
    if data_path is None:
        data_path = DATA_PATH
    
    try:
        # 读取航班数据获取时间范围
        flights_file = os.path.join(data_path, 'flight.csv')
        if os.path.exists(flights_file):
            flights_df = pd.read_csv(flights_file)
            flights_df['std'] = pd.to_datetime(flights_df['std'])
            flights_df['sta'] = pd.to_datetime(flights_df['sta'])
        else:
            flights_df = pd.DataFrame()
        
        # 读取占位任务数据
        ground_duties_file = os.path.join(data_path, 'groundDuty.csv')
        if os.path.exists(ground_duties_file):
            ground_duties_df = pd.read_csv(ground_duties_file)
            ground_duties_df['startTime'] = pd.to_datetime(ground_duties_df['startTime'])
            ground_duties_df['endTime'] = pd.to_datetime(ground_duties_df['endTime'])
        else:
            ground_duties_df = pd.DataFrame()
        
        # 读取大巴数据
        bus_file = os.path.join(data_path, 'busInfo.csv')
        if os.path.exists(bus_file):
            bus_df = pd.read_csv(bus_file)
            bus_df['td'] = pd.to_datetime(bus_df['td'])
            bus_df['ta'] = pd.to_datetime(bus_df['ta'])
        else:
            bus_df = pd.DataFrame()
        
        # 计算总体时间范围
        all_start_times = []
        all_end_times = []
        
        if not flights_df.empty:
            all_start_times.extend(flights_df['std'].tolist())
            all_end_times.extend(flights_df['sta'].tolist())
        
        if not ground_duties_df.empty:
            all_start_times.extend(ground_duties_df['startTime'].tolist())
            all_end_times.extend(ground_duties_df['endTime'].tolist())
        
        if not bus_df.empty:
            all_start_times.extend(bus_df['td'].tolist())
            all_end_times.extend(bus_df['ta'].tolist())
        
        if all_start_times and all_end_times:
            start_date = min(all_start_times).strftime('%Y-%m-%d %H:%M:%S')
            end_date = max(all_end_times).strftime('%Y-%m-%d %H:%M:%S')
            return start_date, end_date
        else:
            raise ValueError("未找到有效的时间数据")
        
    except Exception as e:
        print(f"无法从数据中获取时间范围: {e}")
        # 返回默认值
        return "2025-04-29 00:00:00", "2025-05-07 23:59:59"

# --- 时间设定（支持动态检测）---
# 尝试从数据中自动检测时间范围
try:
    PLANNING_START_DATE, PLANNING_END_DATE = get_planning_dates_from_data()
    print(f"✓ 自动检测到规划时间: {PLANNING_START_DATE} 到 {PLANNING_END_DATE}")
except Exception as e:
    # 如果自动检测失败，使用默认值
    PLANNING_START_DATE = "2025-04-29 00:00:00"
    PLANNING_END_DATE = "2025-05-07 23:59:59"
    print(f"⚠ 使用默认规划时间: {PLANNING_START_DATE} 到 {PLANNING_END_DATE}")
    print(f"  检测失败原因: {e}")

# 提供手动设置时间的函数
def set_planning_dates(start_date, end_date):
    """手动设置规划时间范围"""
    global PLANNING_START_DATE, PLANNING_END_DATE
    PLANNING_START_DATE = start_date
    PLANNING_END_DATE = end_date
    print(f"✓ 手动设置规划时间: {PLANNING_START_DATE} 到 {PLANNING_END_DATE}")

# 提供更新数据路径和重新检测时间的函数
def update_data_path_and_detect_time(new_data_path):
    """更新数据路径并重新检测时间"""
    global DATA_PATH, PLANNING_START_DATE, PLANNING_END_DATE
    DATA_PATH = new_data_path
    try:
        PLANNING_START_DATE, PLANNING_END_DATE = get_planning_dates_from_data(new_data_path)
        print(f"✓ 更新数据路径为: {DATA_PATH}")
        print(f"✓ 重新检测到规划时间: {PLANNING_START_DATE} 到 {PLANNING_END_DATE}")
        return True
    except Exception as e:
        print(f"⚠ 更新数据路径失败: {e}")
        return False