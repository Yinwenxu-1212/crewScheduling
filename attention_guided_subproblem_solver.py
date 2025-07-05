import heapq
import itertools
from datetime import datetime, timedelta, date
from data_models import Crew, Flight, BusInfo, GroundDuty, Node, Roster, RestPeriod
from typing import List, Dict, Set, Optional, Tuple
import csv
import os
import torch
import numpy as np
import random
from attention.model import ActorCritic
from attention import config as attention_config
from scoring_system import ScoringSystem
from constraint_checker import UnifiedConstraintChecker
from collections import defaultdict
import time

# 导入统一配置
from unified_config import config

# 使用统一配置的参数
optimization_params = config.get_optimization_params()
REWARD_PER_FLIGHT_HOUR = -optimization_params['flight_time_reward']  # 注意：子问题中使用负值
PENALTY_PER_AWAY_OVERNIGHT = optimization_params['away_overnight_penalty']
PENALTY_PER_POSITIONING = optimization_params['positioning_penalty']

# 从统一配置获取约束参数
constraint_params = config.get_constraint_params()
MIN_REST_HOURS = constraint_params['min_rest_hours']
MAX_DUTY_DAY_HOURS = constraint_params['max_duty_day_hours']
MAX_FLIGHT_TIME_IN_DUTY_HOURS = constraint_params['max_flight_time_in_duty_hours']

# 添加总飞行时间约束常量
MAX_TOTAL_FLIGHT_HOURS = 60.0  # 计划期内总飞行时间上限（小时）

# 连接时间常量（从统一配置获取，转换为timedelta）
MIN_CONNECTION_TIME_FLIGHT_SAME_AIRCRAFT = timedelta(minutes=config.MIN_CONNECTION_TIME_FLIGHT_SAME_AIRCRAFT_MINUTES)
MIN_CONNECTION_TIME_FLIGHT_DIFFERENT_AIRCRAFT = timedelta(hours=config.MIN_CONNECTION_TIME_FLIGHT_DIFFERENT_AIRCRAFT_HOURS)
MIN_CONNECTION_TIME_BUS = timedelta(hours=config.MIN_CONNECTION_TIME_BUS_HOURS)

# 从data_models导入Label类
from data_models import Label

# ===== 性能优化组件 =====
class ConvergenceManager:
    """智能收敛管理器"""
    
    def __init__(self, improvement_threshold=1e-6, stagnation_limit=5, min_iterations=5):
        self.obj_history = []
        self.improvement_threshold = improvement_threshold
        self.stagnation_limit = stagnation_limit
        self.min_iterations = min_iterations
        self.roster_count_history = []
        
    def should_terminate(self, current_obj, new_rosters_count, iteration):
        """智能收敛判断 - 修复过早终止问题"""
        self.obj_history.append(current_obj)
        self.roster_count_history.append(new_rosters_count)
        
        # 增加最少迭代次数保证，确保充分搜索
        min_required_iterations = max(self.min_iterations, 10)  # 至少10次迭代
        if len(self.obj_history) < min_required_iterations:
            return False
        
        # 在早期迭代中更宽松的终止条件
        if iteration < 20:  # 前20次迭代不轻易终止
            return False
        
        # 检查目标函数改善 - 放宽条件
        if len(self.obj_history) >= 3:  # 需要更多历史数据
            recent_improvements = [
                self.obj_history[i] - self.obj_history[i-1] 
                for i in range(-2, 0)  # 最近2次改善
            ]
            
            # 只有连续多次无改善且无新roster才考虑终止
            all_no_improvement = all(imp < self.improvement_threshold for imp in recent_improvements)
            recent_no_rosters = sum(self.roster_count_history[-3:]) == 0  # 最近3轮无roster
            
            if all_no_improvement and recent_no_rosters:
                return True
        
        # 检查长期停滞 - 增加停滞轮数要求
        extended_stagnation_limit = max(self.stagnation_limit, 8)  # 至少8轮停滞
        if len(self.obj_history) >= extended_stagnation_limit:
            recent_objs = self.obj_history[-extended_stagnation_limit:]
            recent_max = max(recent_objs)
            recent_min = min(recent_objs)
            
            # 目标函数变化很小
            if recent_max - recent_min < self.improvement_threshold:
                # 同时检查roster生成情况 - 更严格的条件
                recent_rosters = sum(self.roster_count_history[-extended_stagnation_limit:])
                if recent_rosters == 0:  # 完全没有新roster
                    return True
        
        return False

class TaskIndexManager:
    """任务索引管理器 - 高效的任务查找和过滤"""
    
    def __init__(self):
        self.tasks_by_time_hour = defaultdict(list)
        self.tasks_by_location = defaultdict(list)
        self.tasks_by_day = defaultdict(list)
        self.tasks_by_type = defaultdict(list)
        self.eligible_tasks_cache = {}
        self.all_tasks = []
        
    def preprocess_tasks(self, all_tasks):
        """预处理任务，建立多维索引"""
        self.all_tasks = all_tasks
        
        for task in all_tasks:
            task_start = task['startTime']
            
            # 按小时索引
            hour_key = task_start.hour
            self.tasks_by_time_hour[hour_key].append(task)
            
            # 按日期索引
            date_key = task_start.date()
            self.tasks_by_day[date_key].append(task)
            
            # 按出发机场索引
            depa_airport = task['depaAirport']
            self.tasks_by_location[depa_airport].append(task)
            
            # 按任务类型索引
            task_type = task['type']
            self.tasks_by_type[task_type].append(task)
    
    def get_time_filtered_tasks(self, current_time, time_window_hours=48):
        """获取时间窗口内的任务"""
        candidates = []
        end_time = current_time + timedelta(hours=time_window_hours)
        
        # 按日期快速过滤
        current_date = current_time.date()
        end_date = end_time.date()
        
        date = current_date
        while date <= end_date:
            if date in self.tasks_by_day:
                for task in self.tasks_by_day[date]:
                    if current_time <= task['startTime'] <= end_time:
                        candidates.append(task)
            date += timedelta(days=1)
        
        return candidates
    
    def get_candidates_optimized(self, current_label, crew, time_window_hours=48):
        """优化的候选任务获取"""
        current_time = current_label.node.time
        current_airport = current_label.node.airport
        used_task_ids = current_label.used_task_ids
        
        # 构建缓存键
        cache_key = (
            current_airport,
            int(current_time.timestamp()) // 3600,  # 小时级别
            len(used_task_ids),
            bool(current_label.duty_start_time)
        )
        
        # 检查缓存
        if cache_key in self.eligible_tasks_cache:
            cached_candidates = self.eligible_tasks_cache[cache_key]
            # 过滤已使用的任务
            return [task for task in cached_candidates if task['taskId'] not in used_task_ids]
        
        # 第一步：时间过滤
        time_candidates = self.get_time_filtered_tasks(current_time, time_window_hours)
        
        # 第二步：地点过滤（包含可达性检查）
        reachable_airports = self._get_reachable_airports(current_airport, current_time)
        location_candidates = [task for task in time_candidates 
                             if task['depaAirport'] in reachable_airports]
        
        # 第三步：基本可行性过滤
        feasible_candidates = []
        for task in location_candidates:
            if self._basic_feasibility_check(task, current_label, crew):
                feasible_candidates.append(task)
        
        # 缓存结果
        self.eligible_tasks_cache[cache_key] = feasible_candidates
        
        # 最终过滤已使用的任务
        final_candidates = [task for task in feasible_candidates if task['taskId'] not in used_task_ids]
        
        return final_candidates
    
    def _get_reachable_airports(self, current_airport, current_time):
        """获取可达机场（包括需要置位的）"""
        reachable = {current_airport}
        
        # 检查是否可以通过置位到达其他机场
        positioning_tasks = self.tasks_by_type.get('positioning_bus', [])
        
        for pos_task in positioning_tasks:
            if (pos_task['depaAirport'] == current_airport and 
                pos_task['startTime'] >= current_time):
                reachable.add(pos_task['arriAirport'])
        
        return reachable
    
    def _basic_feasibility_check(self, task, current_label, crew):
        """基本可行性检查"""
        current_time = current_label.node.time
        
        # 时间检查
        if task['startTime'] <= current_time:
            return False
        
        # 连接时间检查
        connection_time = task['startTime'] - current_time
        if connection_time < timedelta(minutes=30):  # 最小连接时间
            return False
        
        # 值勤日基本检查
        if current_label.duty_start_time:
            # 检查值勤日长度
            potential_duty_end = task['endTime']
            duty_length = potential_duty_end - current_label.duty_start_time
            if duty_length > timedelta(hours=12):  # 最大值勤时间
                return False
            
            # 检查任务数量
            if current_label.duty_task_count >= 6:  # 最大任务数
                return False
            
            # 检查飞行任务数量
            if (task['type'] == 'flight' and 
                current_label.duty_flight_count >= 4):  # 最大飞行任务数
                return False
        
        return True
    
    def clear_cache(self):
        """清理缓存"""
        self.eligible_tasks_cache.clear()

class StateKeyOptimizer:
    """状态键优化器"""
    
    @staticmethod
    def get_compact_state_key(current_label):
        """生成紧凑的状态键"""
        return (
            hash(current_label.node.airport) % 10000,  # 机场哈希压缩
            int(current_label.node.time.timestamp()) // 3600,  # 小时级精度
            len(current_label.used_task_ids),  # 任务数量而非完整集合
            bool(current_label.duty_start_time),  # 是否在值勤中
            current_label.duty_flight_count,
            int(current_label.total_flight_hours),  # 整数小时数
            current_label.current_cycle_days,  # 飞行周期天数
            current_label.duty_days_count  # 值勤日数量
        )

class MemoryManager:
    """内存管理器"""
    
    def __init__(self, max_visited_states=100000, cleanup_interval=1000):
        self.max_visited_states = max_visited_states
        self.cleanup_interval = cleanup_interval
        self.cleanup_counter = 0
        
    def should_cleanup(self, visited_set):
        """判断是否需要清理内存"""
        self.cleanup_counter += 1
        
        return (len(visited_set) > self.max_visited_states or 
                self.cleanup_counter % self.cleanup_interval == 0)
    
    def cleanup_visited_states(self, visited_set, keep_ratio=0.7):
        """清理访问状态集合，保留最近的状态"""
        if len(visited_set) <= self.max_visited_states * keep_ratio:
            return visited_set
        
        # 简单策略：随机保留一部分状态
        states_list = list(visited_set)
        keep_count = int(len(states_list) * keep_ratio)
        
        random.shuffle(states_list)
        new_visited = set(states_list[:keep_count])
        
        return new_visited

class AttentionGuidedSubproblemSolver:
    """使用注意力模型指导的子问题求解器"""
    
    def __init__(self, model_path: str = "models/best_model.pth", debug=False, layover_stations_set=None):
        """初始化求解器并加载预训练的注意力模型"""
        self.debug = debug
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 增大搜索参数以提高覆盖率
        self.max_iterations = config.MAX_SUBPROBLEM_ITERATIONS  # 使用统一配置的子问题迭代次数
        self.beam_width = config.BEAM_WIDTH  # 使用统一配置的beam search宽度
        
        # 约束参数
        self.MAX_DUTY_DAY_HOURS = 12.0
        self.MAX_FLIGHT_TIME_IN_DUTY_HOURS = 8.0
        self.MIN_REST_HOURS = 12.0
        
        # 初始化统一约束检查器
        self.layover_stations_set = layover_stations_set or set()
        self.constraint_checker = UnifiedConstraintChecker(self.layover_stations_set)
        
        # 优化9: 初始化缓存机制
        self._positioning_cache = {}  # 置位任务缓存
        self._constraint_cache = {}   # 约束检查缓存
        self._cache_hits = 0          # 缓存命中计数
        self._cache_misses = 0        # 缓存未命中计数
        
        # 初始化调试日志文件
        debug_dir = "debug"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        
        debug_log_file = os.path.join(debug_dir, "attention_solver_debug.log")
        try:
            # 使用追加模式，避免覆盖之前机组的日志
            self.debug_log = open(debug_log_file, 'a', encoding='utf-8')
            self.debug_log.write(f"\n=== 新的Solver实例启动 ===\n")
            self.debug_log.write(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.debug_log.flush()
        except Exception as e:
            print(f"无法创建调试日志文件: {e}")
            self.debug_log = None
        
        # 加载预训练的注意力模型
        self.model = ActorCritic(
            state_dim=attention_config.STATE_DIM,
            action_dim=attention_config.ACTION_DIM,
            hidden_dim=attention_config.HIDDEN_DIM
        ).to(self.device)
        
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 适配动作特征权重维度
            if 'actor_action_encoder.0.weight' in checkpoint:
                old_weight = checkpoint['actor_action_encoder.0.weight']  # [256, old_dim]
                if old_weight.shape[1] != attention_config.ACTION_DIM:
                    if old_weight.shape[1] > attention_config.ACTION_DIM:
                        # 截取前N维，保留最重要的特征
                        checkpoint['actor_action_encoder.0.weight'] = old_weight[:, :attention_config.ACTION_DIM]
                        if self.debug_log:
                            self.debug_log.write(f"动作特征维度适配: {old_weight.shape[1]} -> {attention_config.ACTION_DIM}\n")
                    else:
                        # 如果旧维度小于新维度，用零填充
                        new_weight = torch.zeros(old_weight.shape[0], attention_config.ACTION_DIM)
                        new_weight[:, :old_weight.shape[1]] = old_weight
                        checkpoint['actor_action_encoder.0.weight'] = new_weight
                        if self.debug_log:
                            self.debug_log.write(f"动作特征维度扩展: {old_weight.shape[1]} -> {attention_config.ACTION_DIM}\n")
            
            # 适配状态特征权重维度
            if 'actor_state_encoder.0.weight' in checkpoint:
                old_weight = checkpoint['actor_state_encoder.0.weight']  # [256, old_dim]
                if old_weight.shape[1] != attention_config.STATE_DIM:
                    if old_weight.shape[1] > attention_config.STATE_DIM:
                        # 截取前N维
                        checkpoint['actor_state_encoder.0.weight'] = old_weight[:, :attention_config.STATE_DIM]
                        if self.debug_log:
                            self.debug_log.write(f"状态特征维度适配: {old_weight.shape[1]} -> {attention_config.STATE_DIM}\n")
                    else:
                        # 用零填充
                        new_weight = torch.zeros(old_weight.shape[0], attention_config.STATE_DIM)
                        new_weight[:, :old_weight.shape[1]] = old_weight
                        checkpoint['actor_state_encoder.0.weight'] = new_weight
                        if self.debug_log:
                            self.debug_log.write(f"状态特征维度扩展: {old_weight.shape[1]} -> {attention_config.STATE_DIM}\n")
            
            # 同样处理critic网络的状态编码器
            if 'critic.0.weight' in checkpoint:
                old_weight = checkpoint['critic.0.weight']  # [256, old_dim]
                if old_weight.shape[1] != attention_config.STATE_DIM:
                    if old_weight.shape[1] > attention_config.STATE_DIM:
                        checkpoint['critic.0.weight'] = old_weight[:, :attention_config.STATE_DIM]
                    else:
                        new_weight = torch.zeros(old_weight.shape[0], attention_config.STATE_DIM)
                        new_weight[:, :old_weight.shape[1]] = old_weight
                        checkpoint['critic.0.weight'] = new_weight
            
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            if self.debug_log:
                self.debug_log.write(f"成功加载预训练模型: {model_path}\n")
        else:
            if self.debug_log:
                self.debug_log.write(f"警告：未找到预训练模型 {model_path}，使用随机初始化的模型\n")
        
        # 注意力引导的参数
        self.max_candidates_per_expansion = 8  # 每次扩展最多考虑的候选任务数
        self.use_attention_guidance = True
        
        # 初始化优化组件
        self.convergence_manager = ConvergenceManager(
            improvement_threshold=getattr(config, 'CONVERGENCE_THRESHOLD', 1e-4),
            stagnation_limit=getattr(config, 'STAGNATION_LIMIT', 3),
            min_iterations=getattr(config, 'MIN_ITERATIONS', 2)
        )
        self.task_index_manager = TaskIndexManager()
        self.state_key_optimizer = StateKeyOptimizer()
        self.memory_manager = MemoryManager(
            max_visited_states=getattr(config, 'MAX_VISITED_STATES', 100000),
            cleanup_interval=getattr(config, 'CLEANUP_INTERVAL', 1000)
        )
    
    def __del__(self):
        """析构函数，关闭日志文件"""
        if hasattr(self, 'debug_log') and self.debug_log:
            self.debug_log.close()
    
    def _log_debug(self, message: str):
        """写入调试信息到日志文件"""
        if hasattr(self, 'debug_log') and self.debug_log:
            self.debug_log.write(f"{message}\n")
            self.debug_log.flush()  # 立即刷新到文件
        if self.debug:
            print(message)
    
    def _extract_state_features(self, label: Label, crew: Crew) -> np.ndarray:
        """从当前标签状态提取状态特征向量"""
        features = np.zeros(attention_config.STATE_DIM)
        
        # 时间特征
        current_time = label.node.time
        features[0] = current_time.weekday()  # 星期几
        features[1] = current_time.hour  # 小时
        features[2] = current_time.day  # 日期
        
        # 添加调试信息和类型检查
        if not hasattr(label.node, 'airport'):
            print(f"Error: label.node does not have 'airport' attribute. Type: {type(label.node)}, Value: {label.node}")
            features[3] = 0
        elif isinstance(label.node.airport, str):
            # 位置特征（机场哈希）
            features[3] = hash(label.node.airport) % 1000
        else:
            print(f"Warning: airport is not a string. Type: {type(label.node.airport)}, Value: {label.node.airport}")
            features[3] = 0
        
        # 值勤状态特征
        if label.duty_start_time:
            duty_duration = (current_time - label.duty_start_time).total_seconds() / 3600
            features[4] = min(duty_duration, 24)  # 当前值勤时长（小时）
            features[5] = label.duty_flight_time  # 值勤内飞行时间
            features[6] = label.duty_flight_count  # 值勤内航班数
            features[7] = label.duty_task_count  # 值勤内任务数
        
        # 累计资源特征
        features[8] = label.total_flight_hours  # 总飞行时间
        features[9] = label.total_positioning  # 总调机次数
        features[10] = label.total_away_overnights  # 总外站过夜
        features[11] = len(label.total_calendar_days)  # 总日历天数
        
        # 成本特征
        features[12] = label.cost / 1000.0  # 归一化成本
        
        # 机组基地特征
        features[13] = 1 if label.node.airport == crew.base else 0
        
        return features
    
    def _extract_task_features(self, task, current_label: Label) -> np.ndarray:
        """提取任务特征向量"""
        features = np.zeros(attention_config.ACTION_DIM)
        
        # 连接时间
        connection_time = (task['startTime'] - current_label.node.time).total_seconds() / 3600
        features[0] = min(connection_time, 48)  # 限制在48小时内
        
        # 任务类型特征
        if task['type'] == 'flight':
            features[1] = 1
            features[2] = task.get('flyTime', 0) / 60.0  # 飞行时间（小时）
        elif 'positioning' in task.get('type', ''):
            features[3] = 1
            if 'bus' in task.get('type', ''):
                features[4] = 1  # 巴士调机
            else:
                features[5] = 1  # 飞行调机
        elif task['type'] == 'ground_duty':
            features[6] = 1  # 占位任务特征
        
        # 机场特征 - 修正索引
        features[7] = hash(task['depaAirport']) % 1000   # 原来是features[6]
        features[8] = hash(task['arriAirport']) % 1000   # 原来是features[7]
        
        # 时间特征 - 相应调整索引
        features[9] = task['startTime'].weekday()        # 原来是features[8]
        features[10] = task['startTime'].hour            # 原来是features[9]
        features[11] = task['endTime'].hour              # 原来是features[10]
        
        # 任务持续时间
        duration = (task['endTime'] - task['startTime']).total_seconds() / 3600
        features[12] = min(duration, 24)                 # 原来是features[11]
        
        return features
    
    def _score_candidates_with_attention(self, candidates: List[Dict], 
                                       current_label: Label, crew: Crew) -> List[Tuple[float, int]]:
        """使用注意力模型为候选任务评分"""
        try:
            if not self.use_attention_guidance or len(candidates) == 0:
                return [(0.0, i) for i in range(len(candidates))]
            
            # 提取状态特征
            state_features = self._extract_state_features(current_label, crew)
            
            # 为所有候选任务提取特征
            candidate_features = []
            for task in candidates:
                task_features = self._extract_task_features(task, current_label)
                candidate_features.append(task_features)
            
            # 转换为张量
            state_tensor = torch.FloatTensor(state_features).unsqueeze(0).to(self.device)  # (1, state_dim)
            # 先转换为numpy数组再转换为张量，避免效率警告
            candidate_features_array = np.array(candidate_features)
            candidates_tensor = torch.FloatTensor(candidate_features_array).unsqueeze(0).to(self.device)  # (1, num_candidates, action_dim)
            action_mask = torch.ones(1, len(candidates)).to(self.device)  # (1, num_candidates) - 所有候选都有效
            
            # 使用注意力模型评分
            with torch.no_grad():
                dist, _ = self.model(state_tensor, candidates_tensor, action_mask)
                action_probs = dist.probs.squeeze(0).cpu().numpy()  # (num_candidates,)
            
            # 返回 (分数, 索引) 的列表，按分数降序排序
            scored_candidates = [(float(action_probs[i]), i) for i in range(len(candidates))]
            scored_candidates.sort(reverse=True, key=lambda x: x[0])
            
            if self.debug:
                print("=== Attention 模型评分 ===")
                for score, idx in scored_candidates[:5]:
                    print(f"Score: {score:.4f}, TaskID: {candidates[idx]['taskId']}, Type: {candidates[idx]['type']}")

            return scored_candidates
            
        except Exception as e:
            print(f"Warning: Attention scoring failed: {e}. Using deterministic order.")
            # 使用确定性排序而不是随机排序
            return [(0.0, i) for i in range(len(candidates))]
    
    def solve_subproblem_with_attention(self, crew: Crew, flights: List[Flight],
                                      buses: List[BusInfo], ground_duties: List[GroundDuty],
                                      dual_prices: Dict[str, float], 
                                      planning_start_dt: datetime, planning_end_dt: datetime,
                                      layover_airports: Set[str], crew_sigma_dual: float, ground_duty_duals: Dict[str, float], iteration_round: int = 0, external_log_func=None) -> List[Roster]:
        """使用注意力模型指导的子问题求解"""
        
        # 初始化
        found_rosters = []
        labels = []
        visited = set()
        tie_breaker = itertools.count()
        
        # 创建初始标签
        initial_node = Node(crew.stayStation, planning_start_dt)  # 使用stayStation而不是stay_station
        # 简化初始成本计算，与原始solver保持一致
        initial_cost = -crew_sigma_dual  # 直接使用crew_sigma_dual，不乘以不存在的常量
        initial_label = Label(
            cost=initial_cost, path=[], current_node=initial_node,
            duty_start_time=None, duty_flight_time=0.0,
            duty_flight_count=0, duty_task_count=0,
            total_flight_hours=0.0, total_positioning=0,
            total_away_overnights=0, total_calendar_days=set(),
            has_flown_in_duty=False, used_task_ids=set(),
            tie_breaker=next(tie_breaker),
            current_cycle_start=None, current_cycle_days=0,
            last_base_return=planning_start_dt.date(),
            duty_days_count=0  # 初始值勤日数量为0
        )
        
        heapq.heappush(labels, (0.0, initial_label))
        
        # 准备任务数据时确保使用最新的对偶价格
        all_tasks = []
        
        # 添加航班任务 - 为每个航班创建执行和置位两种任务类型
        for flight in flights:
            # 确保使用当前迭代的对偶价格
            current_dual_price = dual_prices.get(flight.id, 0.0)
            
            # 1. 执行航班任务（有飞行时间奖励和对偶价格收益）
            execution_task = {
                'type': 'flight',
                'taskId': flight.id,
                'startTime': flight.std,
                'endTime': flight.sta,
                'depaAirport': flight.depaAirport,
                'arriAirport': flight.arriAirport,
                'flyTime': flight.flyTime,
                'aircraftNo': flight.aircraftNo,
                'dual_price': current_dual_price,  # 执行航班有对偶价格收益
                'is_positioning': False  # 标记为执行任务
            }
            all_tasks.append(execution_task)
            
            # 2. 置位航班任务（无飞行时间奖励，有置位惩罚）
            positioning_task = {
                'type': 'positioning_flight',
                'taskId': flight.id + '_pos',  # 添加后缀区分置位任务
                'original_flight_id': flight.id,  # 保存原始航班ID
                'startTime': flight.std,
                'endTime': flight.sta,
                'depaAirport': flight.depaAirport,
                'arriAirport': flight.arriAirport,
                'flyTime': flight.flyTime,
                'aircraftNo': flight.aircraftNo,
                'dual_price': 0.0,  # 置位航班无对偶价格收益
                'is_positioning': True  # 标记为置位任务
            }
            all_tasks.append(positioning_task)
        
        # 添加巴士任务
        for bus in buses:
            task_dict = {
                'type': 'positioning_bus',
                'taskId': bus.id,
                'startTime': bus.td,
                'endTime': bus.ta,
                'depaAirport': bus.depaAirport,
                'arriAirport': bus.arriAirport,
                'dual_price': 0.0
            }
            all_tasks.append(task_dict)
        
        # 添加占位任务
        for ground_duty in ground_duties:
            # 使用传入的占位任务对偶价格
            current_dual_price = ground_duty_duals.get(ground_duty.id, 0.0)
            task_dict = {
                'type': 'ground_duty',
                'taskId': ground_duty.id,
                'startTime': ground_duty.startTime,
                'endTime': ground_duty.endTime,
                'depaAirport': ground_duty.airport,
                'arriAirport': ground_duty.airport,  # 占位任务起降机场相同
                'dual_price': current_dual_price
            }
            all_tasks.append(task_dict)
        
        # 主循环
        iteration_count = 0
        # 动态调整搜索参数，基于迭代轮次增加多样性
        # max_iterations 在上面根据迭代轮次设置
        
        # 根据迭代轮次调整搜索参数
        if iteration_round == 0:  # 第一轮
            max_valuable_rosters = min(len(all_tasks), 100)
            self.max_candidates_per_expansion = 8
            max_iterations = self.max_iterations
        else:
            max_valuable_rosters = min(len(all_tasks), 150)
            self.max_candidates_per_expansion = 12
            max_iterations = self.max_iterations
        
        # 添加随机种子扰动，确保每轮生成不同结果
        random.seed(42 + iteration_round * 17 + hash(crew.crewId) % 1000)
        
        # 添加已找到方案的记录
        found_roster_signatures = set()
        
        # 添加调试计数器
        total_candidates_found = 0
        total_labels_processed = 0
        
        # 添加路径多样性跟踪
        path_signatures = set()  # 记录已探索的路径特征
        diversity_threshold = max(5, iteration_round * 2)  # 多样性阈值
        
        self._log_debug(f"\n=== 机组 {crew.crewId} 子问题求解开始 (第{iteration_round+1}轮) ===")
        self._log_debug(f"初始状态: 队列={len(labels)}, 任务={len(all_tasks)}")
        self._log_debug(f"多样性设置: 候选数={self.max_candidates_per_expansion}, 阈值={diversity_threshold}")
        
        # 基本循环条件 - 添加智能收敛判断
        last_roster_count = 0
        while (labels and 
               iteration_count < max_iterations and 
               len(found_rosters) < max_valuable_rosters):
            
            # 检查智能收敛条件 - 修复过早检查问题
            current_obj = -len(found_rosters)  # 简单的目标函数：最大化roster数量
            new_rosters_count = len(found_rosters) - last_roster_count
            
            # 只在有足够迭代历史后才检查收敛，避免过早终止
            if (iteration_count > 50 and  # 至少50次迭代后才考虑收敛
                self.convergence_manager.should_terminate(current_obj, new_rosters_count, iteration_count)):
                self._log_debug(f"智能收敛终止：迭代{iteration_count}，方案{len(found_rosters)}")
                break
            
            last_roster_count = len(found_rosters)
            iteration_count += 1
            total_labels_processed += 1
            
            current_cost, current_label = heapq.heappop(labels)
            
            # 每5000次迭代输出一次进度
            if iteration_count % 5000 == 0:
                self._log_debug(f"  进度 {iteration_count}: 队列={len(labels)}, 方案={len(found_rosters)}")
            
            # 使用优化的状态键
            state_key = self.state_key_optimizer.get_compact_state_key(current_label)
            
            if state_key in visited:
                continue
            visited.add(state_key)
            
            # 内存管理 - 定期清理访问状态
            if self.memory_manager.should_cleanup(visited):
                visited = self.memory_manager.cleanup_visited_states(visited)
                self.task_index_manager.clear_cache()  # 同时清理任务缓存
            
            # 路径多样性检查
            if iteration_round > 0 and len(current_label.path) >= 2:
                # 创建路径特征：前几个任务的组合
                path_feature = tuple(sorted([
                    task['taskId'] for task in current_label.path[:min(3, len(current_label.path))]
                ]))
                
                if path_feature in path_signatures and len(path_signatures) > diversity_threshold:
                    # 如果路径特征重复且已有足够多样性，跳过
                    continue
                path_signatures.add(path_feature)
            
            # 检查是否到达规划结束时间或找到完整方案
            # 改进终止条件：允许更长的roster，提高覆盖率
            min_tasks_required = 3  # 进一步降低最小任务数量要求
            min_flight_tasks = 1   # 至少包含1个航班任务
            
            # 统计航班任务数量
            flight_tasks_count = sum(1 for task in current_label.path if task['type'] == 'flight')
            
            # 检查是否可以终止：时间结束或返回基地且满足条件
            can_terminate = False
            if current_label.node.time >= planning_end_dt:
                can_terminate = True
            elif (current_label.node.airport == crew.base and 
                  len(current_label.path) >= min_tasks_required and
                  flight_tasks_count >= min_flight_tasks):
                # 放宽休息时间要求，允许更灵活的终止
                if (current_label.duty_start_time is None or 
                    current_label.node.time - current_label.duty_start_time >= timedelta(hours=4)):
                    can_terminate = True
            
            if can_terminate:
                
                # 生成方案签名
                task_ids = tuple(sorted(task_info['taskId'] for task_info in current_label.path))
                roster_signature = (crew.crewId, task_ids)
                
                # 只添加未见过的方案
                if roster_signature not in found_roster_signatures:
                    found_roster_signatures.add(roster_signature)
                    
                    # 构建排班方案 - 添加去重逻辑和置位航班处理
                    roster_tasks = []
                    seen_task_ids = set()
                    for task_info in current_label.path:
                        task_id = task_info['taskId']
                        # 跳过重复的任务ID
                        if task_id in seen_task_ids:
                            continue
                        seen_task_ids.add(task_id)
                        
                        if task_info['type'] == 'flight':
                            # 执行航班
                            flight_obj = next(f for f in flights if f.id == task_id)
                            roster_tasks.append(flight_obj)
                        elif task_info['type'] == 'positioning_flight':
                            # 置位航班：使用原始航班ID查找flight对象
                            original_flight_id = task_info.get('original_flight_id', task_id.replace('_pos', ''))
                            flight_obj = next(f for f in flights if f.id == original_flight_id)
                            roster_tasks.append(flight_obj)
                        elif task_info['type'] == 'positioning_bus':
                            bus_obj = next(b for b in buses if b.id == task_id)
                            roster_tasks.append(bus_obj)
                        elif task_info['type'] == 'ground_duty':
                            ground_duty_obj = next(gd for gd in ground_duties if gd.id == task_id)
                            roster_tasks.append(ground_duty_obj)
                    
                    if roster_tasks:
                        # 创建临时roster用于成本计算
                        temp_roster = Roster(crew.crewId, roster_tasks, 0.0)
                        
                        # 使用scoring_system计算完整成本
                        scoring_system = ScoringSystem(flights, [crew], layover_airports)
                        cost_details = scoring_system.calculate_roster_cost_with_dual_prices(
                            temp_roster, crew, dual_prices, crew_sigma_dual
                        )
                        
                        # 简单质量检查
                        reduced_cost = cost_details['reduced_cost']
                        
                        # 记录所有考虑的roster的详细信息（不管是否有价值）
                        roster_status = "有价值" if reduced_cost < -1e-4 else "无价值"
                        self._log_debug(f"\n考虑的Roster ({roster_status}):")
                        self._log_debug(f"  任务路径: {[task['taskId'] for task in current_label.path]}")
                        self._log_debug(f"  Reduced Cost: {reduced_cost:.6f}")
                        self._log_debug(f"  最大化线性目标函数")
                        self._log_debug(f"  飞行奖励值: {cost_details.get('flight_reward', 0):.6f}")
                        self._log_debug(f"  航班数量: {cost_details['flight_count']}")
                        self._log_debug(f"  总飞行时间: {cost_details['total_flight_hours']:.2f}小时")
                        self._log_debug(f"  值勤天数: {cost_details['duty_days']}")
                        self._log_debug(f"  总成本: {cost_details['total_cost']:.6f}")
                        self._log_debug(f"  航班对偶价格收益: {cost_details.get('dual_price_total', 0):.6f}")
                        self._log_debug(f"  机组对偶价格: {cost_details.get('crew_sigma_dual', 0):.6f}")
                        self._log_debug(f"  对偶价格总贡献: {cost_details.get('dual_contribution', 0):.6f}")
                        
                        # 调用外部日志函数记录roster信息
                        if external_log_func:
                            value_status = "有价值" if reduced_cost < -1e-4 else "无价值"
                            external_log_func(f"机组 {crew.crewId} - 考虑的Roster ({value_status}):")
                            external_log_func(f"  任务路径: {[task['taskId'] for task in current_label.path]}")
                            external_log_func(f"  Reduced Cost: {reduced_cost:.6f}")
                            external_log_func(f"  最大化线性目标函数")
                            external_log_func(f"  飞行奖励值: {cost_details.get('flight_reward', 0):.6f}")
                            external_log_func(f"  航班数量: {cost_details['flight_count']}")
                            external_log_func(f"  总飞行时间: {cost_details['total_flight_hours']:.2f}小时")
                            external_log_func(f"  值勤天数: {cost_details['duty_days']}")
                            external_log_func(f"  总成本: {cost_details['total_cost']:.6f}")
                            external_log_func(f"  航班对偶价格收益: {cost_details.get('dual_price_total', 0):.6f}")
                            external_log_func(f"  机组对偶价格: {cost_details.get('crew_sigma_dual', 0):.6f}")
                            external_log_func(f"  对偶价格总贡献: {cost_details.get('dual_contribution', 0):.6f}")
                            external_log_func("")  # 空行分隔
                        
                        if reduced_cost < -1e-4:  # 基础有价值条件
                            # 使用计算出的成本创建最终roster
                            roster = Roster(crew.crewId, roster_tasks, cost_details['total_cost'])
                            found_rosters.append(roster)
                            self._log_debug(f"  >>> 添加到有价值roster列表 #{len(found_rosters)}")
            
            # 获取候选任务 - 使用优化的任务索引管理器
            if not hasattr(self, '_tasks_preprocessed'):
                self.task_index_manager.preprocess_tasks(all_tasks)
                self._tasks_preprocessed = True
            
            candidates = self.task_index_manager.get_candidates_optimized(
                current_label, crew, time_window_hours=48
            )
            
            # 进一步过滤候选任务
            candidates = self._filter_candidates_with_constraints(
                candidates, current_label, crew, layover_airports, planning_end_dt
            )
            
            total_candidates_found += len(candidates)
            
            if not candidates:
                continue
            
            # 只在前50次迭代输出详细信息
            if iteration_count <= 50:
                self._log_debug(f"    迭代 {iteration_count}: {current_label.node.airport} {current_label.node.time.strftime('%m-%d %H:%M')}, 候选 {len(candidates)}")
            
            # 使用注意力模型对候选任务进行评分和排序
            scored_candidates = self._score_candidates_with_attention(candidates, current_label, crew)
            
            # 引入多样性的候选选择策略
            if iteration_round == 0:
                # 第一轮：选择评分最高的候选（贪婪策略）
                top_candidates = scored_candidates[:self.max_candidates_per_expansion]
            else:
                 # 后续轮次：使用概率采样增加多样性
                
                # 计算温度参数，随着轮次增加而增大（增加随机性）
                temperature = 0.5 + 0.3 * min(iteration_round, 10) / 10
                
                # 将评分转换为概率分布
                scores = np.array([score for score, _ in scored_candidates])
                if len(scores) > 0 and np.std(scores) > 1e-8:
                    # 使用softmax with temperature
                    exp_scores = np.exp(scores / temperature)
                    probs = exp_scores / np.sum(exp_scores)
                    
                    # 概率采样选择候选
                    num_to_select = min(self.max_candidates_per_expansion, len(candidates))
                    selected_indices = np.random.choice(
                        len(scored_candidates), 
                        size=num_to_select, 
                        replace=False, 
                        p=probs
                    )
                    top_candidates = [scored_candidates[i] for i in selected_indices]
                else:
                    # 如果评分差异很小，随机选择
                    random.shuffle(scored_candidates)
                    top_candidates = scored_candidates[:self.max_candidates_per_expansion]
            
            # 扩展标签
            for score, candidate_idx in top_candidates:
                task = candidates[candidate_idx]
                new_labels = self._create_new_label(current_label, task, crew, tie_breaker)
                
                if new_labels:
                    # 处理返回的标签列表（可能包含继续值勤和结束值勤的标签）
                    if isinstance(new_labels, list):
                        for label in new_labels:
                            heapq.heappush(labels, (label.cost, label))
                    else:
                        # 向后兼容，如果返回单个标签
                        heapq.heappush(labels, (new_labels.cost, new_labels))
        
        self._log_debug(f"=== 机组 {crew.crewId} 求解完成 ===\n迭代: {iteration_count}, 方案: {len(found_rosters)}, 平均候选: {total_candidates_found/max(1, total_labels_processed):.1f}")
        self._log_debug(f"多样性统计: 探索了{len(path_signatures)}种不同路径特征")
        
        # 优化11: 输出缓存统计
        self._log_cache_stats()
        
        return found_rosters
    
    def _filter_candidates_with_constraints(self, candidates: List[Dict], current_label: Label,
                                          crew: Crew, layover_airports: Set[str], 
                                          planning_end_dt: datetime) -> List[Dict]:
        """对候选任务进行详细约束检查"""
        filtered_candidates = []
        current_time = current_label.node.time
        current_airport = current_label.node.airport
        
        for task in candidates:
            # 检查是否已使用
            if task['taskId'] in current_label.used_task_ids:
                continue
                
            # 检查时间约束
            if task['startTime'] <= current_time or task['endTime'] > planning_end_dt:
                continue
                
            # 检查总飞行时间约束（规则9：总飞行值勤时间限制）
            if task['type'] == 'flight':
                current_flight_hours = sum(t.get('flyTime', 0) / 60.0 for t in current_label.path if t.get('type') == 'flight')
                task_flight_hours = task.get('flyTime', 0) / 60.0
                if current_flight_hours + task_flight_hours > MAX_TOTAL_FLIGHT_HOURS:
                    continue
            
            # 使用统一约束检查器进行详细检查
            if self.constraint_checker.can_assign_task_to_label(current_label, task, crew):
                filtered_candidates.append(task)
        
        return filtered_candidates
    
    # 原来的_get_valid_candidates方法已被优化的候选任务获取方法替代
    
    def _log_filter_stats(self, filter_stats: dict, current_airport: str, current_time: datetime, candidates_count: int):
        """统一的过滤统计日志输出"""
        # 原有的日志输出逻辑保持不变
        pass
    
    def _log_cache_stats(self):
        """优化10: 输出缓存统计信息"""
        total_requests = self._cache_hits + self._cache_misses
        if total_requests > 0:
            hit_rate = (self._cache_hits / total_requests) * 100
            self._log_debug(f"缓存统计 - 命中: {self._cache_hits}, 未命中: {self._cache_misses}, 命中率: {hit_rate:.1f}%")
    
    def _log_filter_stats_original(self, filter_stats: dict, current_airport: str, current_time: datetime, candidates_count: int):
        """统一的过滤统计日志输出方法"""
        if candidates_count == 0:
            self._log_debug(f"      候选任务过滤统计 - 位置: {current_airport}, 时间: {current_time.strftime('%m-%d %H:%M')}")
            self._log_debug(f"        总任务数: {filter_stats['total_tasks']}")
            self._log_debug(f"        已使用: {filter_stats['already_used']}")
            self._log_debug(f"        时间约束过滤: {filter_stats['time_constraint']}")
            self._log_debug(f"        地点约束过滤: {filter_stats['location_constraint']}")
            self._log_debug(f"        过夜约束过滤: {filter_stats.get('layover_constraint', 0)}")
            self._log_debug(f"        连接时间过滤: {filter_stats['connection_time']}")
            self._log_debug(f"        值勤时长过滤: {filter_stats.get('duty_time', 0)}")
            self._log_debug(f"        任务数量过滤: {filter_stats.get('task_count', 0)}")
            self._log_debug(f"        飞行数量过滤: {filter_stats.get('flight_count', 0)}")
            self._log_debug(f"        值勤飞行时间过滤: {filter_stats.get('duty_flight_time', 0)}")
            self._log_debug(f"        值勤约束过滤: {filter_stats['duty_constraint']}")
            self._log_debug(f"        过夜约束过滤: {filter_stats['overnight_constraint']}")
            self._log_debug(f"        有效候选: {filter_stats['valid_candidates']}")
        elif candidates_count > 0:
            self._log_debug(f"      找到 {candidates_count} 个有效候选任务 - 位置: {current_airport}, 时间: {current_time.strftime('%m-%d %H:%M')}")
    
    def _check_duty_constraints(self, current_label: Label, task: Dict, crew: Crew = None) -> bool:
        """检查值勤时间相关约束 - 使用统一约束检查器"""
        try:
            return self.constraint_checker.can_assign_task_to_label(current_label, task, crew)
        except Exception as e:
            if self.debug:
                print(f"约束检查出错: {e}")
            return False
    
    def _create_new_label(self, current_label: Label, task: Dict, 
                     crew: Crew, tie_breaker) -> Optional[List[Label]]:
        """基于当前标签和新任务创建新标签"""
        try:
            # 计算新的节点
            new_node = Node(task['arriAirport'], task['endTime'])
            
            # 计算成本增量 - 使用更准确的成本计算
            cost_delta = 0.0
            if task['type'] == 'flight':
                # 执行航班：有对偶价格收益
                dual_price = task.get('dual_price', 0.0)
                cost_delta -= dual_price  # 航班的对偶价格收益（负成本）      
                # print(f"  执行航班 {task['taskId']}: 对偶价格={dual_price:.6f}, 成本增量={cost_delta:.6f}")
                
            elif task['type'] == 'positioning_flight':
                # 置位航班：有置位惩罚，无对偶价格收益
                cost_delta += PENALTY_PER_POSITIONING
                # print(f"  置位航班 {task['taskId']}: 惩罚={PENALTY_PER_POSITIONING:.6f}")
                
            elif task['type'] == 'positioning_bus':
                # 置位巴士：有置位惩罚
                cost_delta += PENALTY_PER_POSITIONING
                # print(f"  置位巴士 {task['taskId']}: 惩罚={PENALTY_PER_POSITIONING:.6f}")
                
            elif task['type'] == 'ground_duty':
                # 占位任务的对偶价格收益
                dual_price = task.get('dual_price', 0.0)
                cost_delta -= dual_price  # 占位任务的对偶价格收益（负成本）
                # print(f"  占位任务 {task['taskId']}: 对偶价格={dual_price:.6f}, 成本增量={cost_delta:.6f}")
            
            # 检查是否需要结束当前值勤日或开始新值勤日
            new_duty_start_time = current_label.duty_start_time
            new_duty_days_count = current_label.duty_days_count
            is_new_duty = False
            duty_ended = False
            
            if current_label.duty_start_time is None:
                # 第一个任务，开始第一个值勤日
                new_duty_start_time = task['startTime']
                new_duty_days_count = 1
                is_new_duty = True
            else:
                # 检查是否需要休息（结束当前值勤日）
                rest_time = task['startTime'] - current_label.node.time
                if rest_time >= timedelta(hours=MIN_REST_HOURS):
                    # 足够的休息时间，明确结束当前值勤日
                    duty_ended = True
                    new_duty_start_time = task['startTime']  # 开始新值勤日
                    new_duty_days_count = current_label.duty_days_count + 1
                    is_new_duty = True
                    # 检查外站过夜
                    if current_label.node.airport != crew.base:
                        overnight_days = (task['startTime'].date() - current_label.node.time.date()).days
                        if overnight_days > 0:
                            cost_delta += PENALTY_PER_AWAY_OVERNIGHT * overnight_days
                elif (current_label.node.airport == crew.base and 
                      rest_time >= timedelta(hours=2)):  # 在基地的短暂休息也可以结束值勤日
                    # 在基地的休息，可以选择结束值勤日
                    duty_ended = True
                    new_duty_start_time = task['startTime']  # 开始新值勤日
                    new_duty_days_count = current_label.duty_days_count + 1
                    is_new_duty = True
            
            # 置位规则检查：同一值勤日内，仅允许在开始或结束进行置位
            if not self._validate_positioning_rules_in_duty(current_label, task, is_new_duty):
                return None  # 违反置位规则
            
            # 更新值勤相关计数器
            new_duty_flight_time = current_label.duty_flight_time
            new_duty_flight_count = current_label.duty_flight_count
            new_duty_task_count = current_label.duty_task_count
            
            if is_new_duty:  # 新值勤日，重置计数器
                new_duty_flight_time = 0.0
                new_duty_flight_count = 0
                new_duty_task_count = 0
            
            if task['type'] == 'flight':
                new_duty_flight_time += task.get('flyTime', 0) / 60.0
                new_duty_flight_count += 1
            new_duty_task_count += 1
            
            # 更新总计数器
            new_total_flight_hours = current_label.total_flight_hours
            new_total_positioning = current_label.total_positioning
            if task['type'] == 'flight':
                new_total_flight_hours += task.get('flyTime', 0) / 60.0
            elif 'positioning' in task['type']:
                new_total_positioning += 1
            
            # 更新日历天数
            new_calendar_days = current_label.total_calendar_days.copy()
            task_date = task['startTime'].date()
            new_calendar_days.add(task_date)
            
            # 双重检查：确保任务未被使用（防止重复）
            if task['taskId'] in current_label.used_task_ids:
                return None  # 任务已被使用，不创建新标签
            
            # 更新已使用任务ID
            new_used_task_ids = current_label.used_task_ids.copy()
            new_used_task_ids.add(task['taskId'])
            
            # 更新飞行值勤日状态
            new_has_flown_in_duty = current_label.has_flown_in_duty
            if is_new_duty:
                # 新值勤日，重置飞行状态
                new_has_flown_in_duty = (task['type'] == 'flight')
            else:
                # 继续当前值勤日，更新飞行状态
                new_has_flown_in_duty = new_has_flown_in_duty or (task['type'] == 'flight')
            
            # 验证飞行值勤日的可过夜机场约束
            if not self._validate_flight_duty_day_layover_constraint(current_label, task, is_new_duty, new_has_flown_in_duty):
                return None  # 违反飞行值勤日可过夜机场约束
            
            # 飞行周期管理（规则11：飞行周期约束）
            new_cycle_start = current_label.current_cycle_start
            new_cycle_days = current_label.current_cycle_days
            new_last_base_return = current_label.last_base_return
            
            # 检查是否返回基地
            if task['arriAirport'] == crew.base:
                new_last_base_return = task['endTime'].date()
                # 如果有活跃的飞行周期，结束它
                if new_cycle_start is not None:
                    # 检查飞行周期末尾是否为飞行值勤日
                    if not self._is_flight_duty_day_ending_enhanced(current_label, task, is_new_duty):
                        return None  # 飞行周期末尾必须是飞行值勤日
                    new_cycle_start = None
                    new_cycle_days = 0
            else:
                # 不在基地，检查是否需要开始新的飞行周期
                if new_cycle_start is None:
                    # 飞行周期开始条件：飞行任务、置位任务或值勤占位
                    if (task['type'] == 'flight' or 
                        'positioning' in task['type'] or 
                        task['type'] == 'ground_duty'):
                        # 计算实际周期开始日期（考虑置位任务的影响）
                        new_cycle_start = self._get_cycle_actual_start_date(current_label, task)
                        new_cycle_days = (task_date - new_cycle_start).days + 1
                elif new_cycle_start is not None:
                    # 更新周期天数
                    cycle_duration = (task_date - new_cycle_start).days + 1
                    new_cycle_days = cycle_duration
                    
                    # 检查飞行周期最大持续时间（4个日历日）
                    if new_cycle_days > 4:
                        return None  # 飞行周期不能超过4个日历日
            
            # 创建新标签
            new_label = Label(
                cost=current_label.cost + cost_delta,
                path=current_label.path + [task],
                current_node=new_node,
                duty_start_time=new_duty_start_time,
                duty_flight_time=new_duty_flight_time,
                duty_flight_count=new_duty_flight_count,
                duty_task_count=new_duty_task_count,
                total_flight_hours=new_total_flight_hours,
                total_positioning=new_total_positioning,
                total_away_overnights=current_label.total_away_overnights,
                total_calendar_days=new_calendar_days,
                has_flown_in_duty=new_has_flown_in_duty,
                used_task_ids=new_used_task_ids,
                tie_breaker=next(tie_breaker),
                current_cycle_start=new_cycle_start,
                current_cycle_days=new_cycle_days,
                last_base_return=new_last_base_return,
                duty_days_count=new_duty_days_count  # 传递值勤日数量
            )
            
            # 如果任务结束后在基地且满足条件，创建一个值勤日结束的标签
            if (new_node.airport == crew.base and 
                new_duty_start_time is not None and 
                len(current_label.path) >= 2):  # 至少有一些任务
                
                # 创建值勤日结束标签（duty_start_time=None表示值勤日结束）
                duty_end_label = Label(
                    cost=new_label.cost,  # 相同成本
                    path=new_label.path,
                    current_node=new_node,
                    duty_start_time=None,  # 明确标记值勤日结束
                    duty_flight_time=0.0,  # 重置值勤计数器
                    duty_flight_count=0,
                    duty_task_count=0,
                    total_flight_hours=new_total_flight_hours,
                    total_positioning=new_total_positioning,
                    total_away_overnights=new_label.total_away_overnights,
                    total_calendar_days=new_calendar_days,
                    has_flown_in_duty=False,  # 重置值勤内飞行标记
                    used_task_ids=new_used_task_ids,
                    tie_breaker=next(tie_breaker),
                    current_cycle_start=new_cycle_start,
                    current_cycle_days=new_cycle_days,
                    last_base_return=new_node.time.date(),  # 更新最后回基地时间
                    duty_days_count=new_duty_days_count
                )
                
                # 返回两个标签：继续值勤的和结束值勤的
                return [new_label, duty_end_label]
            
            return [new_label]
            
        except Exception as e:
            print(f"Error creating new label: {e}")
            return None
    
    def _validate_positioning_rules_in_duty(self, current_label, task, is_new_duty):
        """
        验证置位规则：同一值勤日内，仅允许在开始或结束进行置位
        """
        # 如果当前任务不是置位任务，无需检查
        if not self._is_positioning_task_enhanced(task):
            return True
        
        # 如果是新值勤日开始，置位任务可以作为开始
        if is_new_duty:
            return True
        
        # 如果是继续当前值勤日，需要检查置位规则
        if current_label.duty_start_time is not None:
            # 检查当前值勤日中是否已经有置位任务
            duty_positioning_count = 0
            duty_has_flight = False
            
            # 分析当前值勤日的任务组成
            for path_task in current_label.path:
                # 检查任务是否在当前值勤日内
                if (hasattr(path_task, 'startTime') and 
                    path_task.startTime >= current_label.duty_start_time):
                    
                    if self._is_positioning_task_enhanced(path_task):
                        duty_positioning_count += 1
                    elif (hasattr(path_task, 'type') and 
                          str(path_task.type) == 'flight'):
                        duty_has_flight = True
            
            # 如果值勤日中已经有置位任务且有飞行任务，不允许再添加置位
            if duty_positioning_count > 0 and duty_has_flight:
                return False
            
            # 如果值勤日中已经有多个置位任务，不允许
            if duty_positioning_count >= 1:
                return False
        
        return True
    
    def _is_positioning_task_enhanced(self, task):
        """
        增强版置位任务识别
        根据attention模块的逻辑，置位任务包括：
        1. 飞行置位：positioning_flight
        2. 大巴置位：positioning_bus
        注意：groundDuty是占位任务，不是置位任务
        """
        if isinstance(task, dict):
            task_type = task.get('type', '')
        else:
            task_type = getattr(task, 'type', '')
        
        # 置位任务：飞行置位和大巴置位
        return (str(task_type) == 'positioning_flight' or 
                str(task_type) == 'positioning_bus' or
                'positioning' in str(task_type).lower() and 'ground' not in str(task_type).lower())
    
    def _is_ground_duty_task(self, task):
        """
        识别占位任务（groundDuty）
        根据用户澄清，groundDuty的识别可以从ID明确，ID格式为Grd_开头
        """
        if isinstance(task, dict):
            task_type = task.get('type', '')
            task_id = task.get('id', '') or task.get('taskId', '')
        else:
            task_type = getattr(task, 'type', '')
            task_id = getattr(task, 'id', '')
        
        # 占位任务：groundDuty类型或ID以Grd_开头
        return (str(task_type) == 'ground_duty' or 
                str(task_type) == 'groundDuty' or
                str(task_id).startswith('Grd_'))
    
    def _validate_flight_duty_day_layover_constraint(self, current_label, task, is_new_duty, has_flown_in_duty):
        """
        验证飞行值勤日的可过夜机场约束
        飞行值勤日必须从可过夜机场开始到可过夜机场结束
        """
        # 如果不是飞行值勤日，无需检查此约束
        if not has_flown_in_duty:
            return True
        
        # 检查值勤日开始机场
        duty_start_airport = None
        if is_new_duty:
            duty_start_airport = task['depaAirport']
        else:
            # 查找当前值勤日的开始机场
            for path_task in current_label.path:
                if (hasattr(path_task, 'startTime') and 
                    current_label.duty_start_time and
                    path_task.startTime >= current_label.duty_start_time):
                    if hasattr(path_task, 'depaAirport'):
                        duty_start_airport = path_task.depaAirport
                        break
                    elif hasattr(path_task, 'airport'):
                        duty_start_airport = path_task.airport
                        break
        
        # 检查值勤日结束机场
        duty_end_airport = task['arriAirport'] if 'arriAirport' in task else task.get('airport')
        
        # 验证开始和结束机场都是可过夜机场
        if (duty_start_airport and duty_start_airport not in self.layover_stations_set):
            return False
        
        if (duty_end_airport and duty_end_airport not in self.layover_stations_set):
            return False
        
        return True
    
    def _is_flight_duty_day_ending_enhanced(self, current_label, task, is_new_duty):
        """
        增强版飞行值勤日结束检查
        严格区分值勤日和飞行值勤日，确保飞行周期末尾是飞行值勤日
        """
        # 如果是新值勤日开始，需要检查前一个值勤日是否为飞行值勤日
        if is_new_duty and current_label.path:
            # 检查当前标签的值勤日是否包含飞行任务
            return current_label.has_flown_in_duty
        
        # 如果是继续当前值勤日，检查加入当前任务后是否构成飞行值勤日
        if task['type'] == 'flight':
            return True
        
        # 如果当前任务不是飞行任务，检查当前值勤日是否已经包含飞行任务
        return current_label.has_flown_in_duty
    
    def _get_cycle_actual_start_date(self, current_label, task):
        """
        计算飞行周期的实际开始日期
        考虑置位任务和值勤占位对周期开始的影响
        """
        task_date = task['startTime'].date()
        
        # 如果当前标签有路径，检查是否有置位任务影响周期开始
        if current_label.path:
            # 查找最近的置位任务或值勤占位
            for i in range(len(current_label.path) - 1, -1, -1):
                prev_task = current_label.path[i]
                
                # 检查是否为置位任务或值勤占位
                if (hasattr(prev_task, 'type') and 
                    ('positioning' in str(prev_task.type).lower() or 
                     str(prev_task.type) == 'ground_duty')):
                    # 如果找到置位任务，从该任务开始计算周期
                    if hasattr(prev_task, 'startTime'):
                        return prev_task.startTime.date()
                    elif hasattr(prev_task, 'std'):
                        return prev_task.std.date()
                
                # 如果遇到飞行任务，停止向前查找
                if (hasattr(prev_task, 'type') and 
                    str(prev_task.type) == 'flight'):
                    break
        
        # 默认返回当前任务的日期
        return task_date

def solve_subproblem_for_crew_with_attention(
    crew: Crew, all_flights: List[Flight], all_bus_info: List[BusInfo],
    crew_ground_duties: List[GroundDuty], dual_prices: Dict[str, float],
    layover_stations, crew_leg_match_dict: Dict[str, List[str]],
    crew_sigma_dual: float, ground_duty_duals: Dict[str, float] = None, iteration_round: int = 0, external_log_func=None
) -> List[Roster]:
    """使用注意力模型指导的子问题求解包装函数"""
    # 处理layover_stations参数，支持多种类型
    if isinstance(layover_stations, set):
        layover_airports = layover_stations
    elif isinstance(layover_stations, list):
        # 如果是LayoverStation对象列表，提取airport属性
        layover_airports = {station.airport if hasattr(station, 'airport') else str(station) for station in layover_stations}
    elif isinstance(layover_stations, dict):
        # 如果是字典，提取键作为机场代码
        layover_airports = set(layover_stations.keys())
    else:
        layover_airports = set()
    
    # 添加缺失的planning日期定义
    from datetime import datetime
    planning_start_dt = datetime.strptime("2025-04-29 00:00:00", "%Y-%m-%d %H:%M:%S")
    planning_end_dt = datetime.strptime("2025-05-07 23:59:59", "%Y-%m-%d %H:%M:%S")
    
    # 定义模型路径
    model_path = "models/best_model.pth"
    
    solver = AttentionGuidedSubproblemSolver(model_path, layover_stations_set=layover_airports)
    return solver.solve_subproblem_with_attention(
        crew, all_flights, all_bus_info, crew_ground_duties, dual_prices, 
        planning_start_dt, planning_end_dt, layover_airports, crew_sigma_dual, ground_duty_duals or {}, iteration_round, external_log_func
    )



