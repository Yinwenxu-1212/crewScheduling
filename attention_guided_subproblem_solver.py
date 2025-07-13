import heapq
import itertools
from datetime import datetime, timedelta, date
from data_models import Crew, Flight, BusInfo, GroundDuty, Node, Roster, RestPeriod
from typing import List, Dict, Set, Optional, Tuple
import csv
import torch
import torch.nn as nn
import numpy as np
import random
import sys
import os
# 使用attention中的改进版本
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'attention'))
from model import ActorCritic
from config import *
from scoring_system import ScoringSystem
from constraint_checker import UnifiedConstraintChecker
from unified_config import UnifiedConfig
from collections import defaultdict
import time

# 导入统一配置
from unified_config import config

# 动态获取配置参数的函数（不再使用全局缓存变量）
def get_reward_per_flight_hour():
    return UnifiedConfig.FLIGHT_TIME_REWARD

def get_penalty_per_away_overnight():
    return UnifiedConfig.AWAY_OVERNIGHT_PENALTY

def get_penalty_per_positioning():
    return UnifiedConfig.POSITIONING_PENALTY

def get_min_rest_hours():
    return UnifiedConfig.MIN_REST_HOURS

def get_max_duty_day_hours():
    return UnifiedConfig.MAX_DUTY_DAY_HOURS

def get_max_flight_time_in_duty_hours():
    return UnifiedConfig.MAX_FLIGHT_TIME_IN_DUTY_HOURS

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
        """获取时间窗口内的任务（优化时间窗口以平衡搜索范围和效率）"""
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
        """优化的候选任务获取（优化时间窗口以平衡搜索范围和效率）"""
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
    
    def get_candidates_for_label(self, current_label, time_window_hours=48):
        """为当前标签获取候选任务 - 修复版本，优化搜索窗口"""
        current_time = current_label.node.time
        current_airport = current_label.node.airport
        used_task_ids = current_label.used_task_ids
        
        # 简化缓存策略，避免过度缓存导致的问题
        candidates = []
        end_time = current_time + timedelta(hours=time_window_hours)
        
        # 从当前机场出发的任务
        if current_airport in self.tasks_by_location:
            for task in self.tasks_by_location[current_airport]:
                if (current_time < task['startTime'] <= end_time and 
                    task['taskId'] not in used_task_ids):
                    candidates.append(task)
        
        return candidates
    
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
        
        # 全局日均飞时近似分配相关参数
        self.global_duty_days_denominator = 0  # 全局执勤日分母
        
        # 增大搜索参数以提高覆盖率
        self.max_iterations = UnifiedConfig.MAX_SUBPROBLEM_ITERATIONS  # 使用统一配置的子问题迭代次数
        self.beam_width = UnifiedConfig.BEAM_WIDTH  # 使用统一配置的beam search宽度
        
        # 从统一配置获取约束参数
        self.MAX_DUTY_DAY_HOURS = UnifiedConfig.MAX_DUTY_DAY_HOURS
        self.MAX_FLIGHT_TIME_IN_DUTY_HOURS = UnifiedConfig.MAX_FLIGHT_TIME_IN_DUTY_HOURS
        self.MIN_REST_HOURS = UnifiedConfig.MIN_REST_HOURS
        
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
        
        # 加载预训练的注意力模型（使用固定维度以匹配attention-5）
        self.model = ActorCritic(
            state_dim=17,  # 固定为17维状态特征
            action_dim=20,  # 固定为20维动作特征
            hidden_dim=256,  # 固定隐藏层维度
            num_heads=8,    # 多头注意力头数
            num_layers=3,   # Transformer层数
            dropout=0.1     # Dropout率
        ).to(self.device)
        
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 尝试加载模型，如果维度不匹配则跳过不兼容的层
            try:
                self.model.load_state_dict(checkpoint, strict=False)
                self.model.eval()
                if self.debug_log:
                    self.debug_log.write(f"成功加载预训练模型（可能跳过部分不兼容层）: {model_path}\n")
            except Exception as load_error:
                if self.debug_log:
                    self.debug_log.write(f"模型加载失败，使用随机初始化: {load_error}\n")
                # 如果加载失败，继续使用随机初始化的模型
        else:
            if self.debug_log:
                self.debug_log.write(f"警告：未找到预训练模型 {model_path}，使用随机初始化的模型\n")
        
        # 注意力引导的参数
        self.max_candidates_per_expansion = 16  # 每次扩展最多考虑的候选任务数（增大搜索范围）
        self.use_attention_guidance = True
        
        # 初始化优化组件
        self.convergence_manager = ConvergenceManager(
            improvement_threshold=getattr(config, 'CONVERGENCE_THRESHOLD', 1e-6),
            stagnation_limit=getattr(config, 'STAGNATION_LIMIT', 5),
            min_iterations=getattr(config, 'MIN_ITERATIONS', 5)
        )
        self.task_index_manager = TaskIndexManager()
        self.state_key_optimizer = StateKeyOptimizer()
        self.memory_manager = MemoryManager(
            max_visited_states=getattr(config, 'MAX_VISITED_STATES', 100000),
            cleanup_interval=getattr(config, 'CLEANUP_INTERVAL', 1000)
        )
    
    def set_global_duty_days_denominator(self, denominator: float):
        """设置全局日均飞时计算的分母（加权执勤日数）"""
        self.global_duty_days_denominator = denominator
        if self.debug:
            print(f"子问题求解器设置全局执勤日分母（加权）: {denominator:.2f}")
    
    def __del__(self):
        """析构函数，关闭日志文件"""
        if hasattr(self, 'debug_log') and self.debug_log:
            self.debug_log.close()
    
    def _log_debug(self, message: str):
        """写入调试信息到日志文件"""
        # 使用external_log_func（如果可用）
        if hasattr(self, 'external_log_func') and self.external_log_func:
            self.external_log_func(message)
        # 同时使用内部日志文件
        if hasattr(self, 'debug_log') and self.debug_log:
            self.debug_log.write(f"{message}\n")
            self.debug_log.flush()  # 立即刷新到文件
        if self.debug:
            print(message)
    
    def _extract_state_features(self, label: Label, crew: Crew) -> np.ndarray:
        """从当前标签状态提取状态特征向量（17维，与attention-5保持一致）"""
        features = np.zeros(17)  # 固定为17维
        
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
        
        # 全局特征 (14-16)
        try:
            # 14. 全局覆盖压力
            features[14] = self._calculate_coverage_pressure(current_time)
            
            # 15. 全局约束风险
            features[15] = self._estimate_constraint_risk(label, crew)
            
            # 16. 时间紧迫性
            features[16] = self._calculate_time_urgency(current_time)
            
        except Exception as e:
            # 如果全局特征计算失败，使用默认值
            if self.debug:
                print(f"Warning: 全局特征计算失败: {e}")
            features[14:17] = 0.0
        
        return features
    
    def _extract_task_features(self, task, current_label: Label) -> np.ndarray:
        """提取任务特征向量（20维，与新attention架构保持一致）"""
        features = np.zeros(20)  # 固定为20维
        
        # 基础特征 (0-12)
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
        
        # 机场特征 - 使用更稳定的编码方式
        # 避免使用hash函数，改用字符串长度和ASCII值的组合
        depa_code = task['depaAirport']
        arri_code = task['arriAirport']
        features[7] = (len(depa_code) * 100 + sum(ord(c) for c in depa_code[:3])) % 1000
        features[8] = (len(arri_code) * 100 + sum(ord(c) for c in arri_code[:3])) % 1000
        
        # 时间特征
        features[9] = task['startTime'].weekday()
        features[10] = task['startTime'].hour
        features[11] = task['endTime'].hour
        
        # 任务持续时间
        duration = (task['endTime'] - task['startTime']).total_seconds() / 3600
        features[12] = min(duration, 24)
        
        # 对偶价格特征 (13-19) - 7维特征以匹配新模型架构
        try:
            # 13. 航班紧迫性
            features[13] = self._calculate_flight_urgency(task, current_label)
            
            # 14. 置位价值
            features[14] = self._calculate_positioning_value(task, current_label, crew)
            
            # 15. 资源利用效率
            features[15] = self._calculate_resource_efficiency(task, current_label)
            
            # 16. 约束违规风险
            features[16] = self._estimate_violation_risk(task, current_label)
            
            # 17. 关键任务标识
            features[17] = 1.0 if self._is_critical_ground_duty(task) else 0.0
            
            # 18. 中间置位标识
            features[18] = 1.0 if self._is_middle_positioning(task, current_label) else 0.0
            
            # 19. 任务优先级评分（综合评分）
            priority_score = 0.0
            if task['type'] == 'flight':
                priority_score += 0.3 * features[13]  # 航班紧迫性
                priority_score += 0.2 * features[15]  # 资源效率
                priority_score += 0.5 * (1.0 - features[16])  # 低风险加分
            elif 'positioning' in task.get('type', '') or task['type'] == 'positioning_bus':
                priority_score += 0.4 * features[14]  # 置位价值
                priority_score += 0.3 * (1.0 - features[16])  # 低风险加分
                priority_score += 0.3 * (1.0 - features[18])  # 非中间置位加分
            elif task['type'] == 'ground_duty':
                priority_score += 0.5 * features[17]  # 关键任务
                priority_score += 0.5 * (1.0 - features[16])  # 低风险加分
            
            features[19] = min(priority_score, 1.0)
            
        except Exception as e:
            # 如果特征计算失败，使用默认值
            if self.debug:
                print(f"Warning: 对偶价格特征计算失败: {e}")
            features[13:20] = 0.0
        
        # 检查所有特征是否包含NaN或无穷大值
        for i, val in enumerate(features):
            if np.isnan(val) or np.isinf(val):
                if self.debug:
                    print(f"Warning: 特征 {i} 包含无效值 {val}，已替换为0.0")
                features[i] = 0.0
        
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
                action_probs_tensor, _ = self.model(state_tensor, candidates_tensor, action_mask)
                action_probs = action_probs_tensor.squeeze(0).cpu().numpy()  # (num_candidates,)
            
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
    
    def _adjust_candidates_priority_intelligent(self, scored_candidates: List[Tuple[float, int]], 
                                              candidates: List[Dict], crew: Crew, 
                                              current_label: Label) -> List[Tuple[float, int]]:
        """【最终修正版】优先级调整：精确区分第一个和后续飞行周期"""
        # 1. 任务分组 (逻辑保持不变)
        mandatory_ground_duties = []
        execution_flights = []
        positioning_tasks = []
        
        for score, idx in scored_candidates:
            task = candidates[idx]
            if task['type'] == 'ground_duty' and task.get('crewId') == crew.crewId:
                mandatory_ground_duties.append((score, idx))
            elif task['type'] == 'flight' and task.get('subtype') == 'execution':
                execution_flights.append((score, idx))
            elif 'positioning' in task.get('type', '') or task['type'] == 'positioning_bus':
                positioning_tasks.append((score, idx))
        
        # 2. 上下文判断
        is_rested = (current_label.duty_start_time is None)
        
        # 3. 根据不同的上下文，应用不同的优先级策略
        result = []
        
        # 检查是否处于可以开始一个新周期的状态（在某个地方休息）
        if is_rested:
            expected_start_location = None
            log_message = ""
            
            # 策略一：如果第一个周期还没完成，且当前在stayStation，则准备开始第一个周期
            if not current_label.is_first_cycle_done and current_label.node.airport == crew.stayStation:
                expected_start_location = crew.stayStation
                log_message = f"  [决策策略]: 第一个飞行周期开始，从stayStation({crew.stayStation})出发，航班优先。"

            # 策略二：如果第一个周期已完成，且当前在base，则准备开始后续周期
            elif current_label.is_first_cycle_done and current_label.node.airport == crew.base:
                expected_start_location = crew.base
                log_message = f"  [决策策略]: 后续飞行周期开始，从base({crew.base})出发，航班优先。"
            
            # 如果处于一个可以开始新周期的状态
            if expected_start_location:
                self._log_debug(log_message)
                # a. 最高优先级：从正确起始位置出发的执行航班
                result.extend(sorted([t for t in execution_flights if candidates[t[1]]['depaAirport'] == expected_start_location], key=lambda x: x[0], reverse=True))
                # b. 中等优先级：在起始位置的占位任务
                result.extend(sorted([t for t in mandatory_ground_duties if candidates[t[1]]['depaAirport'] == expected_start_location], key=lambda x: x[0], reverse=True))
                # c. 最低优先级：从起始位置出发的置位任务
                result.extend(sorted([t for t in positioning_tasks if candidates[t[1]]['depaAirport'] == expected_start_location], key=lambda x: x[0], reverse=True))
                return result[:self.max_candidates_per_expansion]
        
        # 如果不处于"新周期开始"状态，则根据具体情况采用智能策略
        current_airport = current_label.node.airport
        
        # 策略三：值勤中状态 - 优先高价值连接
        if not is_rested:  # 值勤中
            self._log_debug(f"  [决策策略]: 值勤中状态，在{current_airport}，优先高价值连接。")
            
            # a. 最高优先级：从当前位置出发的执行航班
            current_execution_flights = [t for t in execution_flights if candidates[t[1]]['depaAirport'] == current_airport]
            result.extend(sorted(current_execution_flights, key=lambda x: x[0], reverse=True))
            
            # b. 高优先级：高价值置位任务（用于优化后续连接）
            current_positioning = [t for t in positioning_tasks if candidates[t[1]]['depaAirport'] == current_airport]
            # 计算置位价值并排序
            positioning_with_value = []
            for score, idx in current_positioning:
                task = candidates[idx]
                positioning_value = self._calculate_positioning_value(task, current_label, crew)
                # 综合原始分数和置位价值
                combined_score = score * 0.6 + positioning_value * 0.4
                positioning_with_value.append((combined_score, idx))
            
            # 只选择高价值置位（价值>0.5）
            high_value_positioning = [(s, i) for s, i in positioning_with_value 
                                    if self._calculate_positioning_value(candidates[i], current_label, crew) > 0.5]
            result.extend(sorted(high_value_positioning, key=lambda x: x[0], reverse=True))
            
            # c. 中等优先级：当前位置的占位任务
            current_ground_duties = [t for t in mandatory_ground_duties if candidates[t[1]]['depaAirport'] == current_airport]
            result.extend(sorted(current_ground_duties, key=lambda x: x[0], reverse=True))
            
        # 策略四：外站休息状态 - 优先返回或高价值连接
        else:  # 休息状态但不在正确的周期起始位置
            self._log_debug(f"  [决策策略]: 外站休息状态，在{current_airport}，优先返回基地或高价值连接。")
            
            # a. 最高优先级：从当前位置出发的执行航班
            current_execution_flights = [t for t in execution_flights if candidates[t[1]]['depaAirport'] == current_airport]
            result.extend(sorted(current_execution_flights, key=lambda x: x[0], reverse=True))
            
            # b. 高优先级：返回基地的置位或高价值置位
            current_positioning = [t for t in positioning_tasks if candidates[t[1]]['depaAirport'] == current_airport]
            positioning_with_priority = []
            
            for score, idx in current_positioning:
                task = candidates[idx]
                positioning_value = self._calculate_positioning_value(task, current_label, crew)
                
                # 返回基地的置位获得额外加分
                base_return_bonus = 0.3 if task['arriAirport'] == crew.base else 0.0
                
                # 综合评分：原始分数 + 置位价值 + 返回基地奖励
                combined_score = score * 0.5 + positioning_value * 0.3 + base_return_bonus
                positioning_with_priority.append((combined_score, idx))
            
            result.extend(sorted(positioning_with_priority, key=lambda x: x[0], reverse=True))
            
            # c. 较低优先级：占位任务
            result.extend(sorted(mandatory_ground_duties, key=lambda x: x[0], reverse=True))
        
        return result[:self.max_candidates_per_expansion]
    
    def _calculate_flight_urgency(self, task, current_label):
        """计算航班紧迫性"""
        if task['type'] != 'flight':
            return 0.0
        
        # 基于连接时间的紧迫性
        connection_time = (task['startTime'] - current_label.node.time).total_seconds() / 3600
        if connection_time <= 2:
            return 1.0  # 非常紧迫
        elif connection_time <= 6:
            return 0.7  # 较紧迫
        elif connection_time <= 12:
            return 0.4  # 一般紧迫
        else:
            return 0.1  # 不紧迫
    
    def _calculate_positioning_value(self, task, current_label, crew):
        """计算置位价值 - 基于实际数据的智能评估"""
        if 'positioning' not in task.get('type', '') and task['type'] != 'positioning_bus':
            return 0.0
        
        try:
            target_airport = task['arriAirport']
            arrival_time = task['endTime']
            
            # 1. 基础机场重要性权重（从配置获取）
            important_airports = getattr(config, 'IMPORTANT_AIRPORTS', 
                                       {'VIOC', 'RRES', 'RTHW', 'ENDP', 'TATC', 'TPWY', 'VWSF', 'XVFW'})
            base_value = 0.6 if target_airport in important_airports else 0.3
            
            # 2. 后续航班连接价值分析
            connection_value = self._evaluate_positioning_connections(target_airport, arrival_time, current_label, crew)
            
            # 3. 时间敏感性评估
            time_urgency = self._calculate_positioning_time_urgency(arrival_time, current_label)
            
            # 4. 机场覆盖需求评估
            coverage_need = self._evaluate_airport_coverage_need(target_airport)
            
            # 获取权重配置
            weights = getattr(config, 'POSITIONING_VALUE_WEIGHTS', {
                'base_importance': 0.3,
                'connection_value': 0.4,
                'time_urgency': 0.2,
                'coverage_need': 0.1
            })
            
            # 综合评分（使用配置化权重）
            total_value = (
                weights['base_importance'] * base_value +
                weights['connection_value'] * connection_value +
                weights['time_urgency'] * time_urgency +
                weights['coverage_need'] * coverage_need
            )
            
            return min(total_value, 1.0)
            
        except Exception as e:
            if self.debug:
                print(f"Warning: 置位价值计算失败: {e}")
            # 降级到简化计算
            important_airports = getattr(config, 'IMPORTANT_AIRPORTS', 
                                       {'VIOC', 'RRES', 'RTHW', 'ENDP', 'TATC', 'TPWY', 'VWSF', 'XVFW'})
            return 0.6 if task['arriAirport'] in important_airports else 0.3
    
    def _calculate_resource_efficiency(self, task, current_label):
        """计算资源利用效率"""
        if task['type'] == 'flight':
            fly_time = task.get('flyTime', 0) / 60.0  # 转换为小时
            if fly_time > 0:
                return min(fly_time / 8.0, 1.0)  # 归一化到8小时
        elif task['type'] == 'ground_duty':
            duration = (task['endTime'] - task['startTime']).total_seconds() / 3600
            return min(duration / 12.0, 1.0)  # 归一化到12小时
        
        return 0.5  # 默认中等效率
    
    def _estimate_violation_risk(self, task, current_label):
        """估计约束违规风险"""
        risk = 0.0
        
        # 检查值勤时长风险
        if current_label.duty_start_time:
            potential_duty_end = task['endTime']
            duty_length = (potential_duty_end - current_label.duty_start_time).total_seconds() / 3600
            if duty_length > 10:
                risk += 0.3
            if duty_length > 12:
                risk += 0.5
        
        # 检查飞行时间风险
        if task['type'] == 'flight':
            fly_time = task.get('flyTime', 0) / 60.0
            total_flight_time = current_label.duty_flight_time + fly_time
            if total_flight_time > 6:
                risk += 0.2
            if total_flight_time > 8:
                risk += 0.4
        
        return min(risk, 1.0)
    
    def _is_critical_ground_duty(self, task):
        """判断是否为关键占位任务"""
        if task['type'] != 'ground_duty':
            return False
        
        # 检查是否在关键时段
        hour = task['startTime'].hour
        return 6 <= hour <= 10 or 18 <= hour <= 22  # 早晚高峰时段
    
    def _is_middle_positioning(self, task, current_label):
        """判断是否为值勤中间的置位"""
        if 'positioning' not in task.get('type', '') and task['type'] != 'positioning_bus':
            return False
        
        # 如果当前在值勤中且已有任务，则认为是中间置位
        return current_label.duty_start_time is not None and current_label.duty_task_count > 0
    
    def _evaluate_positioning_connections(self, target_airport, arrival_time, current_label, crew):
        """评估置位后的航班连接价值
        
        Args:
            target_airport: 置位目标机场
            arrival_time: 置位到达时间
            current_label: 当前标签状态
            
        Returns:
            float: 连接价值评分 (0.0-1.0)
        """
        try:
            if not hasattr(self, 'crew_leg_match_dict') or not self.crew_leg_match_dict:
                if self.debug:
                    print("Warning: crew_leg_match_dict 不可用，使用默认连接价值")
                return 0.3  # 默认中等价值
            
            # 获取当前机组的资质航班
            crew_id = crew.crewId
            if not crew_id:
                if self.debug:
                    print("Warning: 无法获取机组ID，使用默认连接价值")
                return 0.3
            
            eligible_flights = self.crew_leg_match_dict.get(crew_id, [])
            if not eligible_flights:
                return 0.2  # 无资质航班，连接价值较低
            
            # 查找置位后可执行的航班（48小时窗口）
            connection_window = timedelta(hours=48)
            connection_count = 0
            high_value_connections = 0
            
            # 获取枢纽机场配置
            hub_airports = getattr(config, 'HUB_AIRPORTS', {'VIOC', 'RRES', 'RTHW'})
            
            # 基于实际航班数据查找后续连接航班
            for flight_id in eligible_flights:
                # 从flights数据中查找匹配的航班
                matching_flights = [f for f in getattr(self, 'flights', []) 
                                  if f.flightId == flight_id and 
                                     f.deptAirport == target_airport and
                                     f.deptTime >= arrival_time and
                                     f.deptTime <= arrival_time + connection_window]
                
                for flight in matching_flights:
                    connection_count += 1
                    # 判断是否为高价值连接（长航线或重要目的地）
                    if (flight.arriAirport in hub_airports or 
                        flight.flyTime > 180):  # 3小时以上航班
                        high_value_connections += 1
            
            # 如果没有flights数据，降级到基于机场重要性的估算
            if not hasattr(self, 'flights') or not self.flights:
                for flight_id in eligible_flights:
                    if target_airport in hub_airports:  # 重要枢纽
                        connection_count += 2
                        high_value_connections += 1
                    else:
                        connection_count += 1
            
            # 计算连接价值
            if connection_count == 0:
                return 0.1  # 无连接机会
            elif connection_count >= 3:
                # 高连接度，额外奖励高价值连接
                return min(0.8 + 0.2 * (high_value_connections / 3), 1.0)
            else:
                # 中等连接度
                return 0.4 + 0.2 * (connection_count / 3)
                
        except Exception as e:
            if self.debug:
                print(f"Warning: 连接价值评估失败: {e}")
            return 0.3
    
    def _calculate_positioning_time_urgency(self, arrival_time, current_label):
        """计算置位的时间紧迫性"""
        try:
            # 使用当前标签的时间而不是系统时间
            current_time = current_label.node.time
            time_to_arrival = (arrival_time - current_time).total_seconds() / 3600
            
            if time_to_arrival <= 6:
                return 0.9  # 非常紧迫
            elif time_to_arrival <= 12:
                return 0.7  # 较紧迫
            elif time_to_arrival <= 24:
                return 0.5  # 一般
            elif time_to_arrival <= 48:
                return 0.3  # 不太紧迫
            else:
                return 0.1  # 不紧迫
        except Exception:
            return 0.5
    
    def _evaluate_airport_coverage_need(self, airport):
        """评估机场的覆盖需求"""
        try:
            # 使用动态机场分析器获取机场分类
            from dynamic_airport_analyzer import get_dynamic_airport_config
            airport_config = get_dynamic_airport_config()
            hub_airports = airport_config.get('HUB_AIRPORTS', set())
            major_airports = airport_config.get('MAJOR_AIRPORTS', set())
            
            if airport in hub_airports:
                return 0.8  # 枢纽机场需求高
            elif airport in major_airports:
                return 0.6  # 主要机场需求中等
            else:
                return 0.4  # 其他机场需求较低
        except Exception:
            return 0.5
    
    def _count_available_flights(self, current_label, crew):
        """计算当前状态下可用的航班数量 - 基于实际数据和机组资质
        
        Args:
            current_label: 当前标签状态，包含时间、位置和已使用任务信息
            crew: 机组对象，用于获取机组资质信息
            
        Returns:
            int: 估算的可用航班数量
            
        Note:
            这个方法现在基于真实的航班时刻表数据进行精确计算。
            当航班数据不可用时，会降级到基于机场重要性、时间段和机组资质的估算。
        """
        try:
            current_time = current_label.node.time
            current_airport = current_label.node.airport
            
            # 获取当前机组的资质航班
            if hasattr(self, 'crew_leg_match_dict') and self.crew_leg_match_dict:
                crew_id = crew.crewId
                if crew_id:
                    eligible_flights = self.crew_leg_match_dict.get(crew_id, [])
                    
                    if self.debug:
                        print(f"机组 {crew_id} 在 {current_airport} 有 {len(eligible_flights)} 个资质航班")
                    
                    # 计算时间窗口内的可用航班
                    time_window = timedelta(hours=24)  # 24小时窗口
                    
                    # 基于实际航班数据统计机场的航班数量
                    if hasattr(self, 'flights') and self.flights:
                        # 统计当前机场在时间窗口内的实际航班数量
                        window_start = current_time
                        window_end = current_time + time_window
                        
                        airport_flights = [f for f in self.flights 
                                         if f.deptAirport == current_airport and
                                            window_start <= f.deptTime <= window_end and
                                            f.flightId in eligible_flights]
                        
                        base_count = len(airport_flights)
                        
                        if self.debug:
                            print(f"实际统计 {current_airport} 在时间窗口内有 {base_count} 个可用航班")
                    else:
                        # 降级到配置化的基础航班数量
                        base_flights_per_airport = {
                            'VIOC': 20, 'RRES': 15, 'RTHW': 12,  # 主要枢纽
                            'ENDP': 8, 'TATC': 7, 'TPWY': 6, 'VWSF': 6, 'XVFW': 6,  # 高频机场
                            'JFEE': 5, 'BTTC': 5, 'GDHI': 4, 'RTWL': 4  # 重要机场
                        }
                        base_count = base_flights_per_airport.get(current_airport, 2)
                         
                        if self.debug:
                             print(f"使用配置数据，{current_airport} 基础航班数量: {base_count}")
                    
                    # 根据时间段调整（反映航班时刻表的实际分布）
                    hour = current_time.hour
                    if 6 <= hour <= 10 or 18 <= hour <= 22:  # 高峰时段
                        time_factor = 1.2
                    elif 22 <= hour <= 6:  # 夜间时段
                        time_factor = 0.4
                    else:  # 平峰时段
                        time_factor = 1.0
                    
                    available_count = int(base_count * time_factor)
                    
                    # 考虑已使用的航班对可用性的影响
                    used_flights = len(current_label.used_task_ids)
                    # 假设每使用2个任务，减少1个可用航班选择
                    available_count = max(0, available_count - used_flights // 2)
                    
                    if self.debug:
                        print(f"估算可用航班: 基础={base_count}, 时间因子={time_factor}, 最终={available_count}")
                    
                    return available_count
            
            # 降级到简化估算（当无法获取机组资质时）
            if self.debug:
                print("Warning: 无法获取机组资质信息，使用简化估算")
            
            # 使用动态机场分析器获取重要机场配置
            try:
                from dynamic_airport_analyzer import get_dynamic_airport_config
                airport_config = get_dynamic_airport_config()
                important_airports = airport_config.get('IMPORTANT_AIRPORTS', set())
            except Exception:
                important_airports = {'VIOC', 'RRES', 'RTHW'}  # 降级配置
            
            base_count = 8 if current_airport in important_airports else 3
            
            hour = current_time.hour
            if 6 <= hour <= 10 or 18 <= hour <= 22:
                return int(base_count * 1.2)
            elif 0 <= hour <= 6:
                return int(base_count * 0.4)
            else:
                return base_count
                
        except Exception as e:
            if self.debug:
                print(f"Warning: 可用航班计算失败: {e}")
            return 3  # 保守的默认值
    

    
    def _calculate_coverage_pressure(self, current_time):
        """计算全局覆盖压力"""
        # 基于时间的覆盖压力
        hour = current_time.hour
        if 6 <= hour <= 10 or 18 <= hour <= 22:
            return 0.8  # 高峰时段压力大
        elif 10 <= hour <= 18:
            return 0.6  # 白天压力中等
        else:
            return 0.3  # 夜间压力小
    
    def _estimate_constraint_risk(self, label, crew):
        """估计全局约束风险"""
        risk = 0.0
        
        # 基于累计飞行时间的风险
        if label.total_flight_hours > 80:
            risk += 0.3
        if label.total_flight_hours > 100:
            risk += 0.4
        
        # 基于值勤天数的风险
        if label.duty_days_count > 15:
            risk += 0.2
        if label.duty_days_count > 20:
            risk += 0.3
        
        return min(risk, 1.0)
    
    def _calculate_time_urgency(self, current_time):
        """计算时间紧迫性"""
        # 基于规划期剩余时间的紧迫性
        from datetime import datetime
        
        # 假设规划期结束时间
        planning_end = datetime(2025, 5, 7, 23, 59, 59)
        remaining_hours = (planning_end - current_time).total_seconds() / 3600
        
        if remaining_hours <= 24:
            return 1.0  # 非常紧迫
        elif remaining_hours <= 72:
            return 0.7  # 较紧迫
        elif remaining_hours <= 168:
            return 0.4  # 一般紧迫
        else:
            return 0.1  # 不紧迫
    
    def _adjust_candidates_priority(self, scored_candidates: List[Tuple[float, int]], 
                                  candidates: List[Dict], crew: Crew) -> List[Tuple[float, int]]:
        """调整候选任务优先级，确保占位任务优先（保留原方法以兼容性）"""
        # 分组
        mandatory_ground_duties = []
        other_ground_duties = []
        other_tasks = []
        
        for score, idx in scored_candidates:
            task = candidates[idx]
            if task['type'] == 'ground_duty':
                if task.get('crewId') == crew.crewId:
                    mandatory_ground_duties.append((score, idx))
                else:
                    other_ground_duties.append((score, idx))
            else:
                other_tasks.append((score, idx))
        
        # 重新组合，确保优先级
        result = []
        result.extend(mandatory_ground_duties)  # 全部加入
        
        remaining_slots = self.max_candidates_per_expansion - len(result)
        if remaining_slots > 0:
            # 按原始分数排序其他任务
            combined_others = other_tasks + other_ground_duties
            combined_others.sort(reverse=True, key=lambda x: x[0])
            result.extend(combined_others[:remaining_slots])
        
        return result[:self.max_candidates_per_expansion]
    
    def solve_subproblem_with_attention(self, crew: Crew, flights: List[Flight],
                                      buses: List[BusInfo], ground_duties: List[GroundDuty],
                                      dual_prices: Dict[str, float], 
                                      planning_start_dt: datetime, planning_end_dt: datetime,
                                      layover_airports: Set[str], crew_sigma_dual: float, ground_duty_duals: Dict[str, float], 
                                      crew_leg_match_dict: Dict[str, List[str]], iteration_round: int = 0, external_log_func=None) -> List[Roster]:
        """使用注意力模型指导的子问题求解"""
        
        # 保存external_log_func到实例变量
        self.external_log_func = external_log_func
        
        # 保存数据到实例变量以供其他方法使用
        self.crew_leg_match_dict = crew_leg_match_dict
        self.flights = flights
        self.buses = buses
        self.ground_duties = ground_duties
        
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
            last_base_return=None,  # 强烈建议将初始值设为None，以避免混淆
            duty_days_count=0,  # 初始值勤日数量为0
            is_first_cycle_done=False  # 明确初始状态：第一个周期尚未完成
        )
        
        heapq.heappush(labels, (0.0, initial_label))
        
        # 准备任务数据时确保使用最新的对偶价格
        all_tasks = []
        
        # 获取该机组的资质航班
        eligible_flights = crew_leg_match_dict.get(crew.crewId, [])
        eligible_flight_set = set(eligible_flights)
        
        # 添加航班任务 - 为每个航班创建执行和置位两种任务类型
        for flight in flights:
            # 确保使用当前迭代的对偶价格
            current_dual_price = dual_prices.get(flight.id, 0.0)
            
            # 1. 如果有资质，可以作为执行任务
            if flight.id in eligible_flight_set:
                execution_task = {
                    'type': 'flight',
                    'subtype': 'execution',  # 明确标记为执行
                    'taskId': f"{flight.id}_exec",
                    'original_flight_id': flight.id,
                    'startTime': flight.std,
                    'endTime': flight.sta,
                    'depaAirport': flight.depaAirport,
                    'arriAirport': flight.arriAirport,
                    'flyTime': flight.flyTime,
                    'aircraftNo': flight.aircraftNo,
                    'dual_price': current_dual_price,
                    'is_positioning': False,
                    # 成本增量：-π_f - α·t_f（使用加权分母计算飞行时间奖励）
                    'cost_delta': self._calculate_execution_flight_cost_delta(current_dual_price, flight.flyTime)
                }
                all_tasks.append(execution_task)
            
            # 2. 所有航班都可以作为置位任务（受置位规则约束）
            positioning_task = {
                'type': 'flight',
                'subtype': 'positioning',  # 明确标记为置位
                'taskId': f"{flight.id}_pos",
                'original_flight_id': flight.id,
                'startTime': flight.std,
                'endTime': flight.sta,
                'depaAirport': flight.depaAirport,
                'arriAirport': flight.arriAirport,
                'flyTime': flight.flyTime,
                'aircraftNo': flight.aircraftNo,
                'dual_price': 0.0,  # 置位无对偶价格收益
                'is_positioning': True,
                # 成本增量：+γ
                'cost_delta': get_penalty_per_positioning()
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
                'dual_price': current_dual_price,
                'crewId': ground_duty.crewId,  # 添加机组ID字段
                # 预计算成本增量：占位任务的对偶价格收益
                'cost_delta': -current_dual_price
            }
            all_tasks.append(task_dict)
        
        # 性能优化：预处理任务索引
        self.task_index_manager.preprocess_tasks(all_tasks)
        
        # 主循环
        iteration_count = 0
        # 动态调整搜索参数，基于迭代轮次增加多样性
        # max_iterations 在上面根据迭代轮次设置
        
        # 根据迭代轮次调整搜索参数 - 性能优化版本
        if iteration_round == 0:  # 第一轮
            max_valuable_rosters = min(len(all_tasks), 80)  # 减少目标数量
            self.max_candidates_per_expansion = 6  # 减少候选数量
            max_iterations = min(self.max_iterations, 1500)  # 限制迭代次数
        else:
            max_valuable_rosters = min(len(all_tasks), 120)  # 适度增加
            self.max_candidates_per_expansion = 8  # 适度增加
            max_iterations = min(self.max_iterations, 2000)  # 限制迭代次数
        
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
                            # 根据subtype判断是执行还是置位
                            if task_info.get('subtype') == 'execution':
                                # 执行航班
                                original_flight_id = task_info.get('original_flight_id', task_id.replace('_exec', ''))
                                original_flight = next(f for f in flights if f.id == original_flight_id)
                                # 创建副本并标记为执行任务
                                import copy
                                flight_obj = copy.deepcopy(original_flight)
                                flight_obj.is_positioning = False
                                roster_tasks.append(flight_obj)
                            else:
                                # 置位航班
                                original_flight_id = task_info.get('original_flight_id', task_id.replace('_pos', ''))
                                original_flight = next(f for f in flights if f.id == original_flight_id)
                                # 创建副本并标记为置位任务
                                import copy
                                flight_obj = copy.deepcopy(original_flight)
                                flight_obj.is_positioning = True
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
                            temp_roster, crew, dual_prices, crew_sigma_dual, self.global_duty_days_denominator, ground_duty_duals
                        )
                        
                        # 简单质量检查
                        reduced_cost = cost_details['reduced_cost']
                        
                        # 记录所有考虑的roster的详细信息（不管是否有价值）
                        roster_status = "有价值" if reduced_cost < 0 else "无价值"
                        
                        # 记录所有reduced cost < 0的roster的详细信息
                        if reduced_cost < 0:
                            self._log_debug(f"\n发现Reduced Cost < 0的Roster:")
                            self._log_debug(f"  任务路径: {[task['taskId'] for task in current_label.path]}")
                            
                            # 详细的路径信息
                            self._log_debug(f"  详细路径信息:")
                            for i, task in enumerate(current_label.path):
                                task_type = task['type']
                                start_time = task['startTime'].strftime('%m-%d %H:%M')
                                end_time = task['endTime'].strftime('%m-%d %H:%M')
                                if task_type == 'flight':
                                    depa = task.get('depaAirport', 'N/A')
                                    arri = task.get('arriAirport', 'N/A')
                                    fly_time = task.get('flyTime', 0) / 60.0
                                    dual_price = task.get('dual_price', 0.0)
                                    self._log_debug(f"    {i+1}. {task['taskId']}: {task_type} {depa}->{arri} {start_time}-{end_time} 飞行时间:{fly_time:.1f}h 对偶价格:{dual_price:.4f}")
                                else:
                                    airport = task.get('airport', task.get('arriAirport', 'N/A'))
                                    dual_price = task.get('dual_price', 0.0)
                                    self._log_debug(f"    {i+1}. {task['taskId']}: {task_type} {airport} {start_time}-{end_time} 对偶价格:{dual_price:.4f}")
                            
                            self._log_debug(f"  Reduced Cost: {reduced_cost:.6f}")
                            self._log_debug(f"  成本分解:")
                            self._log_debug(f"    总成本 (total_cost): {cost_details['total_cost']:.6f}")
                            self._log_debug(f"      - 飞行奖励 (flight_reward): -{cost_details.get('flight_reward', 0):.6f}")
                            self._log_debug(f"      - 置位惩罚 (positioning_penalty): +{cost_details.get('positioning_penalty', 0):.6f}")
                            self._log_debug(f"      - 过夜惩罚 (overnight_penalty): +{cost_details.get('overnight_penalty', 0):.6f}")
                            self._log_debug(f"      - 违规惩罚 (violation_penalty): +{cost_details.get('violation_penalty', 0):.6f}")
                            self._log_debug(f"    对偶价格贡献 (dual_contribution): {cost_details.get('dual_contribution', 0):.6f}")
                            self._log_debug(f"      - 航班对偶价格收益 (flight_dual_total): +{cost_details.get('flight_dual_total', 0):.6f}")
                            self._log_debug(f"      - 占位任务对偶价格收益 (ground_duty_dual_total): +{cost_details.get('ground_duty_dual_total', 0):.6f}")
                            self._log_debug(f"      - 机组对偶价格 (crew_sigma_dual): -{cost_details.get('crew_sigma_dual', 0):.6f}")
                            self._log_debug(f"  统计信息:")
                            self._log_debug(f"    航班数量: {cost_details['flight_count']}")
                            self._log_debug(f"    总飞行时间: {cost_details['total_flight_hours']:.2f}小时")
                            self._log_debug(f"    值勤天数: {cost_details['duty_days']}")
                            self._log_debug(f"    置位任务数: {cost_details.get('positioning_count', 0)}")
                            self._log_debug(f"    过夜次数: {cost_details.get('overnight_count', 0)}")
                            self._log_debug(f"    违规次数: {cost_details.get('violation_count', 0)}")
                        
                        # 调用外部日志函数记录roster信息
                        if external_log_func and reduced_cost < 0:
                            external_log_func(f"机组 {crew.crewId} - 发现Reduced Cost < 0的Roster:")
                            external_log_func(f"  任务路径: {[task['taskId'] for task in current_label.path]}")
                            
                            # 详细的路径信息
                            external_log_func(f"  详细路径信息:")
                            for i, task in enumerate(current_label.path):
                                task_type = task['type']
                                start_time = task['startTime'].strftime('%m-%d %H:%M')
                                end_time = task['endTime'].strftime('%m-%d %H:%M')
                                if task_type == 'flight':
                                    depa = task.get('depaAirport', 'N/A')
                                    arri = task.get('arriAirport', 'N/A')
                                    fly_time = task.get('flyTime', 0) / 60.0
                                    dual_price = task.get('dual_price', 0.0)
                                    external_log_func(f"    {i+1}. {task['taskId']}: {task_type} {depa}->{arri} {start_time}-{end_time} 飞行时间:{fly_time:.1f}h 对偶价格:{dual_price:.4f}")
                                else:
                                    airport = task.get('airport', task.get('arriAirport', 'N/A'))
                                    dual_price = task.get('dual_price', 0.0)
                                    external_log_func(f"    {i+1}. {task['taskId']}: {task_type} {airport} {start_time}-{end_time} 对偶价格:{dual_price:.4f}")
                            
                            external_log_func(f"  Reduced Cost: {reduced_cost:.6f}")
                            external_log_func(f"  成本分解:")
                            external_log_func(f"    总成本 (total_cost): {cost_details['total_cost']:.6f}")
                            external_log_func(f"      - 飞行奖励 (flight_reward): -{cost_details.get('flight_reward', 0):.6f}")
                            external_log_func(f"      - 置位惩罚 (positioning_penalty): +{cost_details.get('positioning_penalty', 0):.6f}")
                            external_log_func(f"      - 过夜惩罚 (overnight_penalty): +{cost_details.get('overnight_penalty', 0):.6f}")
                            external_log_func(f"      - 违规惩罚 (violation_penalty): +{cost_details.get('violation_penalty', 0):.6f}")
                            external_log_func(f"    对偶价格贡献 (dual_contribution): {cost_details.get('dual_contribution', 0):.6f}")
                            external_log_func(f"      - 航班对偶价格收益 (flight_dual_total): +{cost_details.get('flight_dual_total', 0):.6f}")
                            external_log_func(f"      - 占位任务对偶价格收益 (ground_duty_dual_total): +{cost_details.get('ground_duty_dual_total', 0):.6f}")
                            external_log_func(f"      - 机组对偶价格 (crew_sigma_dual): -{cost_details.get('crew_sigma_dual', 0):.6f}")
                            external_log_func(f"  统计信息:")
                            external_log_func(f"    航班数量: {cost_details['flight_count']}")
                            external_log_func(f"    总飞行时间: {cost_details['total_flight_hours']:.2f}小时")
                            external_log_func(f"    值勤天数: {cost_details['duty_days']}")
                            external_log_func(f"    置位任务数: {cost_details.get('positioning_count', 0)}")
                            external_log_func(f"    过夜次数: {cost_details.get('overnight_count', 0)}")
                            external_log_func(f"    违规次数: {cost_details.get('violation_count', 0)}")
                            external_log_func("")  # 空行分隔
                        
                        if reduced_cost < -1e-4:  # 基础有价值条件
                            # 使用计算出的成本创建最终roster
                            roster = Roster(crew.crewId, roster_tasks, cost_details['total_cost'])
                            found_rosters.append(roster)
                            self._log_debug(f"  >>> 添加到有价值roster列表 #{len(found_rosters)}")
            
            # 性能优化：使用任务索引器快速获取候选任务
            candidates = self.task_index_manager.get_candidates_for_label(
                current_label, 
                time_window_hours=min(48, (planning_end_dt - current_label.node.time).total_seconds() / 3600)
            )
            
            # 进一步过滤候选任务 - 批量处理提高效率
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
            
            # 应用智能优先级调整：执行航班优先，置位按需
            prioritized_candidates = self._adjust_candidates_priority_intelligent(
                scored_candidates, candidates, crew, current_label
            )
            
            # 引入多样性的候选选择策略
            if iteration_round == 0:
                # 第一轮：选择评分最高的候选（贪婪策略）
                top_candidates = prioritized_candidates[:self.max_candidates_per_expansion]
            else:
                 # 后续轮次：使用概率采样增加多样性
                
                # 计算温度参数，随着轮次增加而增大（增加随机性）
                temperature = 0.5 + 0.3 * min(iteration_round, 10) / 10
                
                # 将评分转换为概率分布
                scores = np.array([score for score, _ in prioritized_candidates])
                if len(scores) > 0 and np.std(scores) > 1e-8:
                    # 使用softmax with temperature
                    exp_scores = np.exp(scores / temperature)
                    probs = exp_scores / np.sum(exp_scores)
                    
                    # 概率采样选择候选
                    num_to_select = min(self.max_candidates_per_expansion, len(candidates), len(prioritized_candidates))
                    selected_indices = np.random.choice(
                        len(prioritized_candidates), 
                        size=num_to_select, 
                        replace=False, 
                        p=probs
                    )
                    top_candidates = [prioritized_candidates[i] for i in selected_indices]
                else:
                    # 如果评分差异很小，随机选择
                    random.shuffle(prioritized_candidates)
                    top_candidates = prioritized_candidates[:self.max_candidates_per_expansion]
            
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
        """【修正版】只执行最基本的硬性约束过滤，并返回所有合法的候选。"""
        current_time = current_label.node.time
        current_airport = current_label.node.airport
        
        valid_candidates = []
        
        for task in candidates:
            # 检查是否已使用
            if task['taskId'] in current_label.used_task_ids:
                continue
                
            # 检查时间约束（所有任务都使用统一的严格时间过滤）
            if task['startTime'] <= current_time or task['endTime'] > planning_end_dt:
                continue
                
            # 检查总飞行值勤时间约束（规则9：总飞行值勤时间限制）
            # 修正：使用飞行值勤时间而不是flyTime（空中飞行时间）
            if task['type'] == 'flight':
                # 计算当前路径中所有飞行值勤日的总飞行值勤时间
                current_flight_duty_hours = self._calculate_total_flight_duty_hours(current_label)
                
                # 计算添加当前任务后的飞行值勤时间增量
                task_flight_duty_hours_delta = self._calculate_flight_duty_hours_delta(current_label, task)
                
                if current_flight_duty_hours + task_flight_duty_hours_delta > MAX_TOTAL_FLIGHT_HOURS:
                    continue
            
            # 使用统一约束检查器进行详细检查
            if not self.constraint_checker.can_assign_task_to_label(current_label, task, crew, self.crew_leg_match_dict):
                continue
                
            # 通过所有约束检查的任务直接添加到有效候选列表
            valid_candidates.append(task)
        
        # 直接返回所有通过了检查的候选，不进行任何排序
        return valid_candidates
    
    def _calculate_total_flight_duty_hours(self, current_label: Label) -> float:
        """计算当前路径中所有飞行值勤日的总飞行值勤时间
        
        飞行值勤时间 = 飞行值勤日的总时长（从第一个任务开始到最后一个飞行任务结束）
        """
        if not current_label.path:
            return 0.0
        
        total_flight_duty_hours = 0.0
        
        # 将任务按时间排序并组织为值勤日
        sorted_tasks = sorted(current_label.path, key=lambda x: x['startTime'])
        
        # 分组为值勤日（基于休息时间间隔）
        duty_days = []
        current_duty = []
        
        for i, task in enumerate(sorted_tasks):
            if i == 0:
                current_duty = [task]
            else:
                prev_task = sorted_tasks[i-1]
                rest_time = task['startTime'] - prev_task['endTime']
                
                if rest_time >= timedelta(hours=self.MIN_REST_HOURS):
                    # 足够的休息时间，开始新值勤日
                    if current_duty:
                        duty_days.append(current_duty)
                    current_duty = [task]
                else:
                    # 继续当前值勤日
                    current_duty.append(task)
        
        if current_duty:
            duty_days.append(current_duty)
        
        # 计算每个飞行值勤日的飞行值勤时间
        for duty_day in duty_days:
            # 检查是否为飞行值勤日（包含至少一个飞行任务）
            has_flight = any(task['type'] == 'flight' for task in duty_day)
            
            if has_flight:
                # 找到最后一个飞行任务
                last_flight_task = None
                for task in reversed(duty_day):
                    if task['type'] == 'flight':
                        last_flight_task = task
                        break
                
                if last_flight_task:
                    # 飞行值勤时间 = 从第一个任务开始到最后一个飞行任务结束
                    duty_start_time = duty_day[0]['startTime']
                    duty_end_time = last_flight_task['endTime']
                    flight_duty_hours = (duty_end_time - duty_start_time).total_seconds() / 3600.0
                    total_flight_duty_hours += flight_duty_hours
        
        return total_flight_duty_hours
    
    def _calculate_flight_duty_hours_delta(self, current_label: Label, new_task: Dict) -> float:
        """计算添加新任务后的飞行值勤时间增量
        
        Args:
            current_label: 当前标签状态
            new_task: 要添加的新任务
            
        Returns:
            float: 飞行值勤时间增量（小时）
        """
        if new_task['type'] != 'flight':
            return 0.0
        
        # 检查是否会开始新的值勤日
        if not current_label.path:
            # 第一个任务，飞行值勤时间就是任务本身的时长
            return (new_task['endTime'] - new_task['startTime']).total_seconds() / 3600.0
        
        # 检查与最后一个任务的休息时间
        last_task_end_time = current_label.node.time
        rest_time = new_task['startTime'] - last_task_end_time
        
        if rest_time >= timedelta(hours=self.MIN_REST_HOURS):
            # 开始新的值勤日，飞行值勤时间就是新任务的时长
            return (new_task['endTime'] - new_task['startTime']).total_seconds() / 3600.0
        else:
            # 继续当前值勤日
            if current_label.duty_start_time:
                # 检查当前值勤日是否已经是飞行值勤日
                current_duty_has_flight = current_label.has_flown_in_duty
                
                if current_duty_has_flight:
                    # 当前已经是飞行值勤日，计算时间延长
                    current_duty_end_time = last_task_end_time
                    new_duty_end_time = new_task['endTime']
                    
                    # 如果新任务延长了值勤日，增加相应的时间
                    if new_duty_end_time > current_duty_end_time:
                        return (new_duty_end_time - current_duty_end_time).total_seconds() / 3600.0
                    else:
                        return 0.0
                else:
                    # 当前不是飞行值勤日，添加飞行任务后变成飞行值勤日
                    # 飞行值勤时间 = 从值勤日开始到新飞行任务结束
                    return (new_task['endTime'] - current_label.duty_start_time).total_seconds() / 3600.0
            else:
                # 没有duty_start_time，按新值勤日处理
                return (new_task['endTime'] - new_task['startTime']).total_seconds() / 3600.0

    def _log_filter_stats(self, filter_stats: dict, current_airport: str, current_time: datetime, candidates_count: int):
        """统一的过滤统计日志输出"""
        if self.debug and candidates_count == 0:
            self._log_debug(f"      候选任务过滤统计 - 位置: {current_airport}, 时间: {current_time.strftime('%m-%d %H:%M')}")
            self._log_debug(f"        无有效候选任务")
        elif self.debug and candidates_count > 0:
            self._log_debug(f"      找到 {candidates_count} 个智能筛选的候选任务 - 位置: {current_airport}")
    
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
        
        return found_rosters

    def _calculate_execution_flight_cost_delta(self, dual_price: float, fly_time_minutes: float) -> float:
        """计算执行航班任务的成本增量（使用加权分母）"""
        cost_delta = -dual_price  # 对偶价格收益
        
        # 计算飞行时间奖励
        flight_hours = fly_time_minutes / 60.0
        if self.global_duty_days_denominator > 0:
            # 使用全局日均飞时近似分配：REWARD_PER_FLIGHT_HOUR * 该任务飞行时间 / 全局执勤日分母
            flight_time_reward = get_reward_per_flight_hour() * flight_hours / self.global_duty_days_denominator
            cost_delta -= flight_time_reward  # 飞行奖励（负值减少成本）
        else:
            # 回退到原始逻辑
            cost_delta -= get_reward_per_flight_hour() * flight_hours  # 飞行奖励（负值减少成本）
        
        return cost_delta

    def _calculate_task_cost_delta(self, task: Dict, crew: Crew = None, global_duty_days_denominator: float = 0.0) -> float:
        """计算任务的成本增量（使用全局日均飞时近似分配）"""
        cost_delta = 0.0
        
        if task['type'] == 'flight':
            # 获取对偶价格收益
            dual_price = task.get('dual_price', 0.0)
            cost_delta = -dual_price  # 对偶价格收益（负值减少成本）
            
            # 检查是否为执行航班（非置位航班）
            if not task.get('is_positioning', False):
                # 执行航班：使用全局日均飞时近似分配逻辑
                flight_hours = task.get('flyTime', 0) / 60.0
                if global_duty_days_denominator > 0:
                    # 新的飞行奖励 = REWARD_PER_FLIGHT_HOUR * 该任务飞行时间 / 全局执勤日分母
                    flight_time_reward = get_reward_per_flight_hour() * flight_hours / global_duty_days_denominator
                    cost_delta -= flight_time_reward  # 飞行奖励（负值减少成本）
                else:
                    # 回退到原始逻辑
                    cost_delta -= get_reward_per_flight_hour() * flight_hours  # 飞行奖励（负值减少成本）
            else:
                # 置位航班：额外置位惩罚
                cost_delta += get_penalty_per_positioning()
        elif task['type'] == 'positioning_bus':
            # 置位巴士：置位惩罚
            cost_delta = PENALTY_PER_POSITIONING
        elif task['type'] == 'ground_duty':
            # 占位任务：对偶价格收益
            dual_price = task.get('dual_price', 0.0)
            cost_delta = -dual_price
        
        return cost_delta

    def _check_duty_constraints(self, current_label: Label, task: Dict, crew: Crew = None) -> bool:
        """检查值勤时间相关约束 - 使用统一约束检查器"""
        try:
            return self.constraint_checker.can_assign_task_to_label(current_label, task, crew, self.crew_leg_match_dict)
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
            
            # 使用预计算的成本增量
            if 'cost_delta' in task:
                cost_delta = task['cost_delta']
            else:
                # 后备计算逻辑
                cost_delta = self._calculate_task_cost_delta(task, crew, self.global_duty_days_denominator)
            
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
                if rest_time >= timedelta(hours=self.MIN_REST_HOURS):
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
            
            # 置位任务数量限制：每个roster最多6个置位任务
            if ('positioning' in task['type'] or 
                (task['type'] == 'flight' and task.get('subtype') == 'positioning')):
                if current_label.total_positioning >= 6:
                    return None  # 超过置位任务数量限制
            
            # 更新值勤相关计数器
            new_duty_flight_time = current_label.duty_flight_time
            new_duty_flight_count = current_label.duty_flight_count
            new_duty_task_count = current_label.duty_task_count
            
            if is_new_duty:  # 新值勤日，重置计数器
                new_duty_flight_time = 0.0
                new_duty_flight_count = 0
                new_duty_task_count = 0
            
            if task['type'] == 'flight' and not task.get('is_positioning', False):
                new_duty_flight_time += task.get('flyTime', 0) / 60.0
                new_duty_flight_count += 1
            new_duty_task_count += 1
            
            # 更新飞行值勤日状态（需要在使用前初始化）
            new_has_flown_in_duty = current_label.has_flown_in_duty
            if is_new_duty:
                # 新值勤日，重置飞行状态
                new_has_flown_in_duty = (task['type'] == 'flight')
            else:
                # 继续当前值勤日，更新飞行状态
                new_has_flown_in_duty = new_has_flown_in_duty or (task['type'] == 'flight')
            
            # 更新总计数器
            new_total_flight_hours = current_label.total_flight_hours
            new_total_flight_duty_hours = current_label.total_flight_duty_hours
            new_total_positioning = current_label.total_positioning
            if task['type'] == 'flight':
                new_total_flight_hours += task.get('flyTime', 0) / 60.0
            elif 'positioning' in task['type']:
                new_total_positioning += 1
            
            # 如果值勤日结束且包含飞行任务，累加飞行值勤时间
            if duty_ended and new_has_flown_in_duty and new_duty_start_time:
                duty_duration = (task['endTime'] - new_duty_start_time).total_seconds() / 3600.0
                new_total_flight_duty_hours += duty_duration
            
            # 检查总飞行值勤时间限制（60小时）
            if new_total_flight_duty_hours > 60:
                return None
            
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
            
            # 验证飞行值勤日的可过夜机场约束
            if not self._validate_flight_duty_day_layover_constraint(current_label, task, is_new_duty, new_has_flown_in_duty):
                return None  # 违反飞行值勤日可过夜机场约束
            
            # 飞行周期管理（规则11：飞行周期约束）- 修正版
            new_cycle_start = current_label.current_cycle_start
            new_cycle_days = current_label.current_cycle_days
            new_last_base_return = current_label.last_base_return
            
            # 【新增】更新 is_first_cycle_done 状态
            new_is_first_cycle_done = current_label.is_first_cycle_done
            
            # 检查是否返回基地
            if task['arriAirport'] == crew.base:
                new_last_base_return = task['endTime'].date()
                
                # 如果第一个周期尚未完成，并且当前任务的终点是基地
                if not current_label.is_first_cycle_done:
                    new_is_first_cycle_done = True  # 标记第一个周期已完成
                
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
                    # 检查是否可以开始新飞行周期
                    if task['type'] == 'flight':
                        # 飞行任务必须在飞行周期内，检查开始条件
                        if current_label.node.airport != crew.base:
                            return None  # 不从基地出发，不能开始包含飞行的周期
                        
                        # 检查休息时间（2个完整日历日）
                        rest_days = self._calculate_rest_days_since_last_duty(current_label, task)
                        if rest_days < 2:
                            return None  # 休息不足，不能开始包含飞行的周期
                        
                        # 可以开始新周期
                        new_cycle_start = self._get_cycle_actual_start_date(current_label, task)
                        new_cycle_days = (task_date - new_cycle_start).days + 1
                    elif 'positioning' in task['type'] or task['type'] == 'ground_duty':
                        # 置位/占位任务，如果从基地出发且休息充足，可以开始周期
                        if (current_label.node.airport == crew.base and 
                            self._calculate_rest_days_since_last_duty(current_label, task) >= 2):
                            new_cycle_start = self._get_cycle_actual_start_date(current_label, task)
                            new_cycle_days = (task_date - new_cycle_start).days + 1
                        # 否则，置位/占位任务可以在飞行周期后执行，不开始新周期
                elif new_cycle_start is not None:
                    # 继续当前周期
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
                total_flight_duty_hours=new_total_flight_duty_hours,
                total_positioning=new_total_positioning,
                total_away_overnights=current_label.total_away_overnights,
                total_calendar_days=new_calendar_days,
                has_flown_in_duty=new_has_flown_in_duty,
                used_task_ids=new_used_task_ids,
                tie_breaker=next(tie_breaker),
                current_cycle_start=new_cycle_start,
                current_cycle_days=new_cycle_days,
                last_base_return=new_last_base_return,
                duty_days_count=new_duty_days_count,  # 传递值勤日数量
                is_first_cycle_done=new_is_first_cycle_done  # 传递第一个周期完成状态
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
                    total_flight_duty_hours=new_total_flight_duty_hours,
                    total_positioning=new_total_positioning,
                    total_away_overnights=new_label.total_away_overnights,
                    total_calendar_days=new_calendar_days,
                    has_flown_in_duty=False,  # 重置值勤内飞行标记
                    used_task_ids=new_used_task_ids,
                    tie_breaker=next(tie_breaker),
                    current_cycle_start=new_cycle_start,
                    current_cycle_days=new_cycle_days,
                    last_base_return=new_node.time.date(),  # 更新最后回基地时间
                    duty_days_count=new_duty_days_count,
                    is_first_cycle_done=new_is_first_cycle_done  # 传递第一个周期完成状态
                )
                
                # 返回两个标签：继续值勤的和结束值勤的
                return [new_label, duty_end_label]
            
            return [new_label]
            
        except Exception as e:
            # print(f"Error creating new label: {e}")  # 注释掉错误打印
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
    
    def _calculate_rest_days_since_last_duty(self, current_label, task):
        """
        计算自上次值勤结束以来的完整日历日休息天数
        """
        if current_label.last_base_return is None:
            # 如果没有上次返回基地的记录，假设有足够休息
            return 3
        
        # 计算完整日历日数量
        from datetime import datetime
        
        # 上次返回基地的时间
        last_return_time = datetime.combine(current_label.last_base_return, datetime.min.time())
        
        # 当前任务开始时间
        current_start_time = task['startTime']
        
        # 计算完整日历日
        end_date = last_return_time.date()
        start_date = current_start_time.date()
        
        if start_date <= end_date:
            return 0
        
        complete_days = (start_date - end_date).days - 1
        return max(0, complete_days)
    
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
    crew_sigma_dual: float, ground_duty_duals: Dict[str, float] = None, iteration_round: int = 0, external_log_func=None,
    global_duty_days_denominator: int = 0
) -> List[Roster]:
    """使用注意力模型指导的子问题求解包装函数"""
    try:
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
        from unified_config import UnifiedConfig
        
        # 从flight.csv动态获取计划开始和结束时间
        planning_start_dt = UnifiedConfig.get_planning_start_date()
        planning_end_dt = UnifiedConfig.get_planning_end_date()
        
        # 定义模型路径
        model_path = "models/best_model.pth"
        
        # 基本参数验证
        if not crew or not hasattr(crew, 'crewId'):
            raise ValueError(f"无效的机组对象: {crew}")
        
        if not hasattr(crew, 'stayStation') or not crew.stayStation:
            raise ValueError(f"机组 {crew.crewId} 缺少 stayStation 属性")
        
        if not hasattr(crew, 'base') or not crew.base:
            raise ValueError(f"机组 {crew.crewId} 缺少 base 属性")
        
        # 检查机组资质数据
        eligible_flights = crew_leg_match_dict.get(crew.crewId, [])
        if not eligible_flights:
            if external_log_func:
                external_log_func(f"机组 {crew.crewId} 无可执行航班，跳过")
            return []
        
        solver = AttentionGuidedSubproblemSolver(model_path, layover_stations_set=layover_airports)
        # 设置全局日均飞时分母
        solver.set_global_duty_days_denominator(global_duty_days_denominator)
        return solver.solve_subproblem_with_attention(
            crew, all_flights, all_bus_info, crew_ground_duties, dual_prices, 
            planning_start_dt, planning_end_dt, layover_airports, crew_sigma_dual, ground_duty_duals or {}, 
            crew_leg_match_dict, iteration_round, external_log_func
        )
        
    except Exception as e:
        error_msg = f"机组 {crew.crewId if crew and hasattr(crew, 'crewId') else 'Unknown'} 子问题求解失败: {str(e)}"
        if external_log_func:
            external_log_func(error_msg)
            import traceback
            external_log_func(f"详细错误堆栈: {traceback.format_exc()}")
        else:
            print(error_msg)
        
        # 返回空列表而不是抛出异常，让主程序继续运行
        return []



