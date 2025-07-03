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
from attention import config
from scoring_system import ScoringSystem
from unified_constraint_checker import UnifiedConstraintChecker

# 继承原有的常量和规则
from subproblem_solver import (
    TRAINING_DATA_FILE, CSV_HEADER, REWARD_PER_FLIGHT_HOUR, 
    PENALTY_PER_AWAY_OVERNIGHT, PENALTY_PER_POSITIONING,
    MIN_CONNECTION_TIME_FLIGHT_SAME_AIRCRAFT, MIN_CONNECTION_TIME_FLIGHT_DIFFERENT_AIRCRAFT, MIN_CONNECTION_TIME_BUS,
    MAX_DUTY_DAY_HOURS, MIN_REST_HOURS, MAX_FLIGHTS_IN_DUTY, 
    MAX_TASKS_IN_DUTY, MAX_FLIGHT_TIME_IN_DUTY_HOURS,
    is_conflicting, find_positioning_tasks
)

# 添加总飞行时间约束常量
MAX_TOTAL_FLIGHT_HOURS = 60.0  # 计划期内总飞行时间上限（小时）

# 添加缺失的常量定义
MAX_TASKS_PER_CREW = 50  # 每个机组人员的最大任务数
MAX_FLIGHTS_PER_CREW = 30  # 每个机组人员的最大飞行数

# 从data_models导入Label类
from data_models import Label

class AttentionGuidedSubproblemSolver:
    """使用注意力模型指导的子问题求解器"""
    
    def __init__(self, model_path: str = "models/best_model.pth", debug=False, layover_stations_set=None):
        """初始化求解器并加载预训练的注意力模型"""
        self.debug = debug
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 增大搜索参数以提高覆盖率
        self.max_iterations = 1000  # 增加迭代次数
        self.beam_width = 5  # 增大beam search宽度
        
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
            state_dim=config.STATE_DIM,
            action_dim=config.ACTION_DIM,
            hidden_dim=config.HIDDEN_DIM
        ).to(self.device)
        
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 适配动作特征权重维度
            if 'actor_action_encoder.0.weight' in checkpoint:
                old_weight = checkpoint['actor_action_encoder.0.weight']  # [256, old_dim]
                if old_weight.shape[1] != config.ACTION_DIM:
                    if old_weight.shape[1] > config.ACTION_DIM:
                        # 截取前N维，保留最重要的特征
                        checkpoint['actor_action_encoder.0.weight'] = old_weight[:, :config.ACTION_DIM]
                        if self.debug_log:
                            self.debug_log.write(f"动作特征维度适配: {old_weight.shape[1]} -> {config.ACTION_DIM}\n")
                    else:
                        # 如果旧维度小于新维度，用零填充
                        new_weight = torch.zeros(old_weight.shape[0], config.ACTION_DIM)
                        new_weight[:, :old_weight.shape[1]] = old_weight
                        checkpoint['actor_action_encoder.0.weight'] = new_weight
                        if self.debug_log:
                            self.debug_log.write(f"动作特征维度扩展: {old_weight.shape[1]} -> {config.ACTION_DIM}\n")
            
            # 适配状态特征权重维度
            if 'actor_state_encoder.0.weight' in checkpoint:
                old_weight = checkpoint['actor_state_encoder.0.weight']  # [256, old_dim]
                if old_weight.shape[1] != config.STATE_DIM:
                    if old_weight.shape[1] > config.STATE_DIM:
                        # 截取前N维
                        checkpoint['actor_state_encoder.0.weight'] = old_weight[:, :config.STATE_DIM]
                        if self.debug_log:
                            self.debug_log.write(f"状态特征维度适配: {old_weight.shape[1]} -> {config.STATE_DIM}\n")
                    else:
                        # 用零填充
                        new_weight = torch.zeros(old_weight.shape[0], config.STATE_DIM)
                        new_weight[:, :old_weight.shape[1]] = old_weight
                        checkpoint['actor_state_encoder.0.weight'] = new_weight
                        if self.debug_log:
                            self.debug_log.write(f"状态特征维度扩展: {old_weight.shape[1]} -> {config.STATE_DIM}\n")
            
            # 同样处理critic网络的状态编码器
            if 'critic.0.weight' in checkpoint:
                old_weight = checkpoint['critic.0.weight']  # [256, old_dim]
                if old_weight.shape[1] != config.STATE_DIM:
                    if old_weight.shape[1] > config.STATE_DIM:
                        checkpoint['critic.0.weight'] = old_weight[:, :config.STATE_DIM]
                    else:
                        new_weight = torch.zeros(old_weight.shape[0], config.STATE_DIM)
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
        self.max_candidates_per_expansion = 5  # 每次扩展最多考虑的候选任务数
        self.use_attention_guidance = True
    
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
        features = np.zeros(config.STATE_DIM)
        
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
        features = np.zeros(config.ACTION_DIM)
        
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
                                      layover_airports: Set[str], crew_sigma_dual: float, iteration_round: int = 0, external_log_func=None, lambda_k: float = 0.0) -> List[Roster]:
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
        
        # 添加航班任务
        for flight in flights:
            # 确保使用当前迭代的对偶价格
            current_dual_price = dual_prices.get(flight.id, 0.0)
            task_dict = {
                'type': 'flight',
                'taskId': flight.id,
                'startTime': flight.std,
                'endTime': flight.sta,
                'depaAirport': flight.depaAirport,
                'arriAirport': flight.arriAirport,
                'flyTime': flight.flyTime,
                'aircraftNo': flight.aircraftNo,  # 添加飞机尾号信息
                'dual_price': current_dual_price  # 使用最新的对偶价格
            }
            all_tasks.append(task_dict)
        
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
            task_dict = {
                'type': 'ground_duty',
                'taskId': ground_duty.id,
                'startTime': ground_duty.startTime,
                'endTime': ground_duty.endTime,
                'depaAirport': ground_duty.airport,
                'arriAirport': ground_duty.airport,  # 占位任务起降机场相同
                'dual_price': 0.0
            }
            all_tasks.append(task_dict)
        
        # 主循环
        iteration_count = 0
        # 动态调整搜索参数，基于迭代轮次增加多样性
        # max_iterations 在上面根据迭代轮次设置
        
        # 根据迭代轮次调整搜索参数
        if iteration_round == 0:  # 第一轮
            max_valuable_rosters = min(len(all_tasks), 50)
            self.max_candidates_per_expansion = 3
            max_iterations = self.max_iterations
        else:
            max_valuable_rosters = min(len(all_tasks), 60)
            self.max_candidates_per_expansion = 4
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
        
        # 基本循环条件
        while (labels and 
               iteration_count < max_iterations and 
               len(found_rosters) < max_valuable_rosters):
            iteration_count += 1
            total_labels_processed += 1
            
            current_cost, current_label = heapq.heappop(labels)
            
            # 每5000次迭代输出一次进度
            if iteration_count % 5000 == 0:
                self._log_debug(f"  进度 {iteration_count}: 队列={len(labels)}, 方案={len(found_rosters)}")
            
            # 改进状态键，包含更多信息
            state_key = (
                current_label.node.airport, 
                current_label.node.time.replace(second=0, microsecond=0),  # 精确到分钟
                tuple(sorted(current_label.used_task_ids)),
                current_label.duty_start_time.replace(second=0, microsecond=0) if current_label.duty_start_time else None,
                current_label.duty_flight_count,
                current_label.duty_task_count
            )
            
            if state_key in visited:
                continue
            visited.add(state_key)
            
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
                    
                    # 构建排班方案 - 添加去重逻辑
                    roster_tasks = []
                    seen_task_ids = set()
                    for task_info in current_label.path:
                        task_id = task_info['taskId']
                        # 跳过重复的任务ID
                        if task_id in seen_task_ids:
                            continue
                        seen_task_ids.add(task_id)
                        
                        if task_info['type'] == 'flight':
                            flight_obj = next(f for f in flights if f.id == task_id)
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
                            temp_roster, crew, dual_prices, crew_sigma_dual, lambda_k
                        )
                        
                        # 简单质量检查
                        reduced_cost = cost_details['reduced_cost']
                        
                        # 记录所有考虑的roster的详细信息（不管是否有价值）
                        roster_status = "有价值" if reduced_cost < -1e-4 else "无价值"
                        self._log_debug(f"\n考虑的Roster ({roster_status}):")
                        self._log_debug(f"  任务路径: {[task['taskId'] for task in current_label.path]}")
                        self._log_debug(f"  Reduced Cost: {reduced_cost:.6f}")
                        self._log_debug(f"  当前Lambda值: {lambda_k:.6f}")
                        self._log_debug(f"  飞行奖励值: {cost_details.get('flight_reward', 0):.6f}")
                        self._log_debug(f"  航班数量: {cost_details['flight_count']}")
                        self._log_debug(f"  总飞行时间: {cost_details['total_flight_hours']:.2f}小时")
                        self._log_debug(f"  值勤天数: {cost_details['duty_days']}")
                        self._log_debug(f"  总成本: {cost_details['total_cost']:.6f}")
                        self._log_debug(f"  对偶价格收益: {cost_details.get('dual_price_total', 0):.6f}")
                        
                        # 调用外部日志函数记录roster信息
                        if external_log_func:
                            value_status = "有价值" if reduced_cost < -1e-4 else "无价值"
                            external_log_func(f"机组 {crew.crewId} - 考虑的Roster ({value_status}):")
                            external_log_func(f"  任务路径: {[task['taskId'] for task in current_label.path]}")
                            external_log_func(f"  Reduced Cost: {reduced_cost:.6f}")
                            external_log_func(f"  当前Lambda值: {lambda_k:.6f}")
                            external_log_func(f"  飞行奖励值: {cost_details.get('flight_reward', 0):.6f}")
                            external_log_func(f"  航班数量: {cost_details['flight_count']}")
                            external_log_func(f"  总飞行时间: {cost_details['total_flight_hours']:.2f}小时")
                            external_log_func(f"  值勤天数: {cost_details['duty_days']}")
                            external_log_func(f"  总成本: {cost_details['total_cost']:.6f}")
                            external_log_func(f"  对偶价格收益: {cost_details.get('dual_price_total', 0):.6f}")
                            external_log_func("")  # 空行分隔
                        
                        if reduced_cost < -1e-4:  # 基础有价值条件
                            # 使用计算出的成本创建最终roster
                            roster = Roster(crew.crewId, roster_tasks, cost_details['total_cost'])
                            found_rosters.append(roster)
                            self._log_debug(f"  >>> 添加到有价值roster列表 #{len(found_rosters)}")
            
            # 获取候选任务
            candidates = self._get_valid_candidates(
                current_label, all_tasks, crew, layover_airports, planning_end_dt
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
    
    def _get_valid_candidates(self, current_label: Label, all_tasks: List[Dict], 
                            crew: Crew, layover_airports: List[str], 
                            planning_end_dt: datetime) -> List:
        """获取当前标签的有效候选任务 - 优化版本"""
        candidates = []
        current_time = current_label.node.time
        current_airport = current_label.node.airport
        
        # 添加过滤统计
        filter_stats = {
            'total_tasks': len(all_tasks),
            'already_used': 0,
            'time_constraint': 0,
            'location_constraint': 0,
            'layover_constraint': 0,
            'connection_time': 0,
            'duty_time': 0,
            'task_count': 0,
            'flight_count': 0,
            'duty_flight_time': 0,
            'duty_constraint': 0,
            'overnight_constraint': 0,
            'valid_candidates': 0
        }
        
        # 优化1: 快速预筛选 - 批量过滤最基本的约束
        pre_filtered_tasks = []
        used_task_ids = current_label.used_task_ids  # 缓存引用
        
        for task in all_tasks:
            # 最快的检查：已使用和时间约束
            if task['taskId'] in used_task_ids:
                filter_stats['already_used'] += 1
            elif task['startTime'] <= current_time or task['endTime'] > planning_end_dt:
                filter_stats['time_constraint'] += 1
            else:
                pre_filtered_tasks.append(task)
        
        # 优化2: 提前终止 - 如果预筛选后没有任务，直接返回
        if not pre_filtered_tasks:
            self._log_filter_stats(filter_stats, current_airport, current_time, 0)
            return candidates
        
        # 优化3: 批量检查简单约束
        path_length = len(current_label.path)
        current_flight_count = sum(1 for t in current_label.path if t.get('type') == 'flight')
        
        for task in pre_filtered_tasks:
            # 优化4: 快速约束检查 - 按失败概率排序
            
            # 检查任务数量约束（快速检查）
            if path_length >= MAX_TASKS_PER_CREW:
                filter_stats['task_count'] += 1
                continue
            
            # 检查飞行数量约束（快速检查）
            if task['type'] == 'flight':
                if current_flight_count + 1 > MAX_FLIGHTS_PER_CREW:
                    filter_stats['flight_count'] += 1
                    continue
            
            # 优化5: 地点约束检查 - 使用缓存避免重复查找
            if task['depaAirport'] != current_airport:
                # 使用缓存键避免重复查找置位任务
                cache_key = f"{current_airport}_{task['depaAirport']}_{current_time.strftime('%Y%m%d_%H%M')}"
                
                if not hasattr(self, '_positioning_cache'):
                    self._positioning_cache = {}
                
                if cache_key not in self._positioning_cache:
                    # 查找从当前机场到任务出发机场的置位任务
                    from subproblem_solver import find_positioning_tasks
                    
                    # 获取机组的航班列表用于查找DH航班
                    crew_flights = [f for f in all_tasks if f.get('type') == 'flight']
                    all_ddh = [f for f in crew_flights if f.get('flightNo', '').startswith('DH')]
                    
                    # 查找置位任务
                    positioning_tasks = find_positioning_tasks(
                        from_airport=current_airport,
                        to_airport=task['depaAirport'],
                        earliest_start=current_time,
                        all_bus=[t for t in all_tasks if t.get('type') == 'bus'],
                        all_ddh=all_ddh
                    )
                    
                    self._positioning_cache[cache_key] = positioning_tasks
                else:
                    positioning_tasks = self._positioning_cache[cache_key]
                
                # 如果找到置位任务，将其添加到候选列表
                positioning_added = False
                for pos_task in positioning_tasks:
                    pos_start = pos_task.get('startTime') if hasattr(pos_task, 'startTime') else pos_task.get('std')
                    pos_end = pos_task.get('endTime') if hasattr(pos_task, 'endTime') else pos_task.get('sta')
                    
                    if pos_start and pos_end and pos_start >= current_time:
                        # 创建置位任务字典
                        pos_task_dict = {
                            'taskId': getattr(pos_task, 'taskId', f"POS_{pos_task.get('flightNo', 'BUS')}_{pos_start.strftime('%Y%m%d_%H%M')}"),
                            'type': 'positioning' if hasattr(pos_task, 'flightNo') else 'bus',
                            'startTime': pos_start,
                            'endTime': pos_end,
                            'depaAirport': current_airport,
                            'arriAirport': task['depaAirport'],
                            'flightNo': getattr(pos_task, 'flightNo', 'BUS'),
                            'flyTime': getattr(pos_task, 'flyTime', 0)
                        }
                        
                        # 检查置位任务是否已被使用
                        if pos_task_dict['taskId'] not in used_task_ids:
                            candidates.append(pos_task_dict)
                            filter_stats['valid_candidates'] += 1
                            positioning_added = True
                
                # 如果没有找到置位任务，则跳过这个任务
                if not positioning_added:
                    filter_stats['location_constraint'] += 1
                    continue
            
            # 检查值勤日必须从可过夜机场开始
            if current_label.duty_start_time is None and current_label.path:
                # 如果即将开始一个新值勤日（即刚结束一次休息），那么当前机场必须是合法的过夜站
                if current_airport not in layover_airports:
                    filter_stats['layover_constraint'] += 1
                    continue
            
            # 检查连接时间
            connection_time = task['startTime'] - current_time
            
            # 获取最后一个任务用于连接时间检查
            last_task = None
            if current_label.path:
                for t in reversed(current_label.path):
                    if hasattr(t, 'std') or t.get('type') == 'flight':  # 是航班任务
                        last_task = t
                        break
            
            # 根据任务类型和飞机尾号确定最小连接时间
            if task['type'] == 'flight':
                if last_task and hasattr(last_task, 'aircraftNo') and task.get('aircraftNo'):
                    if task['aircraftNo'] == last_task.aircraftNo:
                        min_connection_time = MIN_CONNECTION_TIME_FLIGHT_SAME_AIRCRAFT
                    else:
                        min_connection_time = MIN_CONNECTION_TIME_FLIGHT_DIFFERENT_AIRCRAFT
                else:
                    min_connection_time = MIN_CONNECTION_TIME_FLIGHT_DIFFERENT_AIRCRAFT
                
                if connection_time < min_connection_time:
                    filter_stats['connection_time'] += 1
                    continue
            elif task['type'] in ['positioning_bus', 'bus']:
                if connection_time < MIN_CONNECTION_TIME_BUS:
                    filter_stats['connection_time'] += 1
                    continue
            
            # 检查值勤日时长约束
            if current_label.duty_start_time is not None:
                duty_duration = (task['endTime'] - current_label.duty_start_time).total_seconds() / 3600
                if duty_duration > MAX_DUTY_DAY_HOURS:
                    filter_stats['duty_time'] += 1
                    continue
            
            # 检查任务数量约束
            if len(current_label.path) >= MAX_TASKS_PER_CREW:
                filter_stats['task_count'] += 1
                continue
            
            # 检查飞行数量约束
            flight_count = sum(1 for t in current_label.path if t.get('type') == 'flight')
            if task['type'] == 'flight':
                flight_count += 1
            if flight_count > MAX_FLIGHTS_PER_CREW:
                filter_stats['flight_count'] += 1
                continue
            
            # 优化6: 值勤内飞行时间约束（快速计算）
            if current_label.duty_start_time is not None and task['type'] == 'flight':
                # 使用预计算的值勤飞行时间
                potential_duty_flight_time = current_label.duty_flight_time + task.get('flyTime', 0) / 60.0
                if potential_duty_flight_time > MAX_FLIGHT_TIME_IN_DUTY_HOURS:
                    filter_stats['duty_flight_time'] += 1
                    continue
            
            # 优化7: 延迟复杂约束检查到最后
            # 将最耗时的统一约束检查放到最后
            if not self._check_duty_constraints(current_label, task, crew):
                filter_stats['duty_constraint'] += 1
                continue
            
            # 移除了晚上8点过夜约束检查 - 该约束不在原始竞赛规则中
            
            candidates.append(task)
            filter_stats['valid_candidates'] += 1
        
        # 优化8: 简化日志输出
        self._log_filter_stats(filter_stats, current_airport, current_time, len(candidates))
        
        return candidates
    
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
                # 使用任务字典中存储的最新对偶价格
                dual_price = task.get('dual_price', 0.0)
                cost_delta -= dual_price  # 航班的对偶价格收益（负成本）      
                # print(f"  航班 {task['taskId']}: 对偶价格={dual_price:.6f}, 成本增量={cost_delta:.6f}")
                
            elif 'positioning' in task['type']:
                # 调机的惩罚
                cost_delta += PENALTY_PER_POSITIONING
                # print(f"  调机 {task['taskId']}: 惩罚={PENALTY_PER_POSITIONING:.6f}")
            elif task['type'] == 'ground_duty':
                # 占位任务通常没有额外成本
                # print(f"  占位任务 {task['taskId']}: 无额外成本")
                pass
            
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
            
            # 飞行周期管理（规则11：飞行周期约束）
            new_cycle_start = current_label.current_cycle_start
            new_cycle_days = current_label.current_cycle_days
            new_last_base_return = current_label.last_base_return
            
            # 检查是否返回基地
            if task['arriAirport'] == crew.base:
                new_last_base_return = task['endTime'].date()
                # 如果有活跃的飞行周期，结束它
                if new_cycle_start is not None:
                    # 优化：检查飞行周期末尾是否为飞行值勤日
                    if not self._is_flight_duty_day_ending(current_label, task):
                        return None  # 飞行周期末尾必须是飞行值勤日
                    new_cycle_start = None
                    new_cycle_days = 0
            else:
                # 不在基地，检查是否需要开始新的飞行周期
                if new_cycle_start is None and task['type'] == 'flight':
                    new_cycle_start = task_date
                    new_cycle_days = 1
                elif new_cycle_start is not None:
                    # 更新周期天数
                    cycle_duration = (task_date - new_cycle_start).days + 1
                    new_cycle_days = cycle_duration
            
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
                has_flown_in_duty=current_label.has_flown_in_duty or (task['type'] == 'flight'),
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

def solve_subproblem_for_crew_with_attention(
    crew: Crew, all_flights: List[Flight], all_bus_info: List[BusInfo],
    crew_ground_duties: List[GroundDuty], dual_prices: Dict[str, float],
    layover_stations, crew_leg_match_dict: Dict[str, List[str]],
    crew_sigma_dual: float, iteration_round: int = 0, external_log_func=None, lambda_k: float = 0.0
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
        planning_start_dt, planning_end_dt, layover_airports, crew_sigma_dual, iteration_round, external_log_func, lambda_k
    )

# 在AttentionGuidedSubproblemSolver类中添加值四修二约束检查方法
class AttentionGuidedSubproblemSolverExtension:
    def _check_work_rest_pattern_constraint(self, current_label: Label, task: dict, crew: Crew) -> bool:
        """
        检查值四修二工作模式约束
        规则：连续工作不超过4天，工作4天后必须休息2天
        """
        if not hasattr(current_label, 'duty_days_count'):
            return True  # 如果没有值勤日计数，跳过检查
        
        task_date = task['startTime'].date()
        
        # 简化检查：如果当前已经连续工作了4天，且新任务不是休息，则违规
        if (current_label.duty_days_count >= 4 and 
            task['type'] in ['flight'] and  # 实际工作任务
            current_label.node.time.date() != task_date):  # 不是同一天的任务
            
            # 检查是否有足够的休息时间（简化为检查时间间隔）
            time_gap = task['startTime'] - current_label.node.time
            if time_gap.total_seconds() < 48 * 3600:  # 少于48小时休息
                return False
        
        return True

    def _is_flight_duty_day_ending(self, current_label, task):
        """
        检查飞行周期末尾是否为飞行值勤日
        根据比赛定义：
        1. 飞行值勤日必须包含飞行任务且从可过夜机场开始到可过夜机场结束
        2. 占位任务可以单独构成值勤日，但不能单独构成飞行值勤日
        3. 飞行周期末尾必须是飞行值勤日
        """
        # 如果当前任务是飞行任务，则当前值勤日肯定是飞行值勤日
        if task['type'] == 'flight':
            return True
        
        # 如果当前任务不是飞行任务，检查当前值勤日是否已经包含飞行任务
        if current_label.has_flown_in_duty:
            return True
        
        # 如果当前值勤日没有飞行任务，则不能作为飞行周期的末尾
        # 但占位任务可以单独构成值勤日（只是不能作为飞行值勤日）
        return False
    
    def _can_form_duty_day(self, current_label, task):
        """
        检查任务是否可以构成值勤日
        占位任务可以单独构成值勤日，但不是飞行值勤日
        """
        # 所有类型的任务都可以构成值勤日
        return True
    
    def _is_positioning_task(self, task):
        """
        检查是否为占位任务
        """
        return ('positioning' in task.get('type', '').lower() or 
                'deadhead' in task.get('type', '').lower() or
                task.get('type') == 'bus')
AttentionGuidedSubproblemSolver._check_work_rest_pattern_constraint = AttentionGuidedSubproblemSolverExtension._check_work_rest_pattern_constraint
AttentionGuidedSubproblemSolver._is_flight_duty_day_ending = AttentionGuidedSubproblemSolverExtension._is_flight_duty_day_ending



