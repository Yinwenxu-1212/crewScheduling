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

# 从data_models导入Label类
from data_models import Label

class AttentionGuidedSubproblemSolver:
    """使用注意力模型指导的子问题求解器"""
    
    def __init__(self, model_path: str = "models/best_model.pth"):
        """初始化求解器并加载预训练的注意力模型"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            # 直接加载state_dict，因为文件中保存的就是state_dict
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            if self.debug_log:
                self.debug_log.write(f"成功加载预训练模型: {model_path}\n")
        else:
            if self.debug_log:
                self.debug_log.write(f"警告：未找到预训练模型 {model_path}，使用随机初始化的模型\n")
            print(f"警告：未找到预训练模型 {model_path}，使用随机初始化的模型")
        
        # 注意力引导的参数
        self.max_candidates_per_expansion = 15  # 每次扩展最多考虑的候选任务数
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
        
        # 机场特征
        features[6] = hash(task['depaAirport']) % 1000
        features[7] = hash(task['arriAirport']) % 1000
        
        # 时间特征
        features[8] = task['startTime'].weekday()
        features[9] = task['startTime'].hour
        features[10] = task['endTime'].hour
        
        # 任务持续时间
        duration = (task['endTime'] - task['startTime']).total_seconds() / 3600
        features[11] = min(duration, 24)
        
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
            candidates_tensor = torch.FloatTensor(candidate_features).unsqueeze(0).to(self.device)  # (1, num_candidates, action_dim)
            action_mask = torch.ones(1, len(candidates)).to(self.device)  # (1, num_candidates) - 所有候选都有效
            
            # 使用注意力模型评分
            with torch.no_grad():
                dist, _ = self.model(state_tensor, candidates_tensor, action_mask)
                action_probs = dist.probs.squeeze(0).cpu().numpy()  # (num_candidates,)
            
            # 返回 (分数, 索引) 的列表，按分数降序排序
            scored_candidates = [(float(action_probs[i]), i) for i in range(len(candidates))]
            scored_candidates.sort(reverse=True, key=lambda x: x[0])
            
            return scored_candidates
            
        except Exception as e:
            print(f"Warning: Attention scoring failed: {e}. Using deterministic order.")
            # 使用确定性排序而不是随机排序
            return [(0.0, i) for i in range(len(candidates))]
    
    def solve_subproblem_with_attention(self, crew: Crew, flights: List[Flight],
                                      buses: List[BusInfo], ground_duties: List[GroundDuty],
                                      dual_prices: Dict[str, float], 
                                      planning_start_dt: datetime, planning_end_dt: datetime,
                                      layover_airports: Set[str], crew_sigma_dual: float) -> List[Roster]:
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
            last_base_return=planning_start_dt.date()
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
        # 增加搜索参数
        max_iterations = 80000  # 增加最大迭代次数
        max_rosters_per_crew = 30  # 增加每个机组的方案数量
        
        # 在attention模型初始化时增加候选数量
        self.max_candidates_per_expansion = 15  # 增加候选数量
        
        # 添加已找到方案的记录
        found_roster_signatures = set()
        
        # 添加调试计数器
        total_candidates_found = 0
        total_labels_processed = 0
        
        self._log_debug(f"\n=== 机组 {crew.crewId} 子问题求解开始 ===")
        self._log_debug(f"初始状态: 队列={len(labels)}, 任务={len(all_tasks)}")
        
        # 修改while循环条件，添加早停机制
        while (labels and 
               iteration_count < max_iterations and 
               len(found_rosters) < max_rosters_per_crew):
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
            
            # 检查是否到达规划结束时间或找到完整方案
            # 改进终止条件：确保值勤日完整性和合理的任务数量
            min_tasks_required = 6  # 提高最小任务数量要求
            
            # 检查是否可以终止：时间结束或返回基地且满足条件
            can_terminate = False
            if current_label.node.time >= planning_end_dt:
                can_terminate = True
            elif (current_label.node.airport == crew.base and 
                  len(current_label.path) >= min_tasks_required):
                # 额外检查：确保当前不在值勤日中间
                if (current_label.duty_start_time is None or 
                    current_label.node.time - current_label.duty_start_time >= timedelta(hours=MIN_REST_HOURS)):
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
                            temp_roster, crew, dual_prices, crew_sigma_dual
                        )
                        
                        # 添加详细的调试输出
                         # print(f"\n=== 机组 {crew.crewId} Roster调试信息 ===")
                         # print(f"任务路径: {[task['taskId'] for task in current_label.path]}")
                         # print(f"标签成本 (label cost): {current_label.cost:.6f}")
                         # print(f"完整成本计算结果:")
                         # print(f"  - 总成本 (total_cost): {cost_details['total_cost']:.6f}")
                         # print(f"  - Reduced Cost: {cost_details['reduced_cost']:.6f}")
                         # print(f"  - 飞行奖励: {cost_details['flight_reward']:.6f}")
                         # print(f"  - 对偶价格收益: {cost_details['dual_price_total']:.6f}")
                         # print(f"  - 机组对偶系数: {cost_details['crew_sigma_dual']:.6f}")
                         # print(f"  - 置位惩罚: {cost_details['positioning_penalty']:.6f}")
                         # print(f"  - 外站过夜惩罚: {cost_details['overnight_penalty']:.6f}")
                         # print(f"  - 其他成本: {cost_details['other_costs']:.6f}")
                        
                        # 使用计算出的成本创建最终roster
                        roster = Roster(crew.crewId, roster_tasks, cost_details['total_cost'])
                        found_rosters.append(roster)
            
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
            
            # 确定性选择：直接取前N个最高分的候选
            top_candidates = scored_candidates[:self.max_candidates_per_expansion]
            
            # 扩展标签
            for score, candidate_idx in top_candidates:
                task = candidates[candidate_idx]
                new_label = self._create_new_label(current_label, task, crew, tie_breaker)
                
                if new_label:
                    heapq.heappush(labels, (new_label.cost, new_label))
        
        self._log_debug(f"=== 机组 {crew.crewId} 求解完成 ===\n迭代: {iteration_count}, 方案: {len(found_rosters)}, 平均候选: {total_candidates_found/max(1, total_labels_processed):.1f}")
        
        return found_rosters
    
    def _get_valid_candidates(self, current_label: Label, all_tasks: List[Dict], 
                            crew: Crew, layover_airports: List[str], 
                            planning_end_dt: datetime) -> List:
        """获取当前标签的有效候选任务"""
        candidates = []
        current_time = current_label.node.time
        current_airport = current_label.node.airport
        
        # 添加过滤统计
        filter_stats = {
            'total_tasks': len(all_tasks),
            'already_used': 0,
            'time_constraint': 0,
            'location_constraint': 0,
            'connection_time': 0,
            'duty_constraint': 0,
            'overnight_constraint': 0,
            'valid_candidates': 0
        }
        
        for task in all_tasks:
            # 检查任务是否已被使用
            if task['taskId'] in current_label.used_task_ids:
                filter_stats['already_used'] += 1
                continue
            
            # 检查时间约束
            if task['startTime'] <= current_time or task['endTime'] > planning_end_dt:
                filter_stats['time_constraint'] += 1
                continue
            
            # 检查地点约束
            if task['depaAirport'] != current_airport:
                filter_stats['location_constraint'] += 1
                continue
            
            # 检查连接时间
            connection_time = task['startTime'] - current_time
            if task['type'] == 'flight':
                # 获取当前标签的最后一个任务的飞机尾号
                last_aircraft_no = None
                if current_label.path:
                    last_task = current_label.path[-1]
                    if last_task.get('type') == 'flight' and 'aircraftNo' in last_task:
                        last_aircraft_no = last_task['aircraftNo']
                
                # 根据飞机尾号是否相同选择最小连接时间
                if last_aircraft_no and task.get('aircraftNo') == last_aircraft_no:
                    min_connection_time = MIN_CONNECTION_TIME_FLIGHT_SAME_AIRCRAFT
                else:
                    min_connection_time = MIN_CONNECTION_TIME_FLIGHT_DIFFERENT_AIRCRAFT
                
                if connection_time < min_connection_time:
                    filter_stats['connection_time'] += 1
                    continue
            elif task['type'] == 'positioning_bus':
                if connection_time < MIN_CONNECTION_TIME_BUS:
                    filter_stats['connection_time'] += 1
                    continue
            
            # 检查值勤时间约束
            if not self._check_duty_constraints(current_label, task, crew):
                filter_stats['duty_constraint'] += 1
                continue
            
            # 检查过夜机场约束（规则10：过夜机场限制）
            if task['type'] == 'flight':
                arrival_airport = task['arriAirport']
                task_end_time = task['endTime']
                
                # 如果任务结束时间较晚（可能需要过夜）且不在基地
                if (arrival_airport != crew.base and 
                    task_end_time.hour >= 20):  # 晚上8点后到达视为可能过夜
                    # 检查该机场是否允许过夜
                    if arrival_airport not in layover_airports:
                        filter_stats['overnight_constraint'] += 1
                        continue
            
            candidates.append(task)
            filter_stats['valid_candidates'] += 1
        
        # 输出过滤统计（仅在前几次调用时）
        if len(candidates) == 0 and current_label.node.time.hour < 12:  # 只在早期时间输出
            self._log_debug(f"      候选任务过滤统计 - 位置: {current_airport}, 时间: {current_time}")
            self._log_debug(f"        总任务数: {filter_stats['total_tasks']}")
            self._log_debug(f"        已使用: {filter_stats['already_used']}")
            self._log_debug(f"        时间约束过滤: {filter_stats['time_constraint']}")
            self._log_debug(f"        地点约束过滤: {filter_stats['location_constraint']}")
            self._log_debug(f"        连接时间过滤: {filter_stats['connection_time']}")
            self._log_debug(f"        值勤约束过滤: {filter_stats['duty_constraint']}")
            self._log_debug(f"        过夜约束过滤: {filter_stats['overnight_constraint']}")
            self._log_debug(f"        有效候选: {filter_stats['valid_candidates']}")
        
        return candidates
    
    def _check_duty_constraints(self, current_label: Label, task: Dict, crew: Crew = None) -> bool:
        """检查值勤时间相关约束"""
        # 检查总飞行时间约束（规则9：总飞行值勤时间限制）
        if task['type'] == 'flight':
            potential_total_flight_hours = current_label.total_flight_hours + task.get('flyTime', 0) / 60.0
            if potential_total_flight_hours > MAX_TOTAL_FLIGHT_HOURS:
                return False
        
        # 检查飞行周期约束（规则11：飞行周期限制）
        task_date = task['startTime'].date()
        crew_base = crew.base if crew else ''
        if (current_label.current_cycle_start is not None and 
            task['arriAirport'] != crew_base):
            # 计算潜在的周期天数
            potential_cycle_days = (task_date - current_label.current_cycle_start).days + 1
            if potential_cycle_days > 4:  # MAX_FLIGHT_CYCLE_DAYS
                return False
        
        # 检查基地休息时间约束（规则12：周期间休息）
        if (current_label.last_base_return is not None and 
            task['type'] == 'flight' and 
            task['depaAirport'] != crew_base):
            # 如果要开始新的飞行周期，检查距离上次基地休息的时间
            days_since_base = (task_date - current_label.last_base_return).days
            if current_label.current_cycle_start is None and days_since_base < 2:  # MIN_CYCLE_REST_DAYS
                return False
        
        if current_label.duty_start_time is None:
            return True  # 新值勤日，无约束
        
        # 检查值勤时间是否超限
        potential_duty_end = task['endTime']
        duty_duration = (potential_duty_end - current_label.duty_start_time).total_seconds() / 3600
        if duty_duration > MAX_DUTY_DAY_HOURS:
            return False
        
        # 检查任务数量限制
        if current_label.duty_task_count >= MAX_TASKS_IN_DUTY:
            return False
        
        # 检查航班数量限制
        if task['type'] == 'flight' and current_label.duty_flight_count >= MAX_FLIGHTS_IN_DUTY:
            return False
        
        # 检查值勤内飞行时间限制
        if task['type'] == 'flight':
            potential_duty_flight_time = current_label.duty_flight_time + task.get('flyTime', 0) / 60.0
            if potential_duty_flight_time > MAX_FLIGHT_TIME_IN_DUTY_HOURS:
                return False
        
        return True
    
    def _calculate_roster_reward(self, roster_tasks):
        """
        计算roster的正确奖励，基于值勤日日均飞时
        """
        total_flight_hours = 0.0
        duty_calendar_days = set()
        
        for task in roster_tasks:
            if task['type'] == 'flight':
                total_flight_hours += task.get('flyTime', 0) / 60.0
                
                # 计算值勤日历日
                start_date = task['startTime'].date()
                end_date = task['endTime'].date()
                for day in range((end_date - start_date).days + 1):
                    duty_calendar_days.add(start_date + timedelta(days=day))
        
        total_duty_days = len(duty_calendar_days)
        if total_duty_days > 0:
            avg_daily_fly_time = total_flight_hours / total_duty_days
            return -(avg_daily_fly_time * 1000)  # 负值表示奖励
        return 0.0

    def _create_new_label(self, current_label: Label, task: Dict, 
                     crew: Crew, tie_breaker) -> Optional[Label]:
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
            
            # 检查是否需要开始新值勤日
            new_duty_start_time = current_label.duty_start_time
            if current_label.duty_start_time is None:
                new_duty_start_time = task['startTime']
            else:
                # 检查是否需要休息（结束当前值勤日）
                rest_time = task['startTime'] - current_label.node.time
                if rest_time >= timedelta(hours=MIN_REST_HOURS):
                    new_duty_start_time = task['startTime']
                    # 检查外站过夜
                    if current_label.node.airport != crew.base:
                        overnight_days = (task['startTime'].date() - current_label.node.time.date()).days
                        if overnight_days > 0:
                            cost_delta += PENALTY_PER_AWAY_OVERNIGHT * overnight_days
            
            # 更新值勤相关计数器
            new_duty_flight_time = current_label.duty_flight_time
            new_duty_flight_count = current_label.duty_flight_count
            new_duty_task_count = current_label.duty_task_count
            
            if new_duty_start_time == task['startTime']:  # 新值勤日
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
            
            # 双重检查：确保任务未被使用（防止重复）
            if task['taskId'] in current_label.used_task_ids:
                return None  # 任务已被使用，不创建新标签
            
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
                last_base_return=new_last_base_return
            )
            
            return new_label
            
        except Exception as e:
            print(f"Error creating new label: {e}")
            return None

def solve_subproblem_for_crew_with_attention(
    crew: Crew, all_flights: List[Flight], all_bus_info: List[BusInfo],
    crew_ground_duties: List[GroundDuty], dual_prices: Dict[str, float],
    layover_stations: Dict[str, dict], crew_leg_match_dict: Dict[str, List[str]],
    crew_sigma_dual: float
) -> List[Roster]:
    """使用注意力模型指导的子问题求解包装函数"""
    # layover_stations 已经是机场代码的集合，直接使用
    layover_airports = layover_stations
    
    # 添加缺失的planning日期定义
    from datetime import datetime
    planning_start_dt = datetime.strptime("2025-05-29 00:00:00", "%Y-%m-%d %H:%M:%S")
    planning_end_dt = datetime.strptime("2025-06-05 00:00:00", "%Y-%m-%d %H:%M:%S")
    
    # 定义模型路径
    model_path = "models/best_model.pth"
    
    solver = AttentionGuidedSubproblemSolver(model_path)
    return solver.solve_subproblem_with_attention(
        crew, all_flights, all_bus_info, crew_ground_duties, dual_prices, 
        planning_start_dt, planning_end_dt, layover_airports, crew_sigma_dual
    )



