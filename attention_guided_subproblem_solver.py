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

# 继承原有的常量和规则
from subproblem_solver import (
    TRAINING_DATA_FILE, CSV_HEADER, REWARD_PER_FLIGHT_HOUR, 
    PENALTY_PER_AWAY_OVERNIGHT, PENALTY_PER_POSITIONING,
    MIN_CONNECTION_TIME_FLIGHT_DIFFERENT_AIRCRAFT, MIN_CONNECTION_TIME_BUS,
    MAX_DUTY_DAY_HOURS, MIN_REST_HOURS, MAX_FLIGHTS_IN_DUTY, 
    MAX_TASKS_IN_DUTY, MAX_FLIGHT_TIME_IN_DUTY_HOURS,
    is_conflicting, find_positioning_tasks
)

# 从data_models导入Label类
from data_models import Label

class AttentionGuidedSubproblemSolver:
    """使用注意力模型指导的子问题求解器"""
    
    def __init__(self, model_path: str = "models/best_model.pth"):
        """初始化求解器并加载预训练的注意力模型"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
        else:
            print(f"Warning: Model file {model_path} not found. Using random initialization.")
        
        # 注意力引导的参数
        self.max_candidates_per_expansion = 10  # 每次扩展最多考虑的候选任务数
        self.use_attention_guidance = True
    
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
    
    def _score_candidates_with_attention(self, candidates: List, current_label: Label, crew: Crew) -> List[Tuple[float, int]]:
        """使用注意力模型为候选任务打分"""
        if not candidates or not self.use_attention_guidance:
            return [(0.0, i) for i in range(len(candidates))]
        
        try:
            # 提取状态特征
            state_features = self._extract_state_features(current_label, crew)
            state_tensor = torch.FloatTensor(state_features).unsqueeze(0).to(self.device)
            
            # 提取所有候选任务的特征
            action_features = []
            for task in candidates:
                task_features = self._extract_task_features(task, current_label)
                action_features.append(task_features)
            
            if not action_features:
                return [(0.0, i) for i in range(len(candidates))]
            
            action_tensor = torch.FloatTensor(action_features).unsqueeze(0).to(self.device)
            
            # 创建action_mask，所有候选动作都有效
            action_mask = torch.ones(1, len(action_features)).to(self.device)
            
            # 使用注意力模型计算动作概率
            with torch.no_grad():
                action_dist, _ = self.model(state_tensor, action_tensor, action_mask)
                # 从Categorical分布中获取概率
                scores = action_dist.probs.squeeze(0).cpu().numpy()
            
            # 返回 (分数, 索引) 的列表，按分数降序排序
            scored_candidates = [(scores[i], i) for i in range(len(candidates))]
            scored_candidates.sort(reverse=True, key=lambda x: x[0])
            
            # 添加随机扰动以增加多样性
            # 对分数相近的候选任务添加小幅随机扰动
            for i in range(len(scored_candidates)):
                score, idx = scored_candidates[i]
                # 添加小幅随机扰动 (-0.1 到 +0.1)
                noise = random.uniform(-0.1, 0.1)
                scored_candidates[i] = (score + noise, idx)
            
            # 重新排序
            scored_candidates.sort(reverse=True, key=lambda x: x[0])
            
            return scored_candidates
            
        except Exception as e:
            print(f"Warning: Attention scoring failed: {e}. Using random order.")
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
            tie_breaker=next(tie_breaker)
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
        
        # 主循环
        iteration_count = 0
        # 增加搜索参数
        max_iterations = 50000  # 增加最大迭代次数
        max_rosters_per_crew = 20  # 增加每个机组的方案数量
        
        # 在attention模型初始化时增加候选数量
        self.max_candidates_per_expansion = 10  # 如果当前值较小，可以增加到10-15
        
        # 添加已找到方案的记录
        found_roster_signatures = set()
        
        # 修改while循环条件，添加早停机制
        while (labels and 
               iteration_count < max_iterations and 
               len(found_rosters) < max_rosters_per_crew):
            iteration_count += 1
            
            current_cost, current_label = heapq.heappop(labels)
            
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
            
            # 检查是否找到负成本方案
            if current_label.cost < -1e-6:
                # 生成方案签名
                task_ids = tuple(sorted(task_info['taskId'] for task_info in current_label.path))
                roster_signature = (crew.crewId, task_ids)
                
                # 只添加未见过的方案
                if roster_signature not in found_roster_signatures:
                    found_roster_signatures.add(roster_signature)
                    
                    # 构建排班方案
                    roster_tasks = []
                    for task_info in current_label.path:
                        if task_info['type'] == 'flight':
                            flight_obj = next(f for f in flights if f.id == task_info['taskId'])
                            roster_tasks.append(flight_obj)
                        elif task_info['type'] == 'positioning_bus':
                            bus_obj = next(b for b in buses if b.id == task_info['taskId'])
                            roster_tasks.append(bus_obj)
                    
                    if roster_tasks:
                        roster_cost = sum(getattr(task, 'cost', 0) for task in roster_tasks)
                        roster = Roster(crew.crewId, roster_tasks, roster_cost)
                        found_rosters.append(roster)
                        
                        # 移除这行：return found_rosters
                        # 继续搜索更多方案
            
            # 获取候选任务
            candidates = self._get_valid_candidates(
                current_label, all_tasks, crew, layover_airports, planning_end_dt
            )
            
            if not candidates:
                continue
            
            # 使用注意力模型对候选任务进行评分和排序
            scored_candidates = self._score_candidates_with_attention(candidates, current_label, crew)
            
            # 增加搜索多样性：在前50%的候选中随机选择
            if len(scored_candidates) > self.max_candidates_per_expansion:
                # 取前50%的高分候选
                top_half_count = max(1, len(scored_candidates) // 2)
                top_half = scored_candidates[:top_half_count]
                
                # 从前50%中随机选择
                selection_count = min(self.max_candidates_per_expansion, len(top_half))
                top_candidates = random.sample(top_half, selection_count)
            else:
                top_candidates = scored_candidates[:self.max_candidates_per_expansion]
            
            # 扩展标签
            for score, candidate_idx in top_candidates:
                task = candidates[candidate_idx]
                new_label = self._create_new_label(current_label, task, crew, tie_breaker)
                
                if new_label:
                    heapq.heappush(labels, (new_label.cost, new_label))
        
        return found_rosters
    
    def _get_valid_candidates(self, current_label: Label, all_tasks: List, 
                            crew: Crew, layover_airports: Set[str], 
                            planning_end_dt: datetime) -> List:
        """获取当前标签的有效候选任务"""
        candidates = []
        current_time = current_label.node.time
        current_airport = current_label.node.airport
        
        for task in all_tasks:
            # 检查任务是否已被使用
            if task['taskId'] in current_label.used_task_ids:
                continue
            
            # 检查时间约束
            if task['startTime'] <= current_time or task['endTime'] > planning_end_dt:
                continue
            
            # 检查地点约束
            if task['depaAirport'] != current_airport:
                continue
            
            # 检查连接时间
            connection_time = task['startTime'] - current_time
            if task['type'] == 'flight':
                if connection_time < MIN_CONNECTION_TIME_FLIGHT_DIFFERENT_AIRCRAFT:
                    continue
            elif connection_time < MIN_CONNECTION_TIME_BUS:
                continue
            
            # 检查值勤时间约束
            if not self._check_duty_constraints(current_label, task):
                continue
            
            candidates.append(task)
        
        return candidates
    
    def _check_duty_constraints(self, current_label: Label, task: Dict) -> bool:
        """检查值勤时间相关约束"""
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
    
    def _create_new_label(self, current_label: Label, task: Dict, 
                     crew: Crew, tie_breaker) -> Optional[Label]:
        """基于当前标签和新任务创建新标签"""
        try:
            # 计算新的节点
            new_node = Node(task['arriAirport'], task['endTime'])
            
            # 计算成本增量 - 确保使用任务中存储的最新对偶价格
            cost_delta = 0.0
            if task['type'] == 'flight':
                # 使用任务字典中存储的最新对偶价格
                dual_price = task.get('dual_price', 0.0)
                flight_cost = REWARD_PER_FLIGHT_HOUR * (task.get('flyTime', 0) / 60.0)
                
                cost_delta -= dual_price  # 航班的收益（负成本）
                cost_delta += flight_cost
                
            elif 'positioning' in task['type']:
                # 调机的惩罚
                cost_delta += PENALTY_PER_POSITIONING
            
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
                tie_breaker=next(tie_breaker)
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
    from datetime import datetime
    
    # 设置规划时间范围
    planning_start_dt = datetime(2025, 5, 31, 0, 0)
    planning_end_dt = datetime(2025, 6, 7, 23, 59)
    
    # layover_stations 已经是机场代码的集合，直接使用
    layover_airports = layover_stations
    
    # 定义模型路径
    model_path = "models/best_model.pth"
    
    solver = AttentionGuidedSubproblemSolver(model_path)
    return solver.solve_subproblem_with_attention(
        crew, all_flights, all_bus_info, crew_ground_duties, dual_prices, 
        planning_start_dt, planning_end_dt, layover_airports, crew_sigma_dual
    )
