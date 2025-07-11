# optimized_environment.py
"""
优化的环境类，集成预计算和批量处理
"""
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import config
from precompute_features import PrecomputeManager
from flight_cycle_constraints import FlightCycleConstraints

class OptimizedCrewRosteringEnv:
    """优化的机组排班环境"""
    
    def __init__(self, data_handler):
        self.dh = data_handler
        self.precompute_manager = PrecomputeManager(data_handler)
        
        # 尝试加载缓存，如果失败则重新计算
        if not self.precompute_manager.load_cache():
            self.precompute_manager.precompute_all_features()
            
        # 初始化飞行周期约束检查器
        self.cycle_constraints = FlightCycleConstraints(data_handler)
            
        # 初始化环境状态
        self._init_environment()
        
        # 批量处理优化
        self.batch_size = 32  # 批量处理大小
        self.action_cache = {}  # 动作缓存
        
    def _init_environment(self):
        """初始化环境"""
        self.planning_start_dt = datetime.strptime(config.PLANNING_START_DATE, '%Y-%m-%d %H:%M:%S')
        self.planning_end_dt = datetime.strptime(config.PLANNING_END_DATE, '%Y-%m-%d %H:%M:%S')
        
        # 使用集合加速查找
        self.layover_stations_set = set(self.dh.layover_stations['airport'].tolist())
        self.unassigned_flight_ids = set(self.dh.data['flights']['id'].tolist())
        self.assigned_positioning_ids = set()
        
        # 预构建索引
        self._build_indices()
        
    def _build_indices(self):
        """构建高效索引"""
        # 按时间和机场索引任务
        self.tasks_by_time = defaultdict(list)
        self.tasks_by_airport = defaultdict(list)
        
        # 索引航班
        for _, flight in self.dh.data['flights'].iterrows():
            time_key = flight['std'].strftime('%Y-%m-%d-%H')
            self.tasks_by_time[time_key].append(('flight', flight))
            self.tasks_by_airport[flight['depaAirport']].append(('flight', flight))
            
        # 索引大巴
        for _, bus in self.dh.data['bus_info'].iterrows():
            time_key = bus['td'].strftime('%Y-%m-%d-%H')
            self.tasks_by_time[time_key].append(('bus', bus))
            self.tasks_by_airport[bus['depaAirport']].append(('bus', bus))
            
    def get_valid_actions_batch(self, crew_states, crew_infos):
        """批量获取有效动作（优化版本）"""
        batch_valid_actions = []
        
        for crew_state, crew_info in zip(crew_states, crew_infos):
            # 使用缓存
            cache_key = self._get_state_cache_key(crew_state, crew_info)
            if cache_key in self.action_cache:
                batch_valid_actions.append(self.action_cache[cache_key])
                continue
                
            valid_actions = self._get_valid_actions_single(crew_state, crew_info)
            self.action_cache[cache_key] = valid_actions
            batch_valid_actions.append(valid_actions)
            
        return batch_valid_actions
        
    def _get_valid_actions_single(self, crew_state, crew_info):
        """获取单个机组的有效动作（优化版本）"""
        valid_actions = []
        current_time = crew_state['last_task_end_time']
        current_location = crew_state['last_location']
        
        # 使用时间窗口限制搜索范围
        search_window_hours = 24
        end_time = current_time + timedelta(hours=search_window_hours)
        
        # 1. 获取时间窗口内的候选任务
        candidate_tasks = self._get_tasks_in_window(current_time, end_time, current_location)
        
        # 2. 批量检查约束
        for task_type, task in candidate_tasks:
            if self._quick_feasibility_check(task_type, task, crew_state, crew_info):
                valid_actions.append({
                    'type': task_type,
                    'task': task,
                    'score': self._calculate_action_score(task_type, task, crew_state, crew_info)
                })
                
        # 3. 根据分数排序，只保留前N个
        valid_actions.sort(key=lambda x: x['score'], reverse=True)
        return valid_actions[:config.MAX_CANDIDATE_ACTIONS]
        
    def _get_tasks_in_window(self, start_time, end_time, location):
        """获取时间窗口内的任务"""
        tasks = []
        
        # 使用预构建的索引快速查找
        current = start_time
        while current <= end_time:
            time_key = current.strftime('%Y-%m-%d-%H')
            if time_key in self.tasks_by_time:
                for task_type, task in self.tasks_by_time[time_key]:
                    # 位置过滤
                    if task_type == 'flight' and task['depaAirport'] == location:
                        if task['id'] in self.unassigned_flight_ids:
                            tasks.append((task_type, task))
                    elif task_type == 'bus' and task['depaAirport'] == location:
                        tasks.append((task_type, task))
                        
            current += timedelta(hours=1)
            
        return tasks
        
    def _quick_feasibility_check(self, task_type, task, crew_state, crew_info):
        """快速可行性检查（集成飞行周期约束）"""
        # 使用预计算的兼容性矩阵
        task_id = f"{task_type}_{task['id']}"
        if not self.precompute_manager.is_compatible(crew_info['crewId'], task_id):
            return False
            
        # 时间约束快速检查
        if task_type == 'flight':
            task_start = task['std']
        else:  # bus
            task_start = task['td']
            
        if crew_state.get('last_task_end_time'):
            connection_time = task_start - crew_state['last_task_end_time']
            if connection_time < timedelta(hours=2):  # 最小连接时间
                return False
                
        # 使用新的约束检查器进行详细检查
        # 1. 检查最小休息时间约束
        if not self.cycle_constraints.check_min_rest_before_duty(crew_state, task):
            return False
            
        # 2. 检查最大值勤时间约束
        if not self.cycle_constraints.check_max_duty_time(crew_state, task):
            return False
            
        # 3. 检查飞行周期开始条件（如果适用）
        if crew_state.get('is_cycle_start_candidate', False):
            if not self.cycle_constraints.check_flight_cycle_start(crew_state, crew_info):
                return False
                
        return True
        
    def _calculate_action_score(self, task_type, task, crew_state, crew_info):
        """计算动作分数（用于优先级排序）"""
        score = 0.0
        
        # 飞行任务优先
        if task_type == 'flight':
            score += 100.0
            # 使用预计算的航班链信息
            chains = self.precompute_manager.get_flight_chains(task['id'])
            if chains:
                score += len(chains) * 10  # 有后续连接的航班得分更高
                
        # 机场重要性
        airport_importance = self.precompute_manager.get_airport_importance(task.get('arriAirport', task.get('depaAirport')))
        score += airport_importance * 20
        
        # 时间紧迫性
        if task_type == 'flight':
            time_to_departure = (task['std'] - crew_state['last_task_end_time']).total_seconds() / 3600
        else:
            time_to_departure = (task['td'] - crew_state['last_task_end_time']).total_seconds() / 3600
        score += max(0, 24 - time_to_departure) * 2  # 越紧急得分越高
        
        return score
        
    def _get_state_cache_key(self, crew_state, crew_info):
        """生成状态缓存键"""
        return (
            crew_info['crewId'],
            crew_state.get('last_location', ''),
            crew_state.get('last_task_end_time', datetime.min).strftime('%Y%m%d%H'),
            crew_state.get('duty_flight_count', 0),
            crew_state.get('total_flight_hours', 0) // 10  # 粗粒度
        )
        
    def task_to_feature_vector_batch(self, tasks, crew_states, crew_infos):
        """批量转换任务特征向量"""
        batch_features = []
        
        for task, crew_state, crew_info in zip(tasks, crew_states, crew_infos):
            features = self._task_to_feature_vector_optimized(task, crew_state, crew_info)
            batch_features.append(features)
            
        return np.array(batch_features)
        
    def _task_to_feature_vector_optimized(self, task, crew_state, crew_info):
        """优化的任务特征向量转换"""
        features = np.zeros(20)  # 根据您的ACTION_DIM
        
        # 获取预计算的基础特征
        task_id = f"{task['type']}_{task.get('taskId', task.get('id'))}"
        precomputed = self.precompute_manager.get_task_features(task_id)
        
        if precomputed:
            # 使用预计算特征
            features[9] = precomputed['day_of_week']
            features[10] = precomputed['hour_of_day']
            features[11] = precomputed['is_weekend']
            features[12] = precomputed['duration_hours']
            features[7] = precomputed['depa_airport_hash']
            features[8] = precomputed['arri_airport_hash']
            
        # 动态特征（仍需实时计算）
        connection_time = (task['startTime'] - crew_state['last_task_end_time']).total_seconds() / 3600
        features[0] = min(connection_time, 48)
        
        # 任务类型特征
        if task['type'] == 'flight':
            features[1] = 1
            features[2] = precomputed.get('fly_time_hours', 0)
        elif 'positioning' in task.get('type', ''):
            features[3] = 1
            
        # 添加更多动态特征...
        
        return features