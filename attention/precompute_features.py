# precompute_features.py
"""
批量预计算特征模块，大幅提升训练效率
"""
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import pickle
import os
from optimized_unified_config import OptimizedUnifiedConfig
from flight_cycle_constraints import FlightCycleConstraints

class PrecomputeManager:
    """批量预计算管理器"""
    
    def __init__(self, data_handler):
        self.dh = data_handler
        self.config = OptimizedUnifiedConfig
        
        # 初始化飞行周期约束检查器
        self.cycle_constraints = FlightCycleConstraints(data_handler)
        
        # 预计算的特征缓存
        self.compatibility_matrix = {}
        self.task_features = {}
        self.crew_features = {}
        self.airport_features = {}
        self.cycle_features = {}  # 新增：飞行周期相关特征
        
        # 缓存文件路径
        self.cache_dir = "cache"
        self.cache_file = os.path.join(self.cache_dir, "precomputed_features.pkl")
        
        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 保持向后兼容
        self.features_cache = {}
        
    def precompute_all_features(self):
        """预计算所有特征"""
        print("开始批量预计算特征...")
        
        # 1. 预计算任务特征
        self._precompute_task_features()
        
        # 2. 预计算航班链
        self._precompute_flight_chains()
        
        # 3. 预计算机组-任务兼容性矩阵
        self._precompute_crew_task_compatibility()
        
        # 4. 预计算时空连接图
        self._precompute_spatiotemporal_graph()
        
        # 5. 预计算飞行周期特征
        self._precompute_cycle_features()
        
        # 6. 保存缓存
        self._save_cache()
        
        print("预计算完成！")
        
    def _precompute_task_features(self):
        """预计算所有任务的基础特征"""
        print("预计算任务特征...")
        
        task_features = {}
        
        # 处理航班
        for _, flight in self.dh.data['flights'].iterrows():
            task_id = f"flight_{flight['id']}"
            features = {
                'day_of_week': flight['std'].weekday() / 6.0,
                'hour_of_day': flight['std'].hour / 23.0,
                'is_weekend': 1.0 if flight['std'].weekday() >= 5 else 0.0,
                'duration_hours': (flight['sta'] - flight['std']).total_seconds() / 3600.0,
                'fly_time_hours': flight['flyTime'] / 60.0,
                'depa_airport_hash': hash(flight['depaAirport']) % 1000,
                'arri_airport_hash': hash(flight['arriAirport']) % 1000,
            }
            task_features[task_id] = features
            
        # 处理大巴置位
        for _, bus in self.dh.data['bus_info'].iterrows():
            task_id = f"bus_{bus['id']}"
            features = {
                'day_of_week': bus['td'].weekday() / 6.0,
                'hour_of_day': bus['td'].hour / 23.0,
                'is_weekend': 1.0 if bus['td'].weekday() >= 5 else 0.0,
                'duration_hours': (bus['ta'] - bus['td']).total_seconds() / 3600.0,
                'depa_airport_hash': hash(bus['depaAirport']) % 1000,
                'arri_airport_hash': hash(bus['arriAirport']) % 1000,
            }
            task_features[task_id] = features
            
        self.features_cache['task_features'] = task_features
        
    def _precompute_flight_chains(self):
        """预计算航班链（高效版本）"""
        print("预计算航班链...")
        
        # 按机场和时间组织航班
        airport_flights = defaultdict(list)
        flights_df = self.dh.data['flights']
        
        for _, flight in flights_df.iterrows():
            airport_flights[flight['depaAirport']].append({
                'id': flight['id'],
                'std': flight['std'],
                'sta': flight['sta'],
                'arriAirport': flight['arriAirport'],
                'flyTime': flight['flyTime'],
                'aircraftNo': flight['aircraftNo']
            })
            
        # 为每个机场的航班按时间排序
        for airport in airport_flights:
            airport_flights[airport].sort(key=lambda x: x['std'])
            
        # 构建航班链
        flight_chains = defaultdict(list)
        max_connection_time = timedelta(hours=4)  # 最大连接时间
        
        for airport, flights in airport_flights.items():
            for i, flight1 in enumerate(flights):
                chains_from_flight = []
                
                # 查找可以连接的后续航班
                dest_airport = flight1['arriAirport']
                if dest_airport in airport_flights:
                    for flight2 in airport_flights[dest_airport]:
                        # 检查时间连接
                        if flight2['std'] > flight1['sta']:
                            connection_time = flight2['std'] - flight1['sta']
                            if connection_time <= max_connection_time:
                                # 计算连接质量分数
                                same_aircraft = 1.0 if flight1['aircraftNo'] == flight2['aircraftNo'] else 0.0
                                time_score = 1.0 - (connection_time.total_seconds() / max_connection_time.total_seconds())
                                
                                chains_from_flight.append({
                                    'next_flight': flight2['id'],
                                    'connection_time': connection_time.total_seconds() / 3600.0,
                                    'same_aircraft': same_aircraft,
                                    'quality_score': time_score * 0.7 + same_aircraft * 0.3
                                })
                                
                # 只保留质量最高的前5个连接
                chains_from_flight.sort(key=lambda x: x['quality_score'], reverse=True)
                flight_chains[flight1['id']] = chains_from_flight[:5]
                
        self.features_cache['flight_chains'] = flight_chains
        
    def _precompute_crew_task_compatibility(self):
        """预计算机组-任务兼容性矩阵"""
        print("预计算机组-任务兼容性...")
        
        # 创建稀疏矩阵存储兼容性
        compatibility_matrix = {}
        
        # 处理机组-航班匹配
        for crew_id, flight_ids in self.dh.crew_leg_map.items():
            for flight_id in flight_ids:
                key = (crew_id, f"flight_{flight_id}")
                compatibility_matrix[key] = 1.0
                
        # 所有机组都可以执行大巴置位
        for crew_id in self.dh.data['crews']['crewId']:
            for _, bus in self.dh.data['bus_info'].iterrows():
                key = (crew_id, f"bus_{bus['id']}")
                compatibility_matrix[key] = 1.0
                
        self.features_cache['compatibility_matrix'] = compatibility_matrix
        
    def _precompute_spatiotemporal_graph(self):
        """预计算时空连接图"""
        print("预计算时空连接图...")
        
        # 构建机场之间的连接关系
        airport_connections = defaultdict(set)
        
        # 从航班数据构建
        for _, flight in self.dh.data['flights'].iterrows():
            airport_connections[flight['depaAirport']].add(flight['arriAirport'])
            
        # 从大巴数据构建
        for _, bus in self.dh.data['bus_info'].iterrows():
            airport_connections[bus['depaAirport']].add(bus['arriAirport'])
            
        # 计算机场重要性（基于连接数）
        airport_importance = {}
        for airport, connections in airport_connections.items():
            airport_importance[airport] = len(connections) / len(airport_connections)
            
        self.features_cache['airport_connections'] = dict(airport_connections)
        self.features_cache['airport_importance'] = airport_importance
        
    def _precompute_cycle_features(self):
        """预计算飞行周期相关特征"""
        print("预计算飞行周期特征...")
        
        cycle_features = {}
        
        # 为每个机组预计算飞行周期信息
        for crew_id in self.dh.data['crews']['crewId']:
            crew_cycle_info = {
                'max_flight_hours': self.cycle_constraints.get_max_flight_hours(crew_id),
                'max_duty_hours': self.cycle_constraints.get_max_duty_hours(crew_id),
                'min_rest_hours': self.cycle_constraints.get_min_rest_hours(crew_id),
                'current_cycle_position': self.cycle_constraints.get_cycle_position(crew_id)
            }
            cycle_features[crew_id] = crew_cycle_info
            
        # 为每个航班预计算周期影响
        for _, flight in self.dh.data['flights'].iterrows():
            flight_id = flight['id']
            flight_cycle_info = {
                'flight_hours': flight['flyTime'] / 60.0,
                'duty_hours': (flight['sta'] - flight['std']).total_seconds() / 3600.0,
                'is_night_flight': self._is_night_flight(flight),
                'cycle_impact_score': self._calculate_cycle_impact(flight)
            }
            cycle_features[f"flight_{flight_id}"] = flight_cycle_info
            
        self.features_cache['cycle_features'] = cycle_features
        self.cycle_features = cycle_features
        
    def _is_night_flight(self, flight):
        """判断是否为夜间航班"""
        std_hour = flight['std'].hour
        sta_hour = flight['sta'].hour
        return std_hour >= 22 or std_hour <= 6 or sta_hour >= 22 or sta_hour <= 6
        
    def _calculate_cycle_impact(self, flight):
        """计算航班对飞行周期的影响分数"""
        # 基于飞行时间、时段等因素计算影响分数
        flight_hours = flight['flyTime'] / 60.0
        is_night = self._is_night_flight(flight)
        
        impact_score = flight_hours * 0.6
        if is_night:
            impact_score *= 1.3  # 夜间航班影响更大
            
        return min(impact_score, 1.0)  # 归一化到[0,1]
        
    def _save_cache(self):
        """保存预计算结果"""
        cache_data = {
            'compatibility_matrix': self.compatibility_matrix,
            'task_features': self.task_features,
            'crew_features': self.crew_features,
            'airport_features': self.airport_features,
            'cycle_features': self.cycle_features,
            'features_cache': self.features_cache  # 保持向后兼容
        }
        
        os.makedirs('cache', exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
            
        print(f"缓存已保存到 {self.cache_file}")
            
    def load_cache(self):
        """加载预计算结果"""
        if not os.path.exists(self.cache_file):
            return False
            
        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # 处理新格式的缓存数据
            if isinstance(cache_data, dict) and 'features_cache' in cache_data:
                self.compatibility_matrix = cache_data.get('compatibility_matrix', {})
                self.task_features = cache_data.get('task_features', {})
                self.crew_features = cache_data.get('crew_features', {})
                self.airport_features = cache_data.get('airport_features', {})
                self.cycle_features = cache_data.get('cycle_features', {})
                self.features_cache = cache_data.get('features_cache', {})
            else:
                # 向后兼容旧格式
                self.features_cache = cache_data
            
            print(f"缓存已从 {self.cache_file} 加载")
            return True
            
        except Exception as e:
            print(f"加载缓存失败: {e}")
            return False
        
    def get_task_features(self, task_id):
        """获取任务特征"""
        return self.features_cache.get('task_features', {}).get(task_id, {})
        
    def get_flight_chains(self, flight_id):
        """获取航班链"""
        return self.features_cache.get('flight_chains', {}).get(flight_id, [])
        
    def is_compatible(self, crew_id, task_id):
        """检查机组-任务兼容性"""
        return (crew_id, task_id) in self.features_cache.get('compatibility_matrix', {})
        
    def get_airport_importance(self, airport):
        """获取机场重要性"""
        return self.features_cache.get('airport_importance', {}).get(airport, 0.0)
        
    def get_cycle_features(self, entity_id):
        """获取飞行周期特征"""
        return self.cycle_features.get(entity_id, {})
        
    def check_cycle_feasibility(self, crew_id, flight_id):
        """检查机组执行航班的周期可行性"""
        return self.cycle_constraints.check_flight_feasibility(crew_id, flight_id)
        
    def get_cycle_violation_penalty(self, crew_id, flight_id):
        """获取周期违规惩罚分数"""
        if self.check_cycle_feasibility(crew_id, flight_id):
            return 0.0
        else:
            # 根据违规程度计算惩罚
            crew_features = self.get_cycle_features(crew_id)
            flight_features = self.get_cycle_features(f"flight_{flight_id}")
            
            if not crew_features or not flight_features:
                return 1.0  # 最大惩罚
                
            # 计算具体的违规惩罚
            penalty = 0.0
            
            # 飞行时间超限惩罚
            if 'max_flight_hours' in crew_features and 'flight_hours' in flight_features:
                if flight_features['flight_hours'] > crew_features['max_flight_hours']:
                    penalty += 0.5
                    
            # 值勤时间超限惩罚
            if 'max_duty_hours' in crew_features and 'duty_hours' in flight_features:
                if flight_features['duty_hours'] > crew_features['max_duty_hours']:
                    penalty += 0.3
                    
            # 休息时间不足惩罚
            if 'min_rest_hours' in crew_features:
                penalty += 0.2  # 简化处理
                
            return min(penalty, 1.0)