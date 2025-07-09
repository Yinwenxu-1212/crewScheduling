# dynamic_airport_analyzer.py
# 动态机场分析器 - 基于数据自动分析机场重要性，避免硬编码

import pandas as pd
import numpy as np
from typing import Dict, Set, Tuple, List
import os
from collections import defaultdict
import json

class DynamicAirportAnalyzer:
    """动态机场分析器 - 自动从数据中分析机场重要性和分类"""
    
    def __init__(self, data_path: str = "data"):
        self.data_path = data_path
        self.analysis_cache = {}
        self.config_cache_file = os.path.join(data_path, ".airport_analysis_cache.json")
        
    def analyze_airport_importance(self, force_refresh: bool = False) -> Dict[str, Set[str]]:
        """分析机场重要性并返回分类结果
        
        Args:
            force_refresh: 是否强制重新分析（忽略缓存）
            
        Returns:
            Dict包含:
            - 'HUB_AIRPORTS': 主枢纽机场集合
            - 'MAJOR_AIRPORTS': 主要机场集合  
            - 'IMPORTANT_AIRPORTS': 重要机场集合
        """
        # 检查缓存
        if not force_refresh and self._load_cache():
            return self.analysis_cache
            
        print("正在分析机场重要性...")
        
        # 加载数据
        flight_data = self._load_flight_data()
        crew_data = self._load_crew_data()
        
        # 分析机场统计信息
        airport_stats = self._calculate_airport_statistics(flight_data, crew_data)
        
        # 基于统计信息进行分类
        classification = self._classify_airports(airport_stats)
        
        # 缓存结果
        self.analysis_cache = classification
        self._save_cache()
        
        return classification
    
    def _load_flight_data(self) -> pd.DataFrame:
        """加载航班数据"""
        flight_path = os.path.join(self.data_path, "flight.csv")
        if not os.path.exists(flight_path):
            raise FileNotFoundError(f"航班数据文件不存在: {flight_path}")
        return pd.read_csv(flight_path)
    
    def _load_crew_data(self) -> pd.DataFrame:
        """加载机组数据"""
        crew_path = os.path.join(self.data_path, "crew.csv")
        if not os.path.exists(crew_path):
            raise FileNotFoundError(f"机组数据文件不存在: {crew_path}")
        return pd.read_csv(crew_path)
    
    def _calculate_airport_statistics(self, flight_data: pd.DataFrame, 
                                    crew_data: pd.DataFrame) -> Dict[str, Dict]:
        """计算机场统计信息"""
        airport_stats = defaultdict(lambda: {
            'flight_count': 0,
            'crew_base_count': 0,
            'crew_stay_count': 0,
            'departure_count': 0,
            'arrival_count': 0,
            'total_activity': 0
        })
        
        # 统计航班信息
        for _, flight in flight_data.iterrows():
            depa = flight['depaAirport']
            arri = flight['arriAirport']
            
            airport_stats[depa]['departure_count'] += 1
            airport_stats[depa]['flight_count'] += 1
            airport_stats[arri]['arrival_count'] += 1
            airport_stats[arri]['flight_count'] += 1
        
        # 统计机组基地信息
        crew_base_counts = crew_data['base'].value_counts()
        crew_stay_counts = crew_data['stayStation'].value_counts()
        
        for airport, count in crew_base_counts.items():
            airport_stats[airport]['crew_base_count'] = count
            
        for airport, count in crew_stay_counts.items():
            airport_stats[airport]['crew_stay_count'] = count
        
        # 计算综合活跃度
        for airport in airport_stats:
            stats = airport_stats[airport]
            stats['total_activity'] = (
                stats['flight_count'] * 1.0 +
                stats['crew_base_count'] * 2.0 +  # 机组基地权重更高
                stats['crew_stay_count'] * 1.5
            )
        
        return dict(airport_stats)
    
    def _classify_airports(self, airport_stats: Dict[str, Dict]) -> Dict[str, Set[str]]:
        """基于统计信息对机场进行分类"""
        # 按综合活跃度排序
        sorted_airports = sorted(
            airport_stats.items(),
            key=lambda x: x[1]['total_activity'],
            reverse=True
        )
        
        if not sorted_airports:
            return {
                'HUB_AIRPORTS': set(),
                'MAJOR_AIRPORTS': set(),
                'IMPORTANT_AIRPORTS': set()
            }
        
        # 动态阈值计算
        activities = [stats['total_activity'] for _, stats in sorted_airports]
        max_activity = max(activities)
        mean_activity = np.mean(activities)
        std_activity = np.std(activities)
        
        # 分类阈值（基于统计分布）
        hub_threshold = max_activity * 0.7  # 最高活跃度的70%
        major_threshold = mean_activity + std_activity  # 均值+标准差
        important_threshold = mean_activity * 0.5  # 均值的50%
        
        hub_airports = set()
        major_airports = set()
        important_airports = set()
        
        for airport, stats in sorted_airports:
            activity = stats['total_activity']
            
            if activity >= hub_threshold:
                hub_airports.add(airport)
            elif activity >= major_threshold:
                major_airports.add(airport)
            elif activity >= important_threshold:
                important_airports.add(airport)
        
        # 确保至少有一个枢纽机场
        if not hub_airports and sorted_airports:
            hub_airports.add(sorted_airports[0][0])
        
        # 所有分类的机场都应该在重要机场集合中
        all_important = hub_airports | major_airports | important_airports
        
        # 打印分析结果
        print(f"\n=== 机场重要性分析结果 ===")
        print(f"主枢纽机场 ({len(hub_airports)}个): {sorted(hub_airports)}")
        print(f"主要机场 ({len(major_airports)}个): {sorted(major_airports)}")
        print(f"重要机场 ({len(important_airports)}个): {sorted(important_airports)}")
        print(f"\n前10个最活跃机场:")
        for i, (airport, stats) in enumerate(sorted_airports[:10]):
            print(f"{i+1:2d}. {airport}: 活跃度={stats['total_activity']:.1f} "
                  f"(航班={stats['flight_count']}, 基地={stats['crew_base_count']}, "
                  f"驻留={stats['crew_stay_count']})")
        
        return {
            'HUB_AIRPORTS': hub_airports,
            'MAJOR_AIRPORTS': major_airports,
            'IMPORTANT_AIRPORTS': all_important
        }
    
    def _load_cache(self) -> bool:
        """加载缓存的分析结果"""
        try:
            if os.path.exists(self.config_cache_file):
                with open(self.config_cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    # 转换为set类型
                    self.analysis_cache = {
                        key: set(value) for key, value in cache_data.items()
                    }
                    return True
        except Exception as e:
            print(f"加载缓存失败: {e}")
        return False
    
    def _save_cache(self):
        """保存分析结果到缓存"""
        try:
            # 转换set为list以便JSON序列化
            cache_data = {
                key: list(value) for key, value in self.analysis_cache.items()
            }
            with open(self.config_cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            print(f"分析结果已缓存到: {self.config_cache_file}")
        except Exception as e:
            print(f"保存缓存失败: {e}")
    
    def get_airport_importance_score(self, airport: str) -> float:
        """获取特定机场的重要性评分"""
        if not self.analysis_cache:
            self.analyze_airport_importance()
        
        if airport in self.analysis_cache.get('HUB_AIRPORTS', set()):
            return 1.0
        elif airport in self.analysis_cache.get('MAJOR_AIRPORTS', set()):
            return 0.8
        elif airport in self.analysis_cache.get('IMPORTANT_AIRPORTS', set()):
            return 0.6
        else:
            return 0.3
    
    def validate_airport_exists(self, airport: str) -> bool:
        """验证机场是否存在于数据中"""
        try:
            flight_data = self._load_flight_data()
            crew_data = self._load_crew_data()
            
            # 检查是否在航班数据中
            in_flights = (
                (flight_data['depaAirport'] == airport).any() or
                (flight_data['arriAirport'] == airport).any()
            )
            
            # 检查是否在机组数据中
            in_crew = (
                (crew_data['base'] == airport).any() or
                (crew_data['stayStation'] == airport).any()
            )
            
            return in_flights or in_crew
        except Exception:
            return False

# 全局分析器实例
_global_analyzer = None

def get_airport_analyzer(data_path: str = "data") -> DynamicAirportAnalyzer:
    """获取全局机场分析器实例"""
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = DynamicAirportAnalyzer(data_path)
    return _global_analyzer

def get_dynamic_airport_config(data_path: str = "data", force_refresh: bool = False) -> Dict[str, Set[str]]:
    """获取动态机场配置（推荐的API）"""
    analyzer = get_airport_analyzer(data_path)
    return analyzer.analyze_airport_importance(force_refresh)