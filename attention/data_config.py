# data_config.py
# 数据配置模块，提供通用的数据处理和配置功能

import pandas as pd
import os
from datetime import datetime, timedelta

class DataConfig:
    """数据配置类，提供通用的数据处理和时间范围获取功能"""
    
    def __init__(self, data_path=None):
        """初始化数据配置
        
        Args:
            data_path: 数据文件夹路径，如果为None则使用默认路径
        """
        if data_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_path = os.path.join(current_dir, "..", "data")
        else:
            self.data_path = data_path
            
        self._flight_df = None
        self._ground_duty_df = None
        self._crew_df = None
        
    def get_flight_data(self):
        """获取航班数据"""
        if self._flight_df is None:
            flight_path = os.path.join(self.data_path, "flight.csv")
            self._flight_df = pd.read_csv(flight_path)
            self._flight_df['std'] = pd.to_datetime(self._flight_df['std'])
            self._flight_df['sta'] = pd.to_datetime(self._flight_df['sta'])
        return self._flight_df
    
    def get_ground_duty_data(self):
        """获取占位任务数据"""
        if self._ground_duty_df is None:
            ground_duty_path = os.path.join(self.data_path, "groundDuty.csv")
            self._ground_duty_df = pd.read_csv(ground_duty_path)
            self._ground_duty_df['startTime'] = pd.to_datetime(self._ground_duty_df['startTime'])
            self._ground_duty_df['endTime'] = pd.to_datetime(self._ground_duty_df['endTime'])
        return self._ground_duty_df
    
    def get_crew_data(self):
        """获取机组数据"""
        if self._crew_df is None:
            crew_path = os.path.join(self.data_path, "crew.csv")
            self._crew_df = pd.read_csv(crew_path)
        return self._crew_df
    
    def get_bus_info_data(self):
        """获取班车信息数据"""
        try:
            bus_info_path = os.path.join(self.data_path, 'busInfo.csv')
            if os.path.exists(bus_info_path):
                return pd.read_csv(bus_info_path)
            return pd.DataFrame()
        except Exception as e:
            print(f"警告: 无法读取班车信息数据: {e}")
            return pd.DataFrame()
    
    def get_planning_time_range(self, buffer_hours=0):
        """从数据中获取规划时间范围
        
        Args:
            buffer_hours: 在开始和结束时间前后添加的缓冲时间（小时）
            
        Returns:
            tuple: (start_datetime, end_datetime)
        """
        try:
            # 获取所有数据
            flights_df = self.get_flight_data()
            ground_duties_df = self.get_ground_duty_data()
            bus_info_df = self.get_bus_info_data()
            
            all_times = []
            
            # 从航班数据获取时间
            if not flights_df.empty:
                if 'std' in flights_df.columns:
                    all_times.extend(pd.to_datetime(flights_df['std'], format='%Y/%m/%d %H:%M', errors='coerce').dropna())
                if 'sta' in flights_df.columns:
                    all_times.extend(pd.to_datetime(flights_df['sta'], format='%Y/%m/%d %H:%M', errors='coerce').dropna())
            
            # 从占位任务数据获取时间
            if not ground_duties_df.empty:
                if 'startTime' in ground_duties_df.columns:
                    all_times.extend(pd.to_datetime(ground_duties_df['startTime'], format='%Y/%m/%d %H:%M', errors='coerce').dropna())
                if 'endTime' in ground_duties_df.columns:
                    all_times.extend(pd.to_datetime(ground_duties_df['endTime'], format='%Y/%m/%d %H:%M', errors='coerce').dropna())
            
            # 从班车信息数据获取时间
            if not bus_info_df.empty:
                if 'td' in bus_info_df.columns:
                    all_times.extend(pd.to_datetime(bus_info_df['td'], format='%Y/%m/%d %H:%M', errors='coerce').dropna())
                if 'ta' in bus_info_df.columns:
                    all_times.extend(pd.to_datetime(bus_info_df['ta'], format='%Y/%m/%d %H:%M', errors='coerce').dropna())
            
            if not all_times:
                # 如果没有找到时间数据，返回默认值
                return datetime(2025, 5, 1), datetime(2025, 5, 7, 23, 59, 59)
            
            # 计算时间范围
            min_time = min(all_times)
            max_time = max(all_times)
            
            # 添加缓冲时间
            start_time = min_time - timedelta(hours=buffer_hours)
            end_time = max_time + timedelta(hours=buffer_hours)
            
            # 确保开始时间是当天的00:00:00
            start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
            # 确保结束时间是当天的23:59:59
            end_time = end_time.replace(hour=23, minute=59, second=59, microsecond=0)
            
            return start_time, end_time
            
        except Exception as e:
            print(f"警告: 无法从数据获取时间范围: {e}")
            # 返回默认值
            return datetime(2025, 5, 1), datetime(2025, 5, 7, 23, 59, 59)
    
    def get_data_statistics(self):
        """获取数据统计信息"""
        try:
            flights_df = self.get_flight_data()
            ground_duties_df = self.get_ground_duty_data()
            crew_df = self.get_crew_data()
            bus_info_df = self.get_bus_info_data()
            
            start_time, end_time = self.get_planning_time_range()
            planning_days = (end_time - start_time).days + 1
            
            stats = {
                '规划天数': planning_days,
                '航班数量': len(flights_df) if not flights_df.empty else 0,
                '占位任务数量': len(ground_duties_df) if not ground_duties_df.empty else 0,
                '班车信息数量': len(bus_info_df) if not bus_info_df.empty else 0,
                '机组数量': len(crew_df) if not crew_df.empty else 0,
                '规划开始时间': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                '规划结束时间': end_time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 添加更详细的统计信息
            if not flights_df.empty:
                if 'fleet' in flights_df.columns:
                    stats['机队类型数量'] = flights_df['fleet'].nunique()
                if 'depaAirport' in flights_df.columns and 'arriAirport' in flights_df.columns:
                    airports = set(flights_df['depaAirport'].unique()) | set(flights_df['arriAirport'].unique())
                    stats['机场数量'] = len(airports)
            
            if not ground_duties_df.empty:
                if 'airport' in ground_duties_df.columns:
                    stats['占位任务机场数量'] = ground_duties_df['airport'].nunique()
                if 'crewId' in ground_duties_df.columns:
                    stats['涉及占位任务的机组数量'] = ground_duties_df['crewId'].nunique()
            
            if not bus_info_df.empty:
                if 'depaAirport' in bus_info_df.columns and 'arriAirport' in bus_info_df.columns:
                    bus_airports = set(bus_info_df['depaAirport'].unique()) | set(bus_info_df['arriAirport'].unique())
                    stats['班车服务机场数量'] = len(bus_airports)
            
            return stats
            
        except Exception as e:
            print(f"警告: 无法获取数据统计信息: {e}")
            return {}
    
    def print_data_summary(self):
        """打印数据摘要"""
        print("=== 数据集摘要 ===")
        stats = self.get_data_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")

# 创建全局数据配置实例
_global_data_config = DataConfig()

def get_planning_dates_from_data():
    """从数据文件中动态获取规划时间范围（向后兼容函数）"""
    return _global_data_config.get_planning_time_range()

def get_data_config():
    """获取全局数据配置实例"""
    return _global_data_config

def set_data_path(data_path):
    """设置数据路径"""
    global _global_data_config
    _global_data_config = DataConfig(data_path)