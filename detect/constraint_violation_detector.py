#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
约束违规检测工具
用于检测和分析实际排班数据中的约束违规情况
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional, Tuple
import logging
from collections import defaultdict
from data_models import Flight, Crew, GroundDuty, BusInfo, DutyDay, FlightDutyPeriod
from unified_config import UnifiedConfig
from constraint_checker import UnifiedConstraintChecker

class ConstraintViolationDetector:
    """约束违规检测器"""
    
    def __init__(self):
        # 加载数据
        self.flights_df = pd.read_csv('data/flight.csv')
        self.crews_df = pd.read_csv('data/crew.csv')
        self.ground_duties_df = pd.read_csv('data/groundDuty.csv')
        self.layover_stations_df = pd.read_csv('data/layoverStation.csv')
        
        # 尝试加载排班结果
        roster_paths = [
            'submit/0710-2-（-4224）/rosterResult.csv',
        ]
        
        self.roster_result_df = pd.DataFrame()
        self.has_roster_data = False
        
        for path in roster_paths:
            try:
                self.roster_result_df = pd.read_csv(path)
                print(f"成功加载排班结果数据: {path} - {len(self.roster_result_df)} 条记录")
                self.has_roster_data = True
                break
            except FileNotFoundError:
                continue
        
        if not self.has_roster_data:
            print("警告: 未找到任何 rosterResult.csv 文件，将使用模拟数据进行分析")
        
        # 获取可过夜机场集合
        self.layover_stations_set = set(self.layover_stations_df['airport'].tolist())
        
        # 初始化约束检查器
        self.constraint_checker = UnifiedConstraintChecker(self.layover_stations_set)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # 违规统计
        self.violation_stats = {
            'connection_time_violations': 0,
            'flight_count_violations': 0,
            'task_count_violations': 0,
            'flight_time_violations': 0,
            'rest_time_violations': 0,
            'duty_time_violations': 0,
            'flight_cycle_violations': 0,
            'total_flight_duty_time_violations': 0
        }
    
    def detect_all_violations(self):
        """检测所有约束违规"""
        print("=" * 80)
        print("约束违规检测报告")
        print("=" * 80)
        
        if not self.has_roster_data:
            print("\n无排班数据，执行理论约束分析...")
            self._analyze_theoretical_constraints()
            return
        
        print(f"\n排班数据概况:")
        print(f"  - 排班记录数: {len(self.roster_result_df)}")
        print(f"  - 涉及机组数: {self.roster_result_df['crewId'].nunique()}")
        print(f"  - 涉及任务数: {self.roster_result_df['taskId'].nunique()}")
        
        # 按机组分组检测违规
        crew_groups = self.roster_result_df.groupby('crewId')
        
        total_crews = len(crew_groups)
        processed_crews = 0
        
        for crew_id, crew_tasks in crew_groups:
            processed_crews += 1
            if processed_crews <= 5:  # 只处理前5个机组作为示例
                print(f"\n检测机组 {crew_id} ({processed_crews}/{total_crews})...")
                self._detect_crew_violations(crew_id, crew_tasks)
        
        # 输出总体统计
        self._print_violation_summary()
    
    def _detect_crew_violations(self, crew_id: str, crew_tasks: pd.DataFrame):
        """检测单个机组的违规情况"""
        # 获取机组信息
        crew_info = self.crews_df[self.crews_df['crewId'] == crew_id]
        if crew_info.empty:
            print(f"  警告: 未找到机组 {crew_id} 的基本信息")
            return
        
        crew_base = crew_info.iloc[0]['base']
        
        # 构建任务列表
        tasks = []
        for _, task_row in crew_tasks.iterrows():
            task_id = task_row['taskId']
            
            # 根据任务ID类型获取详细信息
            if task_id.startswith('Flt_'):
                # 飞行任务
                flight_info = self.flights_df[self.flights_df['id'] == task_id]
                if not flight_info.empty:
                    flight = flight_info.iloc[0]
                    task_data = {
                        'taskId': task_id,
                        'type': 'flight',
                        'startTime': pd.to_datetime(flight['std']),
                        'endTime': pd.to_datetime(flight['sta']),
                        'depAirport': flight['depaAirport'],
                        'arrAirport': flight['arriAirport'],
                        'flyTime': flight['flyTime']
                    }
                    tasks.append(task_data)
            elif task_id.startswith('Grd_'):
                # 地面任务
                ground_info = self.ground_duties_df[self.ground_duties_df['id'] == task_id]
                if not ground_info.empty:
                    ground = ground_info.iloc[0]
                    task_data = {
                        'taskId': task_id,
                        'type': 'ground_duty',
                        'startTime': pd.to_datetime(ground['startTime']),
                        'endTime': pd.to_datetime(ground['endTime']),
                        'airport': ground['airport']
                    }
                    tasks.append(task_data)
        
        if not tasks:
            print(f"  机组 {crew_id}: 无有效任务数据")
            return
        
        # 按时间排序
        tasks.sort(key=lambda x: x['startTime'])
        
        print(f"  机组 {crew_id}: {len(tasks)} 个任务")
        
        # 检测各种违规
        self._check_connection_time_violations(crew_id, tasks)
        self._check_duty_day_violations(crew_id, tasks)
        self._check_flight_cycle_violations(crew_id, tasks, crew_base)
        self._check_total_flight_time_violations(crew_id, tasks)
    
    def _check_connection_time_violations(self, crew_id: str, tasks: List[Dict]):
        """检查连接时间违规"""
        violations = 0
        
        for i in range(1, len(tasks)):
            prev_task = tasks[i-1]
            curr_task = tasks[i]
            
            connection_time = curr_task['startTime'] - prev_task['endTime']
            
            # 获取最小连接时间要求
            min_connection = self._get_min_connection_time(prev_task, curr_task)
            
            if connection_time < min_connection:
                violations += 1
                print(f"    连接时间违规: {prev_task['taskId']} -> {curr_task['taskId']}, "
                      f"实际: {connection_time}, 要求: {min_connection}")
        
        if violations > 0:
            self.violation_stats['connection_time_violations'] += violations
            print(f"  机组 {crew_id}: {violations} 个连接时间违规")
    
    def _get_min_connection_time(self, prev_task: Dict, curr_task: Dict) -> timedelta:
        """获取最小连接时间"""
        if curr_task['type'] == 'flight':
            # 检查是否为同一架飞机（简化处理，假设不同）
            return timedelta(hours=UnifiedConfig.MIN_CONNECTION_TIME_FLIGHT_DIFFERENT_AIRCRAFT_HOURS)
        else:
            return timedelta(hours=1)  # 默认1小时
    
    def _check_duty_day_violations(self, crew_id: str, tasks: List[Dict]):
        """检查值勤日违规"""
        # 组织为值勤日
        duty_days = self._organize_tasks_into_duty_days(tasks)
        
        for i, duty_day in enumerate(duty_days):
            violations = []
            
            # 检查值勤时间
            duty_duration = (duty_day['end_time'] - duty_day['start_time']).total_seconds() / 3600
            if duty_duration > UnifiedConfig.MAX_DUTY_DAY_HOURS:
                violations.append(f"值勤时间超限: {duty_duration:.1f}h > {UnifiedConfig.MAX_DUTY_DAY_HOURS}h")
                self.violation_stats['duty_time_violations'] += 1
            
            # 检查任务数量
            if len(duty_day['tasks']) > UnifiedConfig.MAX_TASKS_IN_DUTY:
                violations.append(f"任务数超限: {len(duty_day['tasks'])} > {UnifiedConfig.MAX_TASKS_IN_DUTY}")
                self.violation_stats['task_count_violations'] += 1
            
            # 检查飞行任务数量
            flight_count = sum(1 for task in duty_day['tasks'] if task['type'] == 'flight')
            if flight_count > UnifiedConfig.MAX_FLIGHTS_IN_DUTY:
                violations.append(f"飞行数超限: {flight_count} > {UnifiedConfig.MAX_FLIGHTS_IN_DUTY}")
                self.violation_stats['flight_count_violations'] += 1
            
            # 检查飞行时间
            total_flight_time = sum(task.get('flyTime', 0) for task in duty_day['tasks'] if task['type'] == 'flight' and not task.get('is_positioning', False)) / 60.0
            if total_flight_time > UnifiedConfig.MAX_FLIGHT_TIME_IN_DUTY_HOURS:
                violations.append(f"飞行时间超限: {total_flight_time:.1f}h > {UnifiedConfig.MAX_FLIGHT_TIME_IN_DUTY_HOURS}h")
                self.violation_stats['flight_time_violations'] += 1
            
            if violations:
                print(f"  机组 {crew_id} 值勤日{i+1}: {', '.join(violations)}")
    
    def _organize_tasks_into_duty_days(self, tasks: List[Dict]) -> List[Dict]:
        """将任务组织为值勤日"""
        if not tasks:
            return []
        
        duty_days = []
        current_duty = {
            'start_time': tasks[0]['startTime'],
            'end_time': tasks[0]['endTime'],
            'tasks': [tasks[0]]
        }
        
        for i in range(1, len(tasks)):
            task = tasks[i]
            prev_task = tasks[i-1]
            
            rest_time = task['startTime'] - prev_task['endTime']
            
            # 判断是否开始新值勤日
            if (rest_time >= timedelta(hours=UnifiedConfig.MIN_REST_HOURS) or
                (task['startTime'] - current_duty['start_time']) > timedelta(hours=24)):
                
                # 结束当前值勤日
                duty_days.append(current_duty)
                
                # 开始新值勤日
                current_duty = {
                    'start_time': task['startTime'],
                    'end_time': task['endTime'],
                    'tasks': [task]
                }
            else:
                # 继续当前值勤日
                current_duty['tasks'].append(task)
                current_duty['end_time'] = task['endTime']
        
        # 添加最后一个值勤日
        duty_days.append(current_duty)
        
        return duty_days
    
    def _check_flight_cycle_violations(self, crew_id: str, tasks: List[Dict], crew_base: str):
        """检查飞行周期违规"""
        # 简化的飞行周期检查
        duty_days = self._organize_tasks_into_duty_days(tasks)
        
        current_cycle_days = []
        violations = 0
        
        for duty_day in duty_days:
            # 检查是否包含飞行任务
            has_flight = any(task['type'] == 'flight' for task in duty_day['tasks'])
            
            if has_flight:
                current_cycle_days.append(duty_day)
                
                # 检查周期长度
                if len(current_cycle_days) > 1:
                    cycle_start = current_cycle_days[0]['start_time'].date()
                    cycle_end = duty_day['end_time'].date()
                    cycle_days = (cycle_end - cycle_start).days + 1
                    
                    if cycle_days > 4:
                        violations += 1
                        print(f"  机组 {crew_id}: 飞行周期超限 {cycle_days} 天 > 4 天")
                
                # 检查是否返回基地
                last_task = duty_day['tasks'][-1]
                if (last_task['type'] == 'flight' and 
                    last_task.get('arrAirport') == crew_base):
                    current_cycle_days = []  # 重置周期
        
        if violations > 0:
            self.violation_stats['flight_cycle_violations'] += violations
    
    def _check_total_flight_time_violations(self, crew_id: str, tasks: List[Dict]):
        """检查总飞行值勤时间违规"""
        # 修正：计算飞行值勤时间而不是飞行时间
        # 飞行值勤时间 = 飞行值勤日的总时长（从第一个任务开始到最后一个飞行任务结束）
        
        duty_days = self._organize_tasks_into_duty_days(tasks)
        total_flight_duty_time = 0.0
        
        for duty_day in duty_days:
            # 检查是否包含飞行任务
            flight_tasks = [task for task in duty_day['tasks'] if task['type'] == 'flight']
            if flight_tasks:
                # 飞行值勤日：从第一个任务开始到最后一个飞行任务结束
                first_task_start = duty_day['start_time']
                last_flight_end = max(task['endTime'] for task in flight_tasks)
                flight_duty_duration = (last_flight_end - first_task_start).total_seconds() / 3600.0
                total_flight_duty_time += flight_duty_duration
        
        if total_flight_duty_time > self.constraint_checker.MAX_TOTAL_FLIGHT_HOURS:
            self.violation_stats['total_flight_duty_time_violations'] += 1
            print(f"  机组 {crew_id}: 总飞行值勤时间超限 {total_flight_duty_time:.1f}h > {self.constraint_checker.MAX_TOTAL_FLIGHT_HOURS}h")
    
    def _analyze_theoretical_constraints(self):
        """分析理论约束（无排班数据时）"""
        print("\n理论约束分析:")
        print("-" * 50)
        
        # 分析航班数据的理论约束挑战
        print(f"航班总数: {len(self.flights_df)}")
        print(f"机组总数: {len(self.crews_df)}")
        print(f"地面任务总数: {len(self.ground_duties_df)}")
        
        # 分析航班时间分布
        self.flights_df['std'] = pd.to_datetime(self.flights_df['std'])
        self.flights_df['sta'] = pd.to_datetime(self.flights_df['sta'])
        
        # 计算平均飞行时间
        avg_flight_time = self.flights_df['flyTime'].mean() / 60.0
        max_flight_time = self.flights_df['flyTime'].max() / 60.0
        
        print(f"\n航班特征:")
        print(f"  - 平均飞行时间: {avg_flight_time:.1f} 小时")
        print(f"  - 最长飞行时间: {max_flight_time:.1f} 小时")
        
        # 分析潜在的约束挑战
        long_flights = self.flights_df[self.flights_df['flyTime'] > UnifiedConfig.MAX_FLIGHT_TIME_IN_DUTY_HOURS * 60]
        print(f"  - 超过值勤飞行时间限制的航班: {len(long_flights)}")
        
        # 分析机场分布
        unique_airports = set(self.flights_df['depaAirport'].unique()) | set(self.flights_df['arriAirport'].unique())
        layover_airports = self.layover_stations_set
        non_layover_airports = unique_airports - layover_airports
        
        print(f"\n机场分析:")
        print(f"  - 总机场数: {len(unique_airports)}")
        print(f"  - 可过夜机场数: {len(layover_airports)}")
        print(f"  - 不可过夜机场数: {len(non_layover_airports)}")
        
        if non_layover_airports:
            print(f"  - 不可过夜机场示例: {list(non_layover_airports)[:5]}")
    
    def _print_violation_summary(self):
        """打印违规统计摘要"""
        print("\n=" * 80)
        print("违规统计摘要")
        print("=" * 80)
        
        total_violations = sum(self.violation_stats.values())
        
        print(f"总违规数: {total_violations}")
        print("\n详细统计:")
        for violation_type, count in self.violation_stats.items():
            if count > 0:
                print(f"  - {violation_type.replace('_', ' ').title()}: {count}")
        
        if total_violations == 0:
            print("\n✓ 未检测到约束违规（在检查的样本中）")
        else:
            print(f"\n⚠ 检测到 {total_violations} 个约束违规")

def main():
    """主函数"""
    detector = ConstraintViolationDetector()
    detector.detect_all_violations()

if __name__ == '__main__':
    main()