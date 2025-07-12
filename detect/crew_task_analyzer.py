#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机组任务序列分析器
用于分析提交结果中特定机组的任务序列，识别占位任务需求，并诊断问题
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import os

class CrewTaskAnalyzer:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.flights_df = None
        self.ground_duties_df = None
        self.roster_result_df = None
        self.crew_df = None
        self.load_data()
    
    def load_data(self):
        """加载所有必要的数据文件"""
        try:
            self.flights_df = pd.read_csv(os.path.join(self.data_dir, "flight.csv"))
            self.ground_duties_df = pd.read_csv(os.path.join(self.data_dir, "groundDuty.csv"))
            self.crew_df = pd.read_csv(os.path.join(self.data_dir, "crew.csv"))
            
            # 尝试加载排班结果
            result_files = [
                "rosterResult.csv", 
                "submit/0704-2-（-4221）/rosterResult.csv",
                "submit/0710-2-（-4224）/rosterResult.csv",
                "submit/0701-1-（-5221）/rosterResult.csv",
                "attention_dynamic_ep2/data/rosterResult.csv"
            ]
            for file_path in result_files:
                if file_path.startswith("submit/") or file_path.startswith("attention_dynamic_ep2/"):
                    full_path = file_path
                else:
                    full_path = os.path.join(self.data_dir, file_path)
                
                if os.path.exists(full_path):
                    self.roster_result_df = pd.read_csv(full_path)
                    print(f"已加载排班结果: {full_path}")
                    break
            
            if self.roster_result_df is None:
                print("警告: 未找到排班结果文件")
                
            print("数据加载完成")
            
        except Exception as e:
            print(f"数据加载失败: {e}")
    
    def parse_datetime(self, date_str: str) -> datetime:
        """解析日期时间字符串"""
        try:
            # 处理跨日标记 (+1)
            if '+1' in date_str:
                date_str = date_str.replace('+1', '')
                dt = datetime.strptime(date_str, "%Y/%m/%d %H:%M")
                dt += timedelta(days=1)
                return dt
            else:
                return datetime.strptime(date_str, "%Y/%m/%d %H:%M")
        except:
            return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    
    def get_crew_tasks(self, crew_id: str) -> List[Dict]:
        """获取指定机组的所有任务"""
        if self.roster_result_df is None:
            return []
        
        crew_tasks = self.roster_result_df[self.roster_result_df['crewId'] == crew_id]
        tasks = []
        
        for _, task in crew_tasks.iterrows():
            task_id = task['taskId']
            is_ddh = task.get('isDDH', 0)
            
            if task_id.startswith('Flt_'):
                # 航班任务
                flight_info = self.flights_df[self.flights_df['id'] == task_id]
                if not flight_info.empty:
                    flight = flight_info.iloc[0]
                    tasks.append({
                        'type': 'flight',
                        'task_id': task_id,
                        'departure_airport': flight['depaAirport'],
                        'arrival_airport': flight['arriAirport'],
                        'departure_time': self.parse_datetime(flight['std']),
                        'arrival_time': self.parse_datetime(flight['sta']),
                        'is_ddh': is_ddh,
                        'aircraft': flight.get('aircraftNo', ''),
                        'flight_time': flight.get('flyTime', 0)
                    })
            
            elif task_id.startswith('Grd_'):
                # 地面任务
                ground_info = self.ground_duties_df[self.ground_duties_df['id'] == task_id]
                if not ground_info.empty:
                    ground = ground_info.iloc[0]
                    tasks.append({
                        'type': 'ground',
                        'task_id': task_id,
                        'airport': ground['airport'],
                        'start_time': self.parse_datetime(ground['startTime']),
                        'end_time': self.parse_datetime(ground['endTime']),
                        'is_duty': ground.get('isDuty', 0),
                        'is_ddh': is_ddh
                    })
        
        # 按时间排序
        tasks.sort(key=lambda x: x.get('departure_time') or x.get('start_time'))
        return tasks
    
    def analyze_location_continuity(self, tasks: List[Dict]) -> List[Dict]:
        """分析地点连续性问题"""
        issues = []
        
        for i in range(len(tasks) - 1):
            current_task = tasks[i]
            next_task = tasks[i + 1]
            
            # 获取当前任务的结束地点
            if current_task['type'] == 'flight':
                current_end_location = current_task['arrival_airport']
                current_end_time = current_task['arrival_time']
            else:
                current_end_location = current_task['airport']
                current_end_time = current_task['end_time']
            
            # 获取下一个任务的开始地点
            if next_task['type'] == 'flight':
                next_start_location = next_task['departure_airport']
                next_start_time = next_task['departure_time']
            else:
                next_start_location = next_task['airport']
                next_start_time = next_task['start_time']
            
            # 检查地点连续性（纯粹的地点不连续问题）
            if current_end_location != next_start_location:
                issues.append({
                    'type': 'location_discontinuity',
                    'task1': current_task['task_id'],
                    'task2': next_task['task_id'],
                    'end_location': current_end_location,
                    'start_location': next_start_location,
                    'description': f"地点不连续: {current_end_location} -> {next_start_location}"
                })
        
        return issues
    
    def check_ground_duty_compliance(self, crew_id: str) -> List[Dict]:
        """检查地面任务是否在正确的机场执行"""
        crew_ground_duties = self.ground_duties_df[self.ground_duties_df['crewId'] == crew_id]
        compliance_issues = []
        
        # 获取机组的所有任务（航班+地面）
        all_tasks = self.get_crew_tasks(crew_id)
        
        # 检查每个地面任务
        for _, duty in crew_ground_duties.iterrows():
            duty_start = pd.to_datetime(duty['startTime'])
            duty_end = pd.to_datetime(duty['endTime'])
            duty_location = duty['airport']
            
            # 找到地面任务开始时机组应该在的位置
            crew_location_at_start = self.get_crew_location_at_time(all_tasks, duty_start)
            crew_location_at_end = self.get_crew_location_at_time(all_tasks, duty_end)
            
            if crew_location_at_start != duty_location:
                compliance_issues.append({
                    'duty_id': duty['id'],
                    'duty_location': duty_location,
                    'duty_time': f"{duty_start.strftime('%m/%d %H:%M')} - {duty_end.strftime('%m/%d %H:%M')}",
                    'crew_location': crew_location_at_start,
                    'description': f"地面任务{duty['id']}要求在{duty_location}执行，但机组在{duty_start.strftime('%m/%d %H:%M')}时位于{crew_location_at_start}"
                })
        
        return compliance_issues
    
    def get_crew_location_at_time(self, tasks: List[Dict], target_time: pd.Timestamp) -> str:
        """获取机组在指定时间的位置"""
        for task in tasks:
            if task['type'] == 'flight':
                if task['departure_time'] <= target_time <= task['arrival_time']:
                    return "飞行中"
                elif target_time < task['departure_time']:
                    # 在这个航班之前，检查上一个任务的结束位置
                    continue
            else:  # ground task
                if task['start_time'] <= target_time <= task['end_time']:
                    return task['airport']
        
        # 如果没有找到确切的任务，找最近的任务位置
        closest_task = None
        min_time_diff = float('inf')
        
        for task in tasks:
            if task['type'] == 'flight':
                # 检查航班结束时间
                time_diff = abs((task['arrival_time'] - target_time).total_seconds())
                if time_diff < min_time_diff and task['arrival_time'] <= target_time:
                    min_time_diff = time_diff
                    closest_task = task
            else:
                # 检查地面任务结束时间
                time_diff = abs((task['end_time'] - target_time).total_seconds())
                if time_diff < min_time_diff and task['end_time'] <= target_time:
                    min_time_diff = time_diff
                    closest_task = task
        
        if closest_task:
            if closest_task['type'] == 'flight':
                return closest_task['arrival_airport']
            else:
                return closest_task['airport']
        
        return "未知位置"
    
    def analyze_crew(self, crew_id: str) -> Dict:
        """分析指定机组的完整情况"""
        print(f"\n=== 分析机组 {crew_id} ===")
        
        # 获取机组基本信息
        crew_info = self.crew_df[self.crew_df['crewId'] == crew_id]
        if not crew_info.empty:
            base = crew_info.iloc[0]['base']
            stay_station = crew_info.iloc[0]['stayStation']
            print(f"机组基地: {base}, 驻留站: {stay_station}")
        
        # 获取任务序列
        tasks = self.get_crew_tasks(crew_id)
        if not tasks:
            print("未找到该机组的任务")
            return {}
        
        print(f"\n任务序列 (共{len(tasks)}个任务):")
        print("-" * 100)
        
        for i, task in enumerate(tasks, 1):
            if task['type'] == 'flight':
                print(f"{i:2d}. {task['task_id']:12} | 航班 | {task['departure_airport']} -> {task['arrival_airport']} | "
                      f"{task['departure_time'].strftime('%m/%d %H:%M')} - {task['arrival_time'].strftime('%m/%d %H:%M')} | "
                      f"DDH: {task['is_ddh']}")
            else:
                print(f"{i:2d}. {task['task_id']:12} | 地面 | {task['airport']:13} | "
                      f"{task['start_time'].strftime('%m/%d %H:%M')} - {task['end_time'].strftime('%m/%d %H:%M')} | "
                      f"值勤: {task['is_duty']}")
        
        # 分析地点连续性问题
        issues = self.analyze_location_continuity(tasks)
        if issues:
            print(f"\n发现的问题 (共{len(issues)}个):")
            print("-" * 100)
            for i, issue in enumerate(issues, 1):
                print(f"{i}. {issue['description']}")
        
        # 检查地面任务合规性
        compliance_issues = self.check_ground_duty_compliance(crew_id)
        
        # 显示地面任务合规性检查结果
        crew_ground_duties = self.ground_duties_df[self.ground_duties_df['crewId'] == crew_id]
        if not crew_ground_duties.empty:
            print(f"\n地面任务合规性检查 (共{len(crew_ground_duties)}个地面任务):")
            print("-" * 100)
            for _, duty in crew_ground_duties.iterrows():
                duty_start = pd.to_datetime(duty['startTime'])
                duty_end = pd.to_datetime(duty['endTime'])
                crew_location = self.get_crew_location_at_time(tasks, duty_start)
                status = "✓ 合规" if crew_location == duty['airport'] else "✗ 违规"
                print(f"{duty['id']} | {duty['airport']} | {duty_start.strftime('%m/%d %H:%M')} - {duty_end.strftime('%m/%d %H:%M')} | 机组位置: {crew_location} | {status}")
        
        if compliance_issues:
            print(f"\n发现地面任务违规 (共{len(compliance_issues)}个):")
            print("-" * 100)
            for i, issue in enumerate(compliance_issues, 1):
                print(f"{i}. {issue['description']}")
        
        return {
            'crew_id': crew_id,
            'tasks': tasks,
            'location_issues': issues,
            'compliance_issues': compliance_issues
        }
    
    def batch_analyze_crews(self, crew_ids: List[str]) -> Dict:
        """批量分析多个机组"""
        results = {}
        for crew_id in crew_ids:
            results[crew_id] = self.analyze_crew(crew_id)
        return results

def main():
    """主函数 - 示例用法"""
    analyzer = CrewTaskAnalyzer()
    
    # 直接分析Crew_10002
    crew_id = "Crew_10002"
    print(f"正在分析机组: {crew_id}")
    
    result = analyzer.analyze_crew(crew_id)
    
    # 将结果保存到文件
    with open(f"crew_analysis_{crew_id}.txt", "w", encoding="utf-8") as f:
        f.write(f"=== 机组 {crew_id} 分析报告 ===\n\n")
        
        if result and 'tasks' in result:
            f.write(f"任务序列 (共{len(result['tasks'])}个任务):\n")
            f.write("-" * 100 + "\n")
            
            for i, task in enumerate(result['tasks'], 1):
                if task['type'] == 'flight':
                    f.write(f"{i:2d}. {task['task_id']:12} | 航班 | {task['departure_airport']} -> {task['arrival_airport']} | "
                          f"{task['departure_time'].strftime('%m/%d %H:%M')} - {task['arrival_time'].strftime('%m/%d %H:%M')} | "
                          f"DDH: {task['is_ddh']}\n")
                else:
                    f.write(f"{i:2d}. {task['task_id']:12} | 地面 | {task['airport']:13} | "
                          f"{task['start_time'].strftime('%m/%d %H:%M')} - {task['end_time'].strftime('%m/%d %H:%M')} | "
                          f"值勤: {task['is_duty']}\n")
            
            # 地面任务合规性检查
            crew_ground_duties = analyzer.ground_duties_df[analyzer.ground_duties_df['crewId'] == crew_id]
            if not crew_ground_duties.empty:
                f.write(f"\n地面任务合规性检查 (共{len(crew_ground_duties)}个地面任务):\n")
                f.write("-" * 100 + "\n")
                for _, duty in crew_ground_duties.iterrows():
                    duty_start = pd.to_datetime(duty['startTime'])
                    duty_end = pd.to_datetime(duty['endTime'])
                    crew_location = analyzer.get_crew_location_at_time(result['tasks'], duty_start)
                    status = "✓ 合规" if crew_location == duty['airport'] else "✗ 违规"
                    f.write(f"{duty['id']} | {duty['airport']} | {duty_start.strftime('%m/%d %H:%M')} - {duty_end.strftime('%m/%d %H:%M')} | 机组位置: {crew_location} | {status}\n")
            
            if 'location_issues' in result and result['location_issues']:
                f.write(f"\n地点衔接问题 (共{len(result['location_issues'])}个):\n")
                f.write("-" * 100 + "\n")
                for i, issue in enumerate(result['location_issues'], 1):
                    f.write(f"{i}. {issue['description']}\n")
            
            if 'compliance_issues' in result and result['compliance_issues']:
                f.write(f"\n地面任务违规 (共{len(result['compliance_issues'])}个):\n")
                f.write("-" * 100 + "\n")
                for i, issue in enumerate(result['compliance_issues'], 1):
                    f.write(f"{i}. {issue['description']}\n")
    
    print(f"\n分析完成! 详细报告已保存到: crew_analysis_{crew_id}.txt")
    return result

if __name__ == "__main__":
    main()