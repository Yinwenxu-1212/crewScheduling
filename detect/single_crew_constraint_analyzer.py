#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单个机组约束违规详细分析工具
分析指定机组的所有约束违规情况及计算逻辑
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
from collections import defaultdict

# 添加项目根目录到路径
sys.path.append('..')
sys.path.append('.')

try:
    from data_models import Flight, CrewMember, GroundTask
except ImportError:
    print("警告: 无法导入data_models，将使用基础分析")
    Flight = None
    CrewMember = None
    GroundTask = None

def load_data():
    """加载所有相关数据"""
    data = {}
    
    # 加载排班结果
    roster_paths = [
        "submit/0710-2-（-4224）/rosterResult.csv",
        "data/rosterResult.csv",
        "attention_dynamic_ep2/data/rosterResult.csv",
        "../attention_dynamic_ep2/data/rosterResult.csv"
    ]
    
    for path in roster_paths:
        try:
            data['roster'] = pd.read_csv(path)
            print(f"成功加载排班结果: {path}")
            break
        except:
            continue
    
    # 加载航班数据
    try:
        data['flights'] = pd.read_csv("data/flight.csv")
        print("成功加载航班数据")
    except Exception as e:
        print(f"加载航班数据失败: {e}")
    
    # 加载机组数据
    try:
        data['crews'] = pd.read_csv("data/crew.csv")
        print("成功加载机组数据")
    except Exception as e:
        print(f"加载机组数据失败: {e}")
    
    # 加载地面任务数据
    try:
        data['ground_duties'] = pd.read_csv("data/groundDuty.csv")
        print("成功加载地面任务数据")
    except Exception as e:
        print(f"加载地面任务数据失败: {e}")
    
    return data

def parse_datetime(date_str):
    """解析日期时间字符串"""
    try:
        return pd.to_datetime(date_str)
    except:
        return None

def analyze_single_crew(crew_id, data):
    """分析单个机组的约束违规情况"""
    print(f"\n{'='*80}")
    print(f"机组 {crew_id} 详细约束分析")
    print(f"{'='*80}")
    
    if 'roster' not in data or data['roster'] is None:
        print("错误: 无法加载排班数据")
        return
    
    # 获取该机组的所有任务
    crew_tasks = data['roster'][data['roster']['crewId'] == crew_id].copy()
    
    if crew_tasks.empty:
        print(f"未找到机组 {crew_id} 的任务记录")
        return
    
    print(f"机组 {crew_id} 总任务数: {len(crew_tasks)}")
    
    # 从航班数据中获取日期信息
    flights_data = data.get('flights')
    if flights_data is not None:
        # 为每个任务添加日期信息
        crew_tasks['dutyDate'] = None
        for idx, task in crew_tasks.iterrows():
            task_id = task['taskId']
            if task_id.startswith('Flt_'):
                flight_info = flights_data[flights_data['id'] == task_id]
                if not flight_info.empty:
                    std = flight_info.iloc[0]['std']
                    duty_date = pd.to_datetime(std).date()
                    crew_tasks.at[idx, 'dutyDate'] = duty_date
            elif task_id.startswith('Grd_'):
                # 地面任务，使用默认日期或从任务ID中解析
                crew_tasks.at[idx, 'dutyDate'] = pd.to_datetime('2025-05-01').date()
        
        # 移除没有日期信息的任务
        crew_tasks = crew_tasks.dropna(subset=['dutyDate'])
        crew_tasks['dutyDate'] = pd.to_datetime(crew_tasks['dutyDate'])
        
        # 按日期和任务ID排序（没有taskOrder字段）
        crew_tasks = crew_tasks.sort_values(['dutyDate', 'taskId'])
    else:
        print("无法获取航班数据，无法进行日期分析")
        return
    
    # 分析每个值勤日
    duty_days = crew_tasks.groupby('dutyDate')
    
    total_violations = {
        'connection_time': 0,
        'flight_task_limit': 0,
        'duty_task_limit': 0,
        'flight_time_limit': 0,
        'rest_time_limit': 0,
        'duty_time_limit': 0,
        'cycle_rule': 0,
        'total_duty_time_limit': 0
    }
    
    print(f"\n分析 {len(duty_days)} 个值勤日:")
    
    for duty_date, day_tasks in duty_days:
        print(f"\n--- 值勤日: {duty_date.strftime('%Y-%m-%d')} ---")
        day_tasks = day_tasks.sort_values('taskId')  # 按任务ID排序
        
        # 1. 分析任务连接时间
        connection_violations = analyze_connection_time(day_tasks, data)
        total_violations['connection_time'] += connection_violations
        
        # 2. 分析飞行任务数量限制
        flight_task_violations = analyze_flight_task_limit(day_tasks)
        total_violations['flight_task_limit'] += flight_task_violations
        
        # 3. 分析值勤任务数量限制
        duty_task_violations = analyze_duty_task_limit(day_tasks)
        total_violations['duty_task_limit'] += duty_task_violations
        
        # 4. 分析飞行时间限制
        flight_time_violations = analyze_flight_time_limit(day_tasks, data)
        total_violations['flight_time_limit'] += flight_time_violations
        
        # 5. 分析值勤时间限制
        duty_time_violations = analyze_duty_time_limit(day_tasks, data)
        total_violations['duty_time_limit'] += duty_time_violations
    
    # 6. 分析休息时间限制（跨天）
    rest_violations = analyze_rest_time_limit(crew_tasks, data)
    total_violations['rest_time_limit'] += rest_violations
    
    # 7. 分析飞行周期要求
    cycle_violations = analyze_cycle_rule(crew_tasks)
    total_violations['cycle_rule'] += cycle_violations
    
    # 8. 分析总飞行值勤时间限制
    total_duty_violations = analyze_total_duty_time_limit(crew_tasks, data)
    total_violations['total_duty_time_limit'] += total_duty_violations
    
    # 输出总结
    print(f"\n{'='*60}")
    print(f"机组 {crew_id} 约束违规总结")
    print(f"{'='*60}")
    
    total_count = sum(total_violations.values())
    print(f"总违规数: {total_count}")
    print("\n详细统计:")
    
    violation_names = {
        'connection_time': '飞行值勤日内任务最小连接时间限制',
        'flight_task_limit': '飞行值勤日飞行任务数量限制',
        'duty_task_limit': '飞行值勤日值勤任务数量限制',
        'flight_time_limit': '飞行值勤日最大飞行时间限制',
        'rest_time_limit': '飞行值勤日最小休息时间限制',
        'duty_time_limit': '飞行值勤日最大飞行值勤时间限制',
        'cycle_rule': '飞行周期要求',
        'total_duty_time_limit': '总飞行值勤时间限制'
    }
    
    for key, count in total_violations.items():
        if count > 0:
            print(f"  - {violation_names[key]}: {count}")
    
    return total_violations

def analyze_connection_time(day_tasks, data):
    """分析连接时间违规"""
    violations = 0
    min_connection_time = timedelta(hours=3)  # 最小连接时间3小时
    
    print("\n1. 连接时间分析:")
    
    if len(day_tasks) < 2:
        print("  单个任务，无连接时间检查")
        return 0
    
    flights_data = data.get('flights')
    if flights_data is None:
        print("  无法获取航班数据，跳过连接时间检查")
        return 0
    
    tasks_list = day_tasks.to_dict('records')
    
    for i in range(len(tasks_list) - 1):
        current_task = tasks_list[i]
        next_task = tasks_list[i + 1]
        
        # 获取当前任务结束时间和下一任务开始时间
        current_end_time = get_task_end_time(current_task, flights_data)
        next_start_time = get_task_start_time(next_task, flights_data)
        
        if current_end_time and next_start_time:
            connection_time = next_start_time - current_end_time
            
            print(f"  {current_task.get('taskId', 'Unknown')} -> {next_task.get('taskId', 'Unknown')}")
            print(f"    连接时间: {connection_time}, 要求: {min_connection_time}")
            
            if connection_time < min_connection_time:
                violations += 1
                print(f"    ❌ 违规: 连接时间不足")
            else:
                print(f"    ✅ 符合要求")
    
    print(f"  连接时间违规数: {violations}")
    return violations

def get_task_end_time(task, flights_data):
    """获取任务结束时间"""
    task_id = task.get('taskId', '')
    
    if task_id.startswith('Flt_'):
        # 航班任务
        flight_info = flights_data[flights_data['id'] == task_id]
        if not flight_info.empty:
            return parse_datetime(flight_info.iloc[0]['sta'])
    
    # 地面任务或其他，使用默认逻辑
    return None

def get_task_start_time(task, flights_data):
    """获取任务开始时间"""
    task_id = task.get('taskId', '')
    
    if task_id.startswith('Flt_'):
        # 航班任务
        flight_info = flights_data[flights_data['id'] == task_id]
        if not flight_info.empty:
            return parse_datetime(flight_info.iloc[0]['std'])
    
    # 地面任务或其他，使用默认逻辑
    return None

def analyze_flight_task_limit(day_tasks):
    """分析飞行任务数量限制"""
    print("\n2. 飞行任务数量限制分析:")
    
    flight_tasks = day_tasks[day_tasks['taskId'].str.startswith('Flt_', na=False)]
    flight_count = len(flight_tasks)
    max_flight_tasks = 8  # 假设最大飞行任务数为8
    
    print(f"  当日飞行任务数: {flight_count}")
    print(f"  最大允许数: {max_flight_tasks}")
    
    if flight_count > max_flight_tasks:
        violations = 1
        print(f"  ❌ 违规: 超过最大飞行任务数")
    else:
        violations = 0
        print(f"  ✅ 符合要求")
    
    return violations

def analyze_duty_task_limit(day_tasks):
    """分析值勤任务数量限制"""
    print("\n3. 值勤任务数量限制分析:")
    
    task_count = len(day_tasks)
    max_duty_tasks = 12  # 假设最大值勤任务数为12
    
    print(f"  当日值勤任务数: {task_count}")
    print(f"  最大允许数: {max_duty_tasks}")
    
    if task_count > max_duty_tasks:
        violations = 1
        print(f"  ❌ 违规: 超过最大值勤任务数")
    else:
        violations = 0
        print(f"  ✅ 符合要求")
    
    return violations

def analyze_flight_time_limit(day_tasks, data):
    """分析飞行时间限制"""
    print("\n4. 飞行时间限制分析:")
    
    flights_data = data.get('flights')
    if flights_data is None:
        print("  无法获取航班数据，跳过飞行时间检查")
        return 0
    
    flight_tasks = day_tasks[day_tasks['taskId'].str.startswith('Flt_', na=False)]
    total_flight_time = 0
    
    for _, task in flight_tasks.iterrows():
        task_id = task['taskId']
        flight_info = flights_data[flights_data['id'] == task_id]
        if not flight_info.empty:
            fly_time = flight_info.iloc[0]['flyTime']  # 分钟
            total_flight_time += fly_time
    
    total_flight_hours = total_flight_time / 60
    max_flight_hours = 10  # 假设最大飞行时间为10小时
    
    print(f"  当日总飞行时间: {total_flight_hours:.2f}小时")
    print(f"  最大允许时间: {max_flight_hours}小时")
    
    if total_flight_hours > max_flight_hours:
        violations = 1
        print(f"  ❌ 违规: 超过最大飞行时间")
    else:
        violations = 0
        print(f"  ✅ 符合要求")
    
    return violations

def analyze_duty_time_limit(day_tasks, data):
    """分析值勤时间限制"""
    print("\n5. 值勤时间限制分析:")
    
    flights_data = data.get('flights')
    if flights_data is None or len(day_tasks) == 0:
        print("  无法获取数据或无任务，跳过值勤时间检查")
        return 0
    
    # 计算值勤时间（从第一个任务开始到最后一个任务结束）
    first_task = day_tasks.iloc[0]
    last_task = day_tasks.iloc[-1]
    
    start_time = get_task_start_time(first_task, flights_data)
    end_time = get_task_end_time(last_task, flights_data)
    
    if start_time and end_time:
        duty_time = end_time - start_time
        duty_hours = duty_time.total_seconds() / 3600
        max_duty_hours = 14  # 假设最大值勤时间为14小时
        
        print(f"  值勤开始时间: {start_time}")
        print(f"  值勤结束时间: {end_time}")
        print(f"  总值勤时间: {duty_hours:.2f}小时")
        print(f"  最大允许时间: {max_duty_hours}小时")
        
        if duty_hours > max_duty_hours:
            violations = 1
            print(f"  ❌ 违规: 超过最大值勤时间")
        else:
            violations = 0
            print(f"  ✅ 符合要求")
    else:
        violations = 0
        print(f"  无法计算值勤时间")
    
    return violations

def analyze_rest_time_limit(crew_tasks, data):
    """分析休息时间限制"""
    print("\n6. 休息时间限制分析:")
    
    flights_data = data.get('flights')
    if flights_data is None:
        print("  无法获取航班数据，跳过休息时间检查")
        return 0
    
    violations = 0
    min_rest_hours = 12  # 最小休息时间12小时
    
    # 按日期分组
    duty_days = crew_tasks.groupby('dutyDate')
    duty_dates = sorted(duty_days.groups.keys())
    
    for i in range(len(duty_dates) - 1):
        current_date = duty_dates[i]
        next_date = duty_dates[i + 1]
        
        current_day_tasks = duty_days.get_group(current_date)
        next_day_tasks = duty_days.get_group(next_date)
        
        # 获取当天最后一个任务的结束时间
        last_task_today = current_day_tasks.iloc[-1]
        end_time_today = get_task_end_time(last_task_today, flights_data)
        
        # 获取次日第一个任务的开始时间
        first_task_tomorrow = next_day_tasks.iloc[0]
        start_time_tomorrow = get_task_start_time(first_task_tomorrow, flights_data)
        
        if end_time_today and start_time_tomorrow:
            rest_time = start_time_tomorrow - end_time_today
            rest_hours = rest_time.total_seconds() / 3600
            
            print(f"  {current_date} -> {next_date}")
            print(f"    休息时间: {rest_hours:.2f}小时, 要求: {min_rest_hours}小时")
            
            if rest_hours < min_rest_hours:
                violations += 1
                print(f"    ❌ 违规: 休息时间不足")
            else:
                print(f"    ✅ 符合要求")
    
    print(f"  休息时间违规数: {violations}")
    return violations

def analyze_cycle_rule(crew_tasks):
    """分析飞行周期要求"""
    print("\n7. 飞行周期要求分析:")
    
    # 简化的飞行周期检查：连续工作天数不超过5天
    max_consecutive_days = 5
    violations = 0
    
    duty_dates = sorted(crew_tasks['dutyDate'].dt.date.unique())
    
    if len(duty_dates) <= max_consecutive_days:
        print(f"  总工作天数: {len(duty_dates)}天")
        print(f"  最大连续工作天数: {max_consecutive_days}天")
        print(f"  ✅ 符合要求")
        return 0
    
    # 检查连续工作天数
    consecutive_days = 1
    max_consecutive = 1
    
    for i in range(1, len(duty_dates)):
        if (duty_dates[i] - duty_dates[i-1]).days == 1:
            consecutive_days += 1
            max_consecutive = max(max_consecutive, consecutive_days)
        else:
            consecutive_days = 1
    
    print(f"  总工作天数: {len(duty_dates)}天")
    print(f"  最大连续工作天数: {max_consecutive}天")
    print(f"  允许的最大连续天数: {max_consecutive_days}天")
    
    if max_consecutive > max_consecutive_days:
        violations = 1
        print(f"  ❌ 违规: 连续工作天数超限")
    else:
        print(f"  ✅ 符合要求")
    
    return violations

def analyze_total_duty_time_limit(crew_tasks, data):
    """分析总飞行值勤时间限制"""
    print("\n8. 总飞行值勤时间限制分析:")
    
    flights_data = data.get('flights')
    if flights_data is None:
        print("  无法获取航班数据，跳过总值勤时间检查")
        return 0
    
    # 计算总飞行时间
    flight_tasks = crew_tasks[crew_tasks['taskId'].str.startswith('Flt_', na=False)]
    total_flight_time = 0
    
    for _, task in flight_tasks.iterrows():
        task_id = task['taskId']
        flight_info = flights_data[flights_data['id'] == task_id]
        if not flight_info.empty:
            fly_time = flight_info.iloc[0]['flyTime']  # 分钟
            total_flight_time += fly_time
    
    total_flight_hours = total_flight_time / 60
    max_total_flight_hours = 100  # 假设总飞行时间限制为100小时
    
    print(f"  总飞行时间: {total_flight_hours:.2f}小时")
    print(f"  最大允许时间: {max_total_flight_hours}小时")
    
    if total_flight_hours > max_total_flight_hours:
        violations = 1
        print(f"  ❌ 违规: 超过总飞行时间限制")
    else:
        violations = 0
        print(f"  ✅ 符合要求")
    
    return violations

def main():
    """主函数"""
    print("单个机组约束违规详细分析工具")
    print("=" * 50)
    
    # 加载数据
    print("正在加载数据...")
    data = load_data()
    
    if 'roster' not in data or data['roster'] is None:
        print("错误: 无法加载排班数据")
        return
    
    # 获取所有机组ID
    crew_ids = data['roster']['crewId'].unique()
    print(f"\n找到 {len(crew_ids)} 个机组")
    
    # 让用户选择要分析的机组
    print("\n前10个机组ID:")
    for i, crew_id in enumerate(crew_ids[:10]):
        print(f"  {i+1}. {crew_id}")
    
    # 默认分析第一个机组，或者用户可以修改这里
    target_crew = crew_ids[0]  # 可以修改为具体的机组ID
    print(f"\n正在分析机组: {target_crew}")
    
    # 分析指定机组
    violations = analyze_single_crew(target_crew, data)
    
    print(f"\n分析完成！")

if __name__ == "__main__":
    main()