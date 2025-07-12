#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的机组约束分析工具
输入机组ID，分析其任务路径和约束违规情况
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from collections import defaultdict

def load_data():
    """加载所有必要的数据文件"""
    try:
        import os
        import glob
        
        # 切换到项目根目录
        current_dir = os.getcwd()
        if current_dir.endswith('detect'):
            os.chdir('..')
        
        # 查找submit目录下的rosterResult.csv文件
        roster_files = glob.glob('submit/*/rosterResult.csv')
        if not roster_files:
            raise FileNotFoundError("未找到任何rosterResult.csv文件")
        
        # 使用最新的结果文件（按文件夹名排序）
        roster_file = sorted(roster_files)[-1]
        print(f"使用排班结果文件: {roster_file}")
        roster_result = pd.read_csv(roster_file)
        
        # 加载航班数据
        flights = pd.read_csv('data/flight.csv')
        
        # 加载地面任务数据
        ground_duties = pd.read_csv('data/groundDuty.csv')
        
        # 加载DDH置位任务数据
        ddh_tasks = pd.read_csv('data/busInfo.csv')
        
        # 加载机组数据
        crews = pd.read_csv('data/crew.csv')
        
        # 加载可过夜机场数据
        layover_stations = pd.read_csv('data/layoverStation.csv')
        layover_airports = set(layover_stations['airport'].tolist())
        
        return {
            'roster_result': roster_result,
            'flights': flights,
            'ground_duties': ground_duties,
            'ddh_tasks': ddh_tasks,
            'crews': crews,
            'layover_airports': layover_airports
        }
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None

def parse_datetime(date_str, time_str):
    """解析日期时间字符串"""
    try:
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    except:
        return None

def get_crew_tasks(data, crew_id):
    """获取指定机组的所有任务"""
    crew_tasks = data['roster_result'][data['roster_result']['crewId'] == crew_id].copy()
    
    if crew_tasks.empty:
        print(f"未找到机组 {crew_id} 的任务")
        return None
    
    # 为每个任务添加详细信息
    task_details = []
    
    for _, task in crew_tasks.iterrows():
        task_id = task['taskId']
        is_ddh = task['isDDH']
        
        # 查找任务详情
        if task_id.startswith('Flt_'):
            # 航班任务
            flight_info = data['flights'][data['flights']['id'] == task_id]
            if not flight_info.empty:
                flight = flight_info.iloc[0]
                task_detail = {
                    'taskId': task_id,
                    'type': '航班',
                    'isDDH': is_ddh,
                    'date': flight['std'].split(' ')[0],  # 从std中提取日期部分
                    'depTime': flight['std'],  # 完整的出发时间
                    'arrTime': flight['sta'],  # 完整的到达时间
                    'depAirport': flight['depaAirport'],
                    'arrAirport': flight['arriAirport'],
                    'flyTime': flight['flyTime'] / 60.0,  # 转换为小时（原数据是分钟）
                    'aircraftType': flight['fleet'],
                    'tailNumber': flight['aircraftNo']
                }
                task_details.append(task_detail)
        elif task_id.startswith('Grd_'):
            # 地面任务
            ground_info = data['ground_duties'][data['ground_duties']['id'] == task_id]
            if not ground_info.empty:
                ground = ground_info.iloc[0]
                # 从startTime中提取日期
                start_time_parts = ground['startTime'].split(' ')
                task_date = start_time_parts[0] if len(start_time_parts) > 1 else ground['startTime']
                task_detail = {
                    'taskId': task_id,
                    'type': '地面任务',
                    'isDDH': is_ddh,
                    'date': task_date,
                    'depTime': ground['startTime'],
                    'arrTime': ground['endTime'],
                    'depAirport': ground['airport'],
                    'arrAirport': ground['airport'],  # 地面任务起止都在同一机场
                    'flyTime': 0,
                    'aircraftType': '',
                    'tailNumber': ''
                }
                task_details.append(task_detail)
        elif task_id.startswith('ddh_'):
            # DDH置位任务
            ddh_info = data['ddh_tasks'][data['ddh_tasks']['id'] == task_id]
            if not ddh_info.empty:
                ddh = ddh_info.iloc[0]
                # 计算DDH任务的飞行时间
                ddh_start = parse_datetime(ddh['td'].split(' ')[0], ddh['td'].split(' ')[1])
                ddh_end = parse_datetime(ddh['ta'].split(' ')[0], ddh['ta'].split(' ')[1])
                ddh_duration = 0
                if ddh_start and ddh_end:
                    ddh_duration = (ddh_end - ddh_start).total_seconds() / 3600.0  # 转换为小时
                
                task_detail = {
                    'taskId': task_id,
                    'type': 'DDH置位',
                    'isDDH': True,
                    'date': ddh['td'].split(' ')[0],  # 从td中提取日期部分
                    'depTime': ddh['td'],  # 完整的出发时间
                    'arrTime': ddh['ta'],  # 完整的到达时间
                    'depAirport': ddh['depaAirport'],
                    'arrAirport': ddh['arriAirport'],
                    'flyTime': ddh_duration,  # DDH置位任务计入飞行时间
                    'aircraftType': '',
                    'tailNumber': ''
                }
                task_details.append(task_detail)
    
    # 按日期和时间排序
    task_details.sort(key=lambda x: (x['date'], x['depTime']))
    
    return task_details

def analyze_duty_periods(tasks, layover_airports):
    """分析值勤日和飞行值勤日
    
    值勤日：机组人员一次出勤需要完成的一连串值勤任务（包含飞行任务、置位任务、占位任务）
    飞行值勤日：机组人员一次出勤需要完成的一连串飞行或置位任务，必须包含飞行任务，
                只能从可过夜机场出发到可过夜机场结束
    """
    if not tasks:
        return [], []
    
    duty_periods = []
    flight_duty_periods = []
    
    current_duty = []
    current_flight_duty = []
    
    for i, task in enumerate(tasks):
        task_datetime = parse_datetime(task['date'], task['depTime'])
        
        if i == 0:
            # 第一个任务
            current_duty = [task]
            if task['type'] == '航班':
                current_flight_duty = [task]
        else:
            prev_task = tasks[i-1]
            prev_end_datetime = parse_datetime(prev_task['date'], prev_task['arrTime'])
            
            # 检查是否需要开始新的值勤日
            if task_datetime and prev_end_datetime:
                rest_time = (task_datetime - prev_end_datetime).total_seconds() / 3600
                
                # 计算当前值勤日的时间跨度（从第一个任务开始到当前任务结束）
                duty_start_datetime = parse_datetime(current_duty[0]['date'], current_duty[0]['depTime'])
                current_task_end_datetime = parse_datetime(task['date'], task['arrTime'])
                duty_duration = (current_task_end_datetime - duty_start_datetime).total_seconds() / 3600 if (duty_start_datetime and current_task_end_datetime) else 0
                
                # 调试信息（临时）
                # print(f"调试: 任务{task['taskId']}, 值勤日时长: {duty_duration:.1f}小时, 休息时间: {rest_time:.1f}小时")
                
                # 开始新值勤日的条件：
                # 1. 休息时间>=8小时，或者跨日且有休息时间>=4小时
                # 2. 或者当前值勤日时间跨度将超过24小时
                is_new_duty = (rest_time >= 8) or (
                    task_datetime.date() != prev_end_datetime.date() and rest_time >= 4
                ) or (duty_duration > 24)
                
                if is_new_duty:
                    # 结束当前值勤日
                    if current_duty:
                        duty_periods.append(current_duty.copy())
                    
                    # 检查当前飞行值勤日是否有效（包含飞行任务且从可过夜机场到可过夜机场）
                    if current_flight_duty and _is_valid_flight_duty_period(current_flight_duty, layover_airports):
                        flight_duty_periods.append(current_flight_duty.copy())
                    
                    # 开始新的值勤日
                    current_duty = [task]
                    if task['type'] in ['航班', 'DDH置位']:
                        current_flight_duty = [task]
                    else:
                        current_flight_duty = []
                else:
                    # 继续当前值勤日
                    current_duty.append(task)
                    if task['type'] in ['航班', 'DDH置位']:  # 飞行任务或置位任务
                        current_flight_duty.append(task)
            else:
                current_duty.append(task)
                if task['type'] in ['航班', 'DDH置位']:  # 飞行任务或置位任务
                    current_flight_duty.append(task)
    
    # 添加最后的值勤日
    if current_duty:
        duty_periods.append(current_duty)
    if current_flight_duty and _is_valid_flight_duty_period(current_flight_duty, layover_airports):
        flight_duty_periods.append(current_flight_duty)
    
    return duty_periods, flight_duty_periods

def _is_valid_flight_duty_period(flight_duty_tasks, layover_airports):
    """检查飞行值勤日是否有效
    
    条件：
    1. 必须包含飞行任务
    2. 只能从可过夜机场出发到可过夜机场结束
    """
    if not flight_duty_tasks:
        return False
    
    # 检查是否包含飞行任务
    has_flight = any(task['type'] == '航班' for task in flight_duty_tasks)
    if not has_flight:
        return False
    
    # 检查起始和结束机场是否为可过夜机场
    start_airport = flight_duty_tasks[0]['depAirport']
    end_airport = flight_duty_tasks[-1]['arrAirport']
    
    return start_airport in layover_airports and end_airport in layover_airports

def analyze_flight_cycles(duty_periods, flight_duty_periods):
    """分析飞行周期
    
    飞行周期：由值勤日组成的周期，必须包含飞行值勤日，且飞行周期末尾一定为飞行值勤日
    限制：最多横跨4个日历日，开始前必须连续休息2个完整日历日
    """
    if not duty_periods:
        return []
    
    flight_cycles = []
    current_cycle_duties = []
    
    for i, duty in enumerate(duty_periods):
        if i == 0:
            current_cycle_duties = [duty]
        else:
            prev_duty = duty_periods[i-1]
            prev_end_datetime = parse_datetime(prev_duty[-1]['date'], prev_duty[-1]['arrTime'])
            curr_start_datetime = parse_datetime(duty[0]['date'], duty[0]['depTime'])
            
            if curr_start_datetime and prev_end_datetime:
                rest_time = (curr_start_datetime - prev_end_datetime).total_seconds() / 3600
                
                # 检查是否有2个完整日历日的休息
                # 计算休息期间跨越的完整日历日数量
                rest_start_date = prev_end_datetime.date()
                rest_end_date = curr_start_datetime.date()
                
                # 计算中间的完整日历日数量
                from datetime import timedelta
                complete_rest_days = 0
                current_date = rest_start_date + timedelta(days=1)
                while current_date < rest_end_date:
                    complete_rest_days += 1
                    current_date += timedelta(days=1)
                
                # 如果有至少2个完整日历日的休息，则开始新飞行周期
                if complete_rest_days >= 2:
                    # 检查当前周期是否有效（包含飞行值勤日且末尾为飞行值勤日）
                    if _is_valid_flight_cycle(current_cycle_duties, flight_duty_periods):
                        flight_cycles.append(current_cycle_duties.copy())
                    current_cycle_duties = [duty]
                else:
                    current_cycle_duties.append(duty)
            else:
                current_cycle_duties.append(duty)
    
    # 添加最后的飞行周期
    if current_cycle_duties and _is_valid_flight_cycle(current_cycle_duties, flight_duty_periods):
        flight_cycles.append(current_cycle_duties)
    
    return flight_cycles

def _is_valid_flight_cycle(cycle_duties, flight_duty_periods):
    """检查飞行周期是否有效
    
    条件：
    1. 必须包含飞行值勤日
    2. 末尾一定为飞行值勤日
    """
    if not cycle_duties:
        return False
    
    # 检查是否包含飞行值勤日
    has_flight_duty = False
    for duty in cycle_duties:
        for flight_duty in flight_duty_periods:
            if duty == flight_duty:
                has_flight_duty = True
                break
        if has_flight_duty:
            break
    
    if not has_flight_duty:
        return False
    
    # 检查末尾是否为飞行值勤日
    last_duty = cycle_duties[-1]
    is_last_flight_duty = False
    for flight_duty in flight_duty_periods:
        if last_duty == flight_duty:
            is_last_flight_duty = True
            break
    
    return is_last_flight_duty

def check_constraints(tasks, duty_periods, flight_duty_periods, flight_cycles):
    """检查所有约束"""
    violations = []
    
    # (3) 飞行值勤日内相邻任务的最小连接时间限制
    for period_idx, period in enumerate(flight_duty_periods):
        for i in range(len(period) - 1):
            curr_task = period[i]
            next_task = period[i + 1]
            
            curr_end = parse_datetime(curr_task['date'], curr_task['arrTime'])
            next_start = parse_datetime(next_task['date'], next_task['depTime'])
            
            if curr_end and next_start:
                connection_time = (next_start - curr_end).total_seconds() / 60  # 分钟
                
                # 判断最小连接时间要求
                if curr_task['type'] == '航班' and next_task['type'] == '航班':
                    # 航班到航班
                    if curr_task['tailNumber'] != next_task['tailNumber']:
                        min_time = 180  # 3小时
                    else:
                        min_time = 60   # 同机型1小时（假设）
                elif curr_task['type'] == '地面任务' or next_task['type'] == '地面任务':
                    # 涉及地面任务（大巴置位）
                    min_time = 120  # 2小时
                else:
                    min_time = 60   # 默认1小时
                
                if connection_time < min_time:
                    violations.append({
                        'rule': '(3) 飞行值勤日内相邻任务的最小连接时间限制',
                        'period': period_idx + 1,
                        'detail': f"{curr_task['taskId']} -> {next_task['taskId']}: 实际{connection_time:.0f}分钟 < 要求{min_time}分钟"
                    })
    
    # (4) 飞行值勤日任务数量限制
    for period_idx, period in enumerate(flight_duty_periods):
        flight_count = sum(1 for task in period if task['type'] == '航班')
        total_count = len(period)
        
        if flight_count > 4:
            violations.append({
                'rule': '(4) 飞行值勤日任务数量限制',
                'period': period_idx + 1,
                'detail': f"飞行任务数量{flight_count} > 4"
            })
        
        if total_count > 6:
            violations.append({
                'rule': '(4) 飞行值勤日任务数量限制',
                'period': period_idx + 1,
                'detail': f"总任务数量{total_count} > 6"
            })
    
    # (5) 飞行值勤日最大飞行时间限制
    for period_idx, period in enumerate(flight_duty_periods):
        total_flight_time = sum(task['flyTime'] for task in period if task['type'] in ['航班', 'DDH置位'])
        
        if total_flight_time > 8:
            violations.append({
                'rule': '(5) 飞行值勤日最大飞行时间限制',
                'period': period_idx + 1,
                'detail': f"飞行时间{total_flight_time:.1f}小时 > 8小时"
            })
    
    # (6) 飞行值勤日最大飞行值勤时间限制
    for period_idx, period in enumerate(flight_duty_periods):
        if period:
            start_time = parse_datetime(period[0]['date'], period[0]['depTime'])
            end_time = parse_datetime(period[-1]['date'], period[-1]['arrTime'])
            
            if start_time and end_time:
                duty_time = (end_time - start_time).total_seconds() / 3600
                
                if duty_time > 12:
                    violations.append({
                        'rule': '(6) 飞行值勤日最大飞行值勤时间限制',
                        'period': period_idx + 1,
                        'detail': f"值勤时间{duty_time:.1f}小时 > 12小时"
                    })
    
    # (7) 飞行值勤日开始前最小休息时间限制
    for period_idx in range(1, len(flight_duty_periods)):
        prev_period = flight_duty_periods[period_idx - 1]
        curr_period = flight_duty_periods[period_idx]
        
        if prev_period and curr_period:
            prev_end = parse_datetime(prev_period[-1]['date'], prev_period[-1]['arrTime'])
            curr_start = parse_datetime(curr_period[0]['date'], curr_period[0]['depTime'])
            
            if prev_end and curr_start:
                rest_time = (curr_start - prev_end).total_seconds() / 3600
                
                if rest_time < 12:
                    violations.append({
                        'rule': '(7) 飞行值勤日开始前最小休息时间限制',
                        'period': period_idx + 1,
                        'detail': f"休息时间{rest_time:.1f}小时 < 12小时"
                    })
    
    # (8) 飞行周期限制
    for cycle_idx, cycle_duties in enumerate(flight_cycles):
        if cycle_duties:
            # 获取周期的第一个和最后一个值勤日
            first_duty = cycle_duties[0]
            last_duty = cycle_duties[-1]
            
            # 处理日期格式转换
            date_str = first_duty[0]['date']
            if '/' in date_str:
                # 转换 '2025/5/1' 格式为 '2025-05-01'
                parts = date_str.split('/')
                date_str = f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
            start_date = datetime.strptime(date_str, "%Y-%m-%d")
            
            # 处理结束日期格式转换
            end_date_str = last_duty[-1]['date']
            if '/' in end_date_str:
                # 转换 '2025/5/1' 格式为 '2025-05-01'
                parts = end_date_str.split('/')
                end_date_str = f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
            cycle_days = (end_date - start_date).days + 1
            
            if cycle_days > 4:
                violations.append({
                    'rule': '(8) 飞行周期限制',
                    'cycle': cycle_idx + 1,
                    'detail': f"飞行周期{cycle_days}天 > 4天"
                })
    
    # (9) 总飞行值勤时间限制
    total_flight_time = sum(task['flyTime'] for task in tasks if task['type'] in ['航班', 'DDH置位'])
    if total_flight_time > 60:
        violations.append({
            'rule': '(9) 总飞行值勤时间限制',
            'detail': f"总飞行时间{total_flight_time:.1f}小时 > 60小时"
        })
    
    return violations

def print_task_path(tasks):
    """打印任务路径"""
    print("\n=== 任务路径 ===")
    for i, task in enumerate(tasks, 1):
        ddh_str = "(置位)" if task['isDDH'] else ""
        if task['type'] == '航班':
            print(f"{i:2d}. {task['taskId']} {ddh_str}")
            print(f"    {task['date']} {task['depTime']}-{task['arrTime']}")
            print(f"    {task['depAirport']} -> {task['arrAirport']}")
            print(f"    飞行时间: {task['flyTime']:.1f}小时, 机型: {task['aircraftType']}, 尾号: {task['tailNumber']}")
        elif task['type'] == 'DDH置位':
            print(f"{i:2d}. {task['taskId']} (DDH置位)")
            print(f"    {task['date']} {task['depTime']}-{task['arrTime']}")
            print(f"    {task['depAirport']} -> {task['arrAirport']}")
            print(f"    飞行时间: {task['flyTime']:.1f}小时")
        else:
            print(f"{i:2d}. {task['taskId']} (地面任务)")
            print(f"    {task['date']} {task['depTime']}-{task['arrTime']}")
            print(f"    {task['depAirport']} -> {task['arrAirport']}")
        print()

def print_duty_periods(duty_periods, flight_duty_periods):
    """打印值勤日和飞行值勤日"""
    print("\n=== 值勤日划分 ===")
    for i, period in enumerate(duty_periods, 1):
        print(f"值勤日 {i}: {period[0]['date']} {period[0]['depTime']} - {period[-1]['date']} {period[-1]['arrTime']}")
        print(f"  任务: {', '.join([task['taskId'] for task in period])}")
    
    print("\n=== 飞行值勤日划分 ===")
    for i, period in enumerate(flight_duty_periods, 1):
        if period:
            print(f"飞行值勤日 {i}: {period[0]['date']} {period[0]['depTime']} - {period[-1]['date']} {period[-1]['arrTime']}")
            flight_tasks = [task['taskId'] for task in period if task['type'] in ['航班', 'DDH置位']]
            print(f"  飞行任务: {', '.join(flight_tasks)}")

def print_flight_cycles(flight_cycles):
    """打印飞行周期"""
    print("\n=== 飞行周期划分 ===")
    for i, cycle_duties in enumerate(flight_cycles, 1):
        if cycle_duties:
            first_duty = cycle_duties[0]
            last_duty = cycle_duties[-1]
            print(f"飞行周期 {i}: {first_duty[0]['date']} - {last_duty[-1]['date']}")
            print(f"  包含 {len(cycle_duties)} 个值勤日")
            for j, duty in enumerate(cycle_duties, 1):
                tasks_str = ', '.join([task['taskId'] for task in duty])
                print(f"    值勤日{j}: {tasks_str}")

def print_constraint_results(violations):
    """打印约束检查结果"""
    print("\n=== 约束检查结果 ===")
    
    rules = [
        '(3) 飞行值勤日内相邻任务的最小连接时间限制',
        '(4) 飞行值勤日任务数量限制',
        '(5) 飞行值勤日最大飞行时间限制',
        '(6) 飞行值勤日最大飞行值勤时间限制',
        '(7) 飞行值勤日开始前最小休息时间限制',
        '(8) 飞行周期限制',
        '(9) 总飞行值勤时间限制'
    ]
    
    for rule in rules:
        rule_violations = [v for v in violations if v['rule'] == rule]
        if rule_violations:
            print(f"❌ {rule}: 发现 {len(rule_violations)} 个违规")
            for violation in rule_violations:
                print(f"   - {violation['detail']}")
        else:
            print(f"✅ {rule}: 满足")
    
    print(f"\n总违规数: {len(violations)}")

def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("使用方法: python simplified_crew_analyzer.py <crew_id>")
        print("例如: python simplified_crew_analyzer.py Crew_10002")
        return
    
    crew_id = sys.argv[1]
    
    # 加载数据
    data = load_data()
    if not data:
        return
    
    # 获取机组任务
    tasks = get_crew_tasks(data, crew_id)
    if not tasks:
        return
    
    print(f"\n分析机组: {crew_id}")
    print(f"总任务数: {len(tasks)}")
    print(f"可过夜机场数量: {len(data['layover_airports'])}")
    
    # 打印任务路径
    print_task_path(tasks)
    
    # 分析值勤日和飞行值勤日
    duty_periods, flight_duty_periods = analyze_duty_periods(tasks, data['layover_airports'])
    print_duty_periods(duty_periods, flight_duty_periods)
    
    # 分析飞行周期
    flight_cycles = analyze_flight_cycles(duty_periods, flight_duty_periods)
    print_flight_cycles(flight_cycles)
    
    # 检查约束
    violations = check_constraints(tasks, duty_periods, flight_duty_periods, flight_cycles)
    print_constraint_results(violations)

if __name__ == "__main__":
    main()