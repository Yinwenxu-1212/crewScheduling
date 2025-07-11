#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务优先级分析脚本
分析不同类型任务的优先级分布，帮助找到合适的平衡点
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import DataHandler
from environment import CrewRosteringEnv
import config
import pandas as pd
import numpy as np

def analyze_task_priorities():
    """分析任务优先级分布"""
    print("=== 任务优先级分析 ===")
    
    # 1. 初始化环境
    print("\n1. 初始化环境...")
    dh = DataHandler()
    env = CrewRosteringEnv(dh)
    env.reset()
    
    # 2. 选择一个机组进行分析
    crew = env.crews[0]
    crew_info = crew
    crew_state = env.crew_states[crew['crewId']]
    
    print(f"\n2. 分析机组: {crew['crewId']}")
    print(f"   当前位置: {crew_state['last_location']}")
    print(f"   当前时间: {crew_state['last_task_end_time']}")
    
    # 3. 获取可用动作
    valid_actions, relaxed_actions = env._get_valid_actions(crew_info, crew_state)
    
    print(f"\n3. 可用动作统计:")
    print(f"   有效动作数: {len(valid_actions)}")
    print(f"   宽松动作数: {len(relaxed_actions)}")
    
    # 合并所有动作进行分析
    all_actions = valid_actions + relaxed_actions
    
    flight_tasks = [a for a in all_actions if a.get('type') == 'flight']
    ground_duty_tasks = [a for a in all_actions if a.get('type') == 'ground_duty']
    positioning_tasks = [a for a in all_actions if 'positioning' in str(a.get('type', ''))]
    
    print(f"   航班任务数量: {len(flight_tasks)}")
    print(f"   占位任务数量: {len(ground_duty_tasks)}")
    print(f"   置位任务数量: {len(positioning_tasks)}")
    
    # 4. 计算各类任务的优先级分布
    print(f"\n4. 任务优先级分析:")
    
    # 航班任务优先级
    if flight_tasks:
        flight_priorities = []
        for task in flight_tasks[:5]:  # 分析前5个航班任务
            priority = env._calculate_dynamic_priority(task, crew_state, crew_info)
            flight_priorities.append(priority)
            print(f"   航班任务 {task.get('taskId', 'N/A')}: 优先级 {priority:.1f}")
        
        print(f"   航班任务优先级范围: {min(flight_priorities):.1f} - {max(flight_priorities):.1f}")
        print(f"   航班任务平均优先级: {np.mean(flight_priorities):.1f}")
    
    # 占位任务优先级
    if ground_duty_tasks:
        ground_priorities = []
        for task in ground_duty_tasks[:3]:  # 分析前3个占位任务
            priority = env._calculate_dynamic_priority(task, crew_state, crew_info)
            ground_priorities.append(priority)
            print(f"   占位任务 {task.get('taskId', 'N/A')}: 优先级 {priority:.1f}")
        
        print(f"   占位任务优先级范围: {min(ground_priorities):.1f} - {max(ground_priorities):.1f}")
        print(f"   占位任务平均优先级: {np.mean(ground_priorities):.1f}")
    
    # 置位任务优先级
    if positioning_tasks:
        positioning_priorities = []
        positioning_strategic_values = []
        
        for task in positioning_tasks[:5]:  # 分析前5个置位任务
            priority = env._calculate_dynamic_priority(task, crew_state, crew_info)
            strategic_value = env._calculate_positioning_strategic_value(task, crew_info)
            positioning_priorities.append(priority)
            positioning_strategic_values.append(strategic_value)
            
            print(f"   置位任务 -> {task.get('arriAirport', 'N/A')}: 优先级 {priority:.1f}, 战略价值 {strategic_value:.1f}")
        
        print(f"   置位任务优先级范围: {min(positioning_priorities):.1f} - {max(positioning_priorities):.1f}")
        print(f"   置位任务平均优先级: {np.mean(positioning_priorities):.1f}")
        print(f"   置位任务战略价值范围: {min(positioning_strategic_values):.1f} - {max(positioning_strategic_values):.1f}")
    
    # 5. 分析优先级竞争情况
    print(f"\n5. 优先级竞争分析:")
    
    all_tasks_with_priorities = []
    
    # 添加航班任务
    for task in flight_tasks[:3]:
        priority = env._calculate_dynamic_priority(task, crew_state, crew_info)
        all_tasks_with_priorities.append({
            'type': 'flight',
            'id': task.get('taskId', 'N/A'),
            'priority': priority,
            'details': f"航班 {task.get('taskId', 'N/A')}"
        })
    
    # 添加占位任务
    for task in ground_duty_tasks[:2]:
        priority = env._calculate_dynamic_priority(task, crew_state, crew_info)
        all_tasks_with_priorities.append({
            'type': 'ground_duty',
            'id': task.get('taskId', 'N/A'),
            'priority': priority,
            'details': f"占位 {task.get('taskId', 'N/A')}"
        })
    
    # 添加置位任务
    for task in positioning_tasks[:3]:
        priority = env._calculate_dynamic_priority(task, crew_state, crew_info)
        strategic_value = env._calculate_positioning_strategic_value(task, crew_info)
        all_tasks_with_priorities.append({
            'type': 'positioning',
            'id': task.get('taskId', f"pos_{task.get('arriAirport', 'N/A')}"),
            'priority': priority,
            'strategic_value': strategic_value,
            'details': f"置位 -> {task.get('arriAirport', 'N/A')} (战略价值: {strategic_value:.1f})"
        })
    
    # 按优先级排序
    all_tasks_with_priorities.sort(key=lambda x: x['priority'], reverse=True)
    
    print("   综合优先级排序 (前10名):")
    for i, task in enumerate(all_tasks_with_priorities[:10]):
        print(f"   {i+1:2d}. {task['details']:40s} 优先级: {task['priority']:6.1f}")
    
    # 6. 分析置位任务需要的最低优先级
    print(f"\n6. 置位任务竞争力分析:")
    
    if flight_tasks and positioning_tasks:
        min_flight_priority = min([env._calculate_dynamic_priority(task, crew_state, crew_info) for task in flight_tasks[:5]])
        max_positioning_priority = max([env._calculate_dynamic_priority(task, crew_state, crew_info) for task in positioning_tasks[:5]])
        
        print(f"   最低航班任务优先级: {min_flight_priority:.1f}")
        print(f"   最高置位任务优先级: {max_positioning_priority:.1f}")
        print(f"   优先级差距: {min_flight_priority - max_positioning_priority:.1f}")
        
        if max_positioning_priority >= min_flight_priority:
            print("   ✓ 置位任务具有竞争力")
        else:
            print(f"   ✗ 置位任务需要提升 {min_flight_priority - max_positioning_priority:.1f} 分才能竞争")
    
    # 7. 覆盖率影响分析
    print(f"\n7. 覆盖率影响分析:")
    current_coverage = env._calculate_current_coverage_rate()
    print(f"   当前覆盖率: {current_coverage:.1%}")
    
    if positioning_tasks:
        # 模拟不同覆盖率下的置位优先级
        sample_positioning = positioning_tasks[0]
        
        # 临时修改覆盖率来测试影响
        original_unassigned = len(env.unassigned_flight_ids)
        total_flights = len(env.dh.data['flights'])
        
        test_coverages = [0.3, 0.5, 0.7, 0.9]
        for test_coverage in test_coverages:
            # 临时设置未分配航班数量
            test_unassigned = int(total_flights * (1 - test_coverage))
            env.unassigned_flight_ids = set(list(env.unassigned_flight_ids)[:test_unassigned])
            
            priority = env._calculate_dynamic_priority(sample_positioning, crew_state, crew_info)
            print(f"   覆盖率 {test_coverage:.1%} 时，置位优先级: {priority:.1f}")
        
        # 恢复原始状态
        env.unassigned_flight_ids = set(list(env.unassigned_flight_ids)[:original_unassigned])
    
    print("\n=== 分析完成 ===")

if __name__ == "__main__":
    analyze_task_priorities()