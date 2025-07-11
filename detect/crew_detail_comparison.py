#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机组排班结果与官方数据对比分析工具
对比rosterResult.csv与crew_detail.csv的差异
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

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

def load_roster_result(file_path):
    """加载排班结果数据"""
    try:
        data = pd.read_csv(file_path)
        print(f"成功加载排班结果: {file_path}")
        print(f"数据形状: {data.shape}")
        print(f"列名: {list(data.columns)}")
        return data
    except Exception as e:
        print(f"加载排班结果失败: {e}")
        return None

def load_crew_detail(file_path):
    """加载官方机组详细数据"""
    try:
        data = pd.read_csv(file_path)
        print(f"成功加载官方数据: {file_path}")
        print(f"数据形状: {data.shape}")
        print(f"列名: {list(data.columns)}")
        return data
    except Exception as e:
        print(f"加载官方数据失败: {e}")
        return None

def analyze_roster_stats(roster_data):
    """分析排班结果统计信息"""
    stats = {}
    
    if roster_data is None or roster_data.empty:
        return stats
    
    try:
        # 基本统计
        stats['total_records'] = len(roster_data)
        
        # 机组统计
        if 'crewId' in roster_data.columns:
            stats['unique_crews'] = roster_data['crewId'].nunique()
        
        # 航班统计
        if 'flightId' in roster_data.columns:
            stats['unique_flights'] = roster_data['flightId'].nunique()
        
        # 计算飞行时间统计
        if 'flyTime' in roster_data.columns:
            flight_times = roster_data['flyTime'].dropna()
            if len(flight_times) > 0:
                stats['avg_flight_time'] = flight_times.mean()
                stats['total_flight_time'] = flight_times.sum()
                stats['max_flight_time'] = flight_times.max()
        
        # 值勤日统计
        if 'dutyDate' in roster_data.columns:
            stats['unique_duty_dates'] = roster_data['dutyDate'].nunique()
        
        # 任务类型统计
        if 'taskType' in roster_data.columns:
            stats['task_types'] = roster_data['taskType'].value_counts().to_dict()
        
    except Exception as e:
        print(f"分析排班统计时出错: {e}")
    
    return stats

def analyze_crew_detail_stats(crew_data):
    """分析官方机组数据统计信息"""
    stats = {}
    
    if crew_data is None or crew_data.empty:
        return stats
    
    try:
        # 基本统计
        stats['total_records'] = len(crew_data)
        
        # 机组统计
        if 'crewId' in crew_data.columns:
            stats['unique_crews'] = crew_data['crewId'].nunique()
        elif 'crew_id' in crew_data.columns:
            stats['unique_crews'] = crew_data['crew_id'].nunique()
        
        # 分析所有数值列
        numeric_cols = crew_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_data = crew_data[col].dropna()
            if len(col_data) > 0:
                stats[f'{col}_mean'] = col_data.mean()
                stats[f'{col}_sum'] = col_data.sum()
                stats[f'{col}_max'] = col_data.max()
                stats[f'{col}_min'] = col_data.min()
        
    except Exception as e:
        print(f"分析官方数据统计时出错: {e}")
    
    return stats

def compare_datasets(roster_stats, crew_stats):
    """对比两个数据集的统计信息"""
    comparison = {}
    
    # 对比共同的指标
    common_keys = set(roster_stats.keys()) & set(crew_stats.keys())
    
    for key in common_keys:
        roster_val = roster_stats.get(key, 0)
        crew_val = crew_stats.get(key, 0)
        
        comparison[key] = {
            'roster_result': roster_val,
            'crew_detail': crew_val,
            'difference': roster_val - crew_val if isinstance(roster_val, (int, float)) and isinstance(crew_val, (int, float)) else 'N/A'
        }
    
    return comparison

def generate_report(roster_stats, crew_stats, comparison):
    """生成对比分析报告"""
    report = []
    report.append("=" * 80)
    report.append("机组排班结果与官方数据对比分析报告")
    report.append("=" * 80)
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # 排班结果统计
    report.append("1. 排班结果统计 (rosterResult.csv)")
    report.append("-" * 40)
    for key, value in roster_stats.items():
        if isinstance(value, dict):
            report.append(f"{key}:")
            for sub_key, sub_value in value.items():
                report.append(f"  {sub_key}: {sub_value}")
        else:
            report.append(f"{key}: {value}")
    report.append("")
    
    # 官方数据统计
    report.append("2. 官方数据统计 (crew_detail.csv)")
    report.append("-" * 40)
    for key, value in crew_stats.items():
        if isinstance(value, dict):
            report.append(f"{key}:")
            for sub_key, sub_value in value.items():
                report.append(f"  {sub_key}: {sub_value}")
        else:
            report.append(f"{key}: {value}")
    report.append("")
    
    # 对比分析
    report.append("3. 对比分析")
    report.append("-" * 40)
    if comparison:
        for key, comp in comparison.items():
            report.append(f"{key}:")
            report.append(f"  排班结果: {comp['roster_result']}")
            report.append(f"  官方数据: {comp['crew_detail']}")
            report.append(f"  差异: {comp['difference']}")
            report.append("")
    else:
        report.append("未找到可对比的共同指标")
    
    return "\n".join(report)

def main():
    """主函数"""
    print("开始机组排班结果与官方数据对比分析...")
    
    # 文件路径
    roster_path = "submit/0710-2-（-4224）/rosterResult.csv"
    crew_detail_path = "submit/0710-2-（-4224）/crew_detail.csv"
    
    # 加载数据
    print("\n正在加载数据...")
    roster_data = load_roster_result(roster_path)
    crew_data = load_crew_detail(crew_detail_path)
    
    if roster_data is None and crew_data is None:
        print("错误: 无法加载任何数据文件")
        return
    
    # 分析统计信息
    print("\n正在分析统计信息...")
    roster_stats = analyze_roster_stats(roster_data)
    crew_stats = analyze_crew_detail_stats(crew_data)
    
    # 对比分析
    print("\n正在进行对比分析...")
    comparison = compare_datasets(roster_stats, crew_stats)
    
    # 生成报告
    report = generate_report(roster_stats, crew_stats, comparison)
    
    # 输出报告
    print("\n" + report)
    
    # 保存报告到文件
    try:
        with open("crew_comparison_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n报告已保存到: crew_comparison_report.txt")
    except Exception as e:
        print(f"保存报告失败: {e}")

if __name__ == "__main__":
    main()