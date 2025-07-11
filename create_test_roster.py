#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
创建测试用的rosterResult.csv文件
"""

import pandas as pd
import os


def convert_alns_result_to_roster_format(input_file: str, output_file: str = None):
    """
    将ALNS结果文件转换为标准的rosterResult格式
    
    Args:
        input_file: 输入的ALNS结果文件
        output_file: 输出的rosterResult文件
    """
    if output_file is None:
        output_file = "test_rosterResult.csv"
    
    print(f"正在转换 {input_file} 到 {output_file}")
    
    # 读取ALNS结果文件
    df = pd.read_csv(input_file)
    
    print(f"原始文件包含 {len(df)} 条记录")
    print(f"列名: {list(df.columns)}")
    
    # 转换为标准格式
    roster_data = []
    
    for _, row in df.iterrows():
        crew_id = row['crewId']
        task_id = row['dutyId']
        
        # 根据任务类型判断是否为置位任务
        # 这里简化处理，假设所有大巴任务都是置位任务
        is_ddh = 1 if 'Bus_' in str(task_id) else 0
        
        roster_data.append({
            'crewId': crew_id,
            'taskId': task_id,
            'isDDH': is_ddh
        })
    
    # 创建DataFrame并保存
    roster_df = pd.DataFrame(roster_data)
    roster_df.to_csv(output_file, index=False)
    
    print(f"转换完成，保存到 {output_file}")
    print(f"转换后包含 {len(roster_df)} 条记录")
    
    return output_file


def create_simple_test_roster():
    """创建一个简单的测试rosterResult文件"""
    
    # 创建一些测试数据
    test_data = [
        # 机组1：有航班和地面任务
        {'crewId': 'Crew_10003', 'taskId': 'Flt_15787', 'isDDH': 0},
        {'crewId': 'Crew_10003', 'taskId': 'Flt_15783', 'isDDH': 0},
        {'crewId': 'Crew_10003', 'taskId': 'Flt_16113', 'isDDH': 0},
        
        # 机组2：有地面任务
        {'crewId': 'Crew_10002', 'taskId': 'Grd_2124_10000', 'isDDH': 0},
        {'crewId': 'Crew_10002', 'taskId': 'Grd_2124_10001', 'isDDH': 0},
        
        # 机组3：有航班和大巴置位
        {'crewId': 'Crew_10004', 'taskId': 'Flt_15399', 'isDDH': 0},
        {'crewId': 'Crew_10004', 'taskId': 'Bus_12345', 'isDDH': 1},  # 置位任务
        {'crewId': 'Crew_10004', 'taskId': 'Flt_15402', 'isDDH': 0},
        
        # 机组4：更多航班
        {'crewId': 'Crew_10005', 'taskId': 'Flt_15403', 'isDDH': 0},
        {'crewId': 'Crew_10005', 'taskId': 'Flt_15404', 'isDDH': 0},
    ]
    
    df = pd.DataFrame(test_data)
    output_file = "simple_test_roster.csv"
    df.to_csv(output_file, index=False)
    
    print(f"创建简单测试文件: {output_file}")
    print(f"包含 {len(df)} 条记录，{df['crewId'].nunique()} 个机组")
    
    return output_file


def main():
    """主函数"""
    print("=== 创建测试用的rosterResult文件 ===")
    
    # 检查是否有ALNS结果文件可以转换
    alns_file = "output/alns_result_20250710_152029.csv"
    
    if os.path.exists(alns_file):
        print(f"找到ALNS结果文件: {alns_file}")
        converted_file = convert_alns_result_to_roster_format(alns_file)
        print(f"✅ 转换完成: {converted_file}")
    else:
        print(f"未找到ALNS结果文件: {alns_file}")
    
    # 创建简单测试文件
    simple_file = create_simple_test_roster()
    print(f"✅ 创建简单测试文件: {simple_file}")
    
    print("\n现在可以使用以下文件测试CSV加载器:")
    if os.path.exists("test_rosterResult.csv"):
        print("  - test_rosterResult.csv (从ALNS结果转换)")
    print("  - simple_test_roster.csv (简单测试数据)")
    
    print("\n测试命令:")
    print("  python csv_initial_solution_loader.py test_rosterResult.csv")
    print("  python csv_initial_solution_loader.py simple_test_roster.csv --report")


if __name__ == "__main__":
    main()
