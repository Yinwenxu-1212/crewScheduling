#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试CSV初始解加载器的脚本
"""

import os
import sys
from csv_initial_solution_loader import load_initial_solution_from_csv, CSVInitialSolutionLoader


def test_csv_loader():
    """测试CSV加载器功能"""
    print("=== 测试CSV初始解加载器 ===")
    
    # 检查是否有可用的CSV文件
    test_files = [
        "output/alns_result_20250710_152029.csv",
        "rosterResult.csv",
        "output/initial_solution.csv"
    ]
    
    csv_file = None
    for file_path in test_files:
        if os.path.exists(file_path):
            csv_file = file_path
            break
    
    if not csv_file:
        print("错误: 未找到可用的CSV文件进行测试")
        print("请确保以下文件之一存在:")
        for file_path in test_files:
            print(f"  - {file_path}")
        return False
    
    print(f"使用测试文件: {csv_file}")
    
    try:
        # 测试加载功能
        rosters, new_layover_stations, stats = load_initial_solution_from_csv(
            csv_file, 
            data_path="./data/",
            calculate_costs=True,
            save_new_layovers=True
        )
        
        print("\n=== 测试结果 ===")
        print(f"✅ 成功加载 {len(rosters)} 个排班方案")
        print(f"✅ 分析了 {stats['new_layover_count']} 个过夜机场")
        
        if stats['added_layover_count'] > 0:
            print(f"✅ 发现 {stats['added_layover_count']} 个新的过夜机场")
        else:
            print("ℹ️  未发现新的过夜机场")
        
        # 显示一些示例数据
        if rosters:
            print(f"\n示例排班方案 (前3个):")
            for i, roster in enumerate(rosters[:3]):
                print(f"  {i+1}. 机组 {roster.crew_id}: {len(roster.duties)} 个任务, 成本 {roster.cost:.2f}")
        
        # 显示过夜机场统计
        if stats['overnight_frequency']:
            print(f"\n过夜机场使用频率 (前5个):")
            sorted_freq = sorted(stats['overnight_frequency'].items(), 
                               key=lambda x: x[1], reverse=True)[:5]
            for airport, freq in sorted_freq:
                marker = " (新增)" if airport in stats['new_layover_stations'] else ""
                print(f"  {airport}: {freq} 次{marker}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_individual_components():
    """测试各个组件"""
    print("\n=== 测试各个组件 ===")
    
    try:
        # 测试基础数据加载
        loader = CSVInitialSolutionLoader("./data/")
        all_data = loader.load_base_data()
        
        print(f"✅ 基础数据加载成功:")
        print(f"  - 航班: {len(all_data['flights'])} 个")
        print(f"  - 机组: {len(all_data['crews'])} 个")
        print(f"  - 地面任务: {len(all_data['ground_duties'])} 个")
        print(f"  - 大巴任务: {len(all_data['bus_info'])} 个")
        print(f"  - 原始过夜机场: {len(all_data['layover_stations'])} 个")
        
        # 显示一些原始过夜机场
        layover_list = sorted(list(all_data['layover_stations']))
        print(f"  - 原始过夜机场示例: {layover_list[:10]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 组件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("开始测试CSV初始解加载器...")
    
    # 测试各个组件
    component_test_passed = test_individual_components()
    
    # 测试完整功能
    full_test_passed = test_csv_loader()
    
    print("\n=== 测试总结 ===")
    if component_test_passed and full_test_passed:
        print("✅ 所有测试通过！")
        print("\n使用方法:")
        print("1. 作为模块导入:")
        print("   from csv_initial_solution_loader import load_initial_solution_from_csv")
        print("   rosters, layovers, stats = load_initial_solution_from_csv('your_file.csv')")
        print("\n2. 命令行使用:")
        print("   python csv_initial_solution_loader.py your_file.csv")
        print("   python csv_initial_solution_loader.py your_file.csv --report")
        return 0
    else:
        print("❌ 部分测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
