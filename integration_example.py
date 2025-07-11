#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
集成示例：展示如何在现有代码中使用CSV初始解加载器
"""

import os
from csv_initial_solution_loader import load_initial_solution_from_csv, CSVInitialSolutionLoader
from data_loader import load_all_data


def example_1_load_external_solution():
    """示例1：从外部优化器加载初始解"""
    print("=== 示例1：从外部优化器加载初始解 ===")
    
    csv_file = "test_rosterResult.csv"  # 假设这是外部优化器的结果
    
    if not os.path.exists(csv_file):
        print(f"错误：文件 {csv_file} 不存在")
        return None
    
    # 加载初始解
    rosters, new_layover_stations, stats = load_initial_solution_from_csv(
        csv_file,
        calculate_costs=True,
        save_new_layovers=False  # 暂时不保存，先分析
    )
    
    print(f"✅ 成功加载 {len(rosters)} 个排班方案")
    print(f"✅ 发现 {stats['added_layover_count']} 个新的过夜机场")
    
    # 如果有新的过夜机场，询问是否更新配置
    if stats['added_layover_count'] > 0:
        print(f"新增过夜机场: {sorted(list(stats['new_layover_stations']))}")
        
        # 在实际应用中，这里可以添加用户确认逻辑
        # update_config = input("是否更新过夜机场配置？(y/n): ")
        # if update_config.lower() == 'y':
        #     loader = CSVInitialSolutionLoader()
        #     loader.save_new_layover_stations(new_layover_stations, 'data/layoverStation.csv')
        #     print("✅ 过夜机场配置已更新")
    
    return rosters, new_layover_stations, stats


def example_2_compare_solutions():
    """示例2：比较不同排班方案的过夜机场使用情况"""
    print("\n=== 示例2：比较不同排班方案 ===")
    
    solutions = [
        ("简单测试方案", "simple_test_roster.csv"),
        ("完整测试方案", "test_rosterResult.csv")
    ]
    
    comparison_results = []
    
    for name, csv_file in solutions:
        if not os.path.exists(csv_file):
            print(f"跳过 {name}：文件 {csv_file} 不存在")
            continue
        
        try:
            _, _, stats = load_initial_solution_from_csv(
                csv_file,
                calculate_costs=False,  # 为了速度，跳过成本计算
                save_new_layovers=False
            )
            
            comparison_results.append({
                'name': name,
                'rosters': len(stats['crew_duty_day_analysis']),
                'original_layovers': stats['original_layover_count'],
                'new_layovers': stats['added_layover_count'],
                'total_layovers': stats['new_layover_count'],
                'top_airports': list(sorted(stats['overnight_frequency'].items(), 
                                          key=lambda x: x[1], reverse=True)[:5])
            })
            
        except Exception as e:
            print(f"处理 {name} 时出错: {e}")
    
    # 显示比较结果
    print("\n排班方案比较:")
    print(f"{'方案名称':<15} {'机组数':<8} {'原始过夜':<10} {'新增过夜':<10} {'总过夜':<8} {'主要过夜机场'}")
    print("-" * 80)
    
    for result in comparison_results:
        top_airports_str = ", ".join([f"{airport}({count})" for airport, count in result['top_airports'][:3]])
        print(f"{result['name']:<15} {result['rosters']:<8} {result['original_layovers']:<10} "
              f"{result['new_layovers']:<10} {result['total_layovers']:<8} {top_airports_str}")


def example_3_integrate_with_alns():
    """示例3：与ALNS算法集成"""
    print("\n=== 示例3：与ALNS算法集成示例 ===")
    
    csv_file = "test_rosterResult.csv"
    
    if not os.path.exists(csv_file):
        print(f"错误：文件 {csv_file} 不存在")
        return
    
    try:
        # 1. 加载基础数据
        print("1. 加载基础数据...")
        all_data = load_all_data('./data/')
        
        # 2. 从CSV加载初始解
        print("2. 从CSV加载初始解...")
        rosters, new_layover_stations, stats = load_initial_solution_from_csv(
            csv_file,
            calculate_costs=True,
            save_new_layovers=False
        )
        
        # 3. 更新layover_stations（如果需要）
        if stats['added_layover_count'] > 0:
            print(f"3. 发现 {stats['added_layover_count']} 个新过夜机场，更新配置...")
            all_data['layover_stations'] = new_layover_stations
        else:
            print("3. 未发现新过夜机场，使用原始配置")
        
        # 4. 创建ALNS初始解对象（伪代码）
        print("4. 准备ALNS算法...")
        print(f"   - 初始排班方案: {len(rosters)} 个")
        print(f"   - 过夜机场: {len(new_layover_stations)} 个")
        print(f"   - 航班: {len(all_data['flights'])} 个")
        print(f"   - 机组: {len(all_data['crews'])} 个")
        
        # 在实际应用中，这里会创建ALNSSolution对象并运行算法
        # initial_solution = ALNSSolution(rosters, all_data['flights'], 
        #                                all_data['ground_duties'], all_data['crews'])
        # alns = ALNSAlgorithm(...)
        # best_solution = alns.solve(initial_solution)
        
        print("✅ 集成准备完成，可以开始ALNS优化")
        
    except Exception as e:
        print(f"集成过程中出错: {e}")
        import traceback
        traceback.print_exc()


def example_4_batch_analysis():
    """示例4：批量分析多个排班方案"""
    print("\n=== 示例4：批量分析 ===")
    
    # 查找所有CSV文件
    csv_files = []
    for file in os.listdir('.'):
        if file.endswith('.csv') and 'roster' in file.lower():
            csv_files.append(file)
    
    if not csv_files:
        print("未找到排班CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个排班文件:")
    
    batch_results = []
    
    for csv_file in csv_files:
        try:
            print(f"  分析 {csv_file}...")
            _, _, stats = load_initial_solution_from_csv(
                csv_file,
                calculate_costs=False,
                save_new_layovers=False
            )
            
            batch_results.append({
                'file': csv_file,
                'crews': len(stats['crew_duty_day_analysis']),
                'new_layovers': stats['added_layover_count'],
                'efficiency': stats['new_layover_count'] / len(stats['crew_duty_day_analysis']) if stats['crew_duty_day_analysis'] else 0
            })
            
        except Exception as e:
            print(f"    错误: {e}")
    
    # 按效率排序
    batch_results.sort(key=lambda x: x['efficiency'])
    
    print("\n批量分析结果 (按过夜机场效率排序):")
    print(f"{'文件名':<25} {'机组数':<8} {'新增过夜':<10} {'效率':<8}")
    print("-" * 55)
    
    for result in batch_results:
        print(f"{result['file']:<25} {result['crews']:<8} {result['new_layovers']:<10} {result['efficiency']:.3f}")


def main():
    """主函数：运行所有示例"""
    print("CSV初始解加载器集成示例")
    print("=" * 50)
    
    # 运行各个示例
    example_1_load_external_solution()
    example_2_compare_solutions()
    example_3_integrate_with_alns()
    example_4_batch_analysis()
    
    print("\n" + "=" * 50)
    print("所有示例运行完成！")
    print("\n使用提示:")
    print("1. 在实际项目中，可以将这些功能集成到主要的优化流程中")
    print("2. 建议在运行ALNS之前先分析和更新过夜机场配置")
    print("3. 可以使用批量分析功能比较不同算法的结果")
    print("4. 详细的分析报告可以帮助理解排班方案的特点")


if __name__ == "__main__":
    main()
