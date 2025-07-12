#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
约束分析工具
专门用于分析和验证各种约束限制的实现情况
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional, Tuple
import logging
from data_models import Flight, Crew, GroundDuty, BusInfo, DutyDay, FlightDutyPeriod
from unified_config import UnifiedConfig
from constraint_checker import UnifiedConstraintChecker

class ConstraintAnalysisTool:
    """约束分析工具类"""
    
    def __init__(self):
        # 加载数据
        self.flights_df = pd.read_csv('data/flight.csv')
        self.crews_df = pd.read_csv('data/crew.csv')
        self.ground_duties_df = pd.read_csv('data/groundDuty.csv')
        self.layover_stations_df = pd.read_csv('data/layoverStation.csv')
        
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
    
    def analyze_constraint_definitions(self):
        """分析约束定义和当前实现"""
        print("=" * 80)
        print("约束定义分析报告")
        print("=" * 80)
        
        # 1. 基本定义分析
        self._analyze_basic_definitions()
        
        # 2. 约束参数分析
        self._analyze_constraint_parameters()
        
        # 3. 实现逻辑分析
        self._analyze_implementation_logic()
        
        # 4. 潜在问题识别
        self._identify_potential_issues()
    
    def _analyze_basic_definitions(self):
        """分析基本定义"""
        print("\n1. 基本定义分析")
        print("-" * 50)
        
        print("值勤日定义:")
        print("  - 机组人员一次出勤需要完成的一连串值勤任务")
        print("  - 包含：飞行任务、置位任务、占位任务")
        print("  - 可以只包含占位任务")
        print("  - 跨度一般不超过24小时")
        
        print("\n飞行值勤日定义:")
        print("  - 机组人员一次出勤需要完成的一连串飞行或置位任务")
        print("  - 必须包含飞行任务")
        print("  - 只可从任一可过夜机场出发，到任一可过夜机场结束")
        
        print("\n飞行周期定义:")
        print("  - 由值勤日组成的周期（可能包含少于2个完整日历日的休息）")
        print("  - 必须包含飞行值勤日")
        print("  - 飞行周期末尾一定为飞行值勤日")
        print("  - 最多横跨4个日历日")
        print("  - 开始前必须连续休息2个完整的日历日")
        
        print(f"\n可过夜机场数量: {len(self.layover_stations_set)}")
        layover_list = sorted(list(self.layover_stations_set))
        print(f"可过夜机场列表: {layover_list[:10]}{'...' if len(layover_list) > 10 else ''}")
    
    def _analyze_constraint_parameters(self):
        """分析约束参数"""
        print("\n2. 约束参数分析")
        print("-" * 50)
        
        # 从统一配置获取参数
        config = UnifiedConfig()
        
        print("当前约束参数设置:")
        print(f"  - 飞行值勤日内任务最小连接时间限制:")
        print(f"    * 同一飞机: {config.MIN_CONNECTION_TIME_FLIGHT_SAME_AIRCRAFT_MINUTES} 分钟")
        print(f"    * 不同飞机: {config.MIN_CONNECTION_TIME_FLIGHT_DIFFERENT_AIRCRAFT_HOURS} 小时")
        print(f"    * 大巴置位: {config.MIN_CONNECTION_TIME_BUS_HOURS} 小时")
        
        print(f"  - 飞行值勤日飞行任务数量限制: {config.MAX_FLIGHTS_IN_DUTY}")
        print(f"  - 飞行值勤日值勤任务数量限制: {config.MAX_TASKS_IN_DUTY}")
        print(f"  - 飞行值勤日最大飞行时间限制: {config.MAX_FLIGHT_TIME_IN_DUTY_HOURS} 小时")
        print(f"  - 飞行值勤日最小休息时间限制: {config.MIN_REST_HOURS} 小时")
        print(f"  - 飞行值勤日最大飞行值勤时间限制: {config.MAX_DUTY_DAY_HOURS} 小时")
        print(f"  - 飞行周期要求: 最多 {self.constraint_checker.MAX_FLIGHT_CYCLE_DAYS} 个日历日")
        print(f"  - 总飞行值勤时间限制: {self.constraint_checker.MAX_TOTAL_FLIGHT_HOURS} 小时")
        
        # 分析用户提到的数值
        print("\n用户提到的约束违规统计:")
        print("  - 飞行值勤日内任务最小连接时间限制：1 (违规)")
        print("  - 飞行值勤日飞行任务数量限制：0 (无违规)")
        print("  - 飞行值勤日值勤任务数量限制：0 (无违规)")
        print("  - 飞行值勤日最大飞行时间限制：1 (违规)")
        print("  - 飞行值勤日最小休息时间限制：38 (违规)")
        print("  - 飞行值勤日最大飞行值勤时间限制：92 (违规)")
        print("  - 飞行周期要求：112 (违规)")
        print("  - 总飞行值勤时间限制：3 (违规)")
    
    def _analyze_implementation_logic(self):
        """分析实现逻辑"""
        print("\n3. 实现逻辑分析")
        print("-" * 50)
        
        print("值勤日组织逻辑:")
        print("  - 按时间排序任务")
        print("  - 休息时间超过12小时 -> 开始新值勤日")
        print("  - 当前值勤日超过24小时 -> 开始新值勤日")
        print("  - 跨越超过2个日历日 -> 开始新值勤日")
        
        print("\n飞行值勤日判断逻辑:")
        print("  - 包含飞行任务")
        print("  - 从可过夜机场开始")
        print("  - 到可过夜机场结束")
        
        print("\n飞行周期检查逻辑:")
        print("  - 必须包含飞行值勤日")
        print("  - 末尾必须是飞行值勤日")
        print("  - 最多横跨4个日历日")
        print("  - 开始前必须休息2个完整日历日")
        
        print("\n连接时间检查逻辑:")
        print("  - 同一飞机: 0分钟最小间隔")
        print("  - 不同飞机: 3小时最小间隔")
        print("  - 大巴置位: 2小时最小间隔")
        print("  - 占位任务: 只需时间顺序正确")
    
    def _identify_potential_issues(self):
        """识别潜在问题"""
        print("\n4. 潜在问题识别")
        print("-" * 50)
        
        issues = []
        
        # 检查约束参数的合理性
        config = UnifiedConfig()
        
        # 1. 连接时间设置问题
        if config.MIN_CONNECTION_TIME_FLIGHT_SAME_AIRCRAFT_MINUTES == 0:
            issues.append("同一飞机最小连接时间为0分钟，可能导致时间冲突")
        
        # 2. 总飞行值勤时间定义问题
        if self.constraint_checker.MAX_TOTAL_FLIGHT_HOURS == 60:
            issues.append("总飞行值勤时间限制为60小时，但用户报告有3个违规，可能定义不清")
        
        # 3. 飞行值勤日定义问题
        issues.append("飞行值勤日必须从可过夜机场开始/结束，但置位任务可能打破这个限制")
        
        # 4. 值勤日跨度问题
        issues.append("值勤日跨度限制为24小时，但用户报告92个值勤时间违规")
        
        # 5. 休息时间计算问题
        issues.append("最小休息时间12小时，但用户报告38个休息时间违规")
        
        # 6. 飞行周期计算问题
        issues.append("飞行周期最多4个日历日，但用户报告112个周期违规")
        
        if issues:
            print("发现的潜在问题:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("未发现明显的潜在问题")
    
    def analyze_specific_crew_constraints(self, crew_id: str):
        """分析特定机组的约束情况"""
        print(f"\n=" * 80)
        print(f"机组 {crew_id} 约束详细分析")
        print(f"=" * 80)
        
        # 获取机组信息
        crew_info = self.crews_df[self.crews_df['crewId'] == crew_id]
        if crew_info.empty:
            print(f"未找到机组 {crew_id}")
            return
        
        crew_base = crew_info.iloc[0]['base']
        print(f"机组基地: {crew_base}")
        
        # 这里可以添加更详细的机组约束分析
        # 由于没有具体的排班数据，我们先分析理论约束
        
        print("\n理论约束分析:")
        print(f"  - 该机组应从基地 {crew_base} 开始和结束飞行周期")
        print(f"  - 每个值勤日最多 {UnifiedConfig.MAX_TASKS_IN_DUTY} 个任务")
        print(f"  - 每个值勤日最多 {UnifiedConfig.MAX_FLIGHTS_IN_DUTY} 个飞行任务")
        print(f"  - 每个值勤日最多 {UnifiedConfig.MAX_DUTY_DAY_HOURS} 小时")
        print(f"  - 值勤日间最少休息 {UnifiedConfig.MIN_REST_HOURS} 小时")
    
    def generate_constraint_improvement_suggestions(self):
        """生成约束改进建议"""
        print("\n=" * 80)
        print("约束改进建议")
        print("=" * 80)
        
        suggestions = [
            {
                'category': '定义澄清',
                'items': [
                    '明确"总飞行值勤时间"的计算方法：是飞行时间总和还是飞行值勤日时长总和',
                    '澄清飞行值勤日的开始/结束条件：置位任务是否影响可过夜机场要求',
                    '明确值勤日与飞行值勤日的区别和转换条件'
                ]
            },
            {
                'category': '参数优化',
                'items': [
                    '重新评估同一飞机0分钟连接时间的合理性',
                    '调整总飞行值勤时间限制以匹配实际需求',
                    '优化休息时间计算逻辑，考虑跨时区和实际操作需求'
                ]
            },
            {
                'category': '实现改进',
                'items': [
                    '增强值勤日组织算法，更好地处理边界情况',
                    '改进飞行周期检测逻辑，正确处理置位任务',
                    '优化约束检查的性能和准确性'
                ]
            },
            {
                'category': '验证增强',
                'items': [
                    '添加更详细的约束违规报告',
                    '实现约束检查的单元测试',
                    '增加约束参数的敏感性分析'
                ]
            }
        ]
        
        for suggestion in suggestions:
            print(f"\n{suggestion['category']}:")
            for i, item in enumerate(suggestion['items'], 1):
                print(f"  {i}. {item}")

def main():
    """主函数"""
    tool = ConstraintAnalysisTool()
    
    # 执行约束分析
    tool.analyze_constraint_definitions()
    
    # 分析特定机组（如果需要）
    # tool.analyze_specific_crew_constraints('Crew_10002')
    
    # 生成改进建议
    tool.generate_constraint_improvement_suggestions()

if __name__ == '__main__':
    main()