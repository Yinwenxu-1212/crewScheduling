#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一配置文件
解决主问题与子问题成本计算不一致的问题
"""

class UnifiedConfig:
    """
    统一的配置类，确保主问题和子问题使用相同的参数
    """
    
    # === 核心成本参数 ===
    # 这些参数必须在主问题和子问题中保持完全一致
    FLIGHT_TIME_REWARD = 100        # 飞行时间奖励（正值，在目标函数中用减号表示奖励）
    POSITIONING_PENALTY = 0.5       # 占位惩罚（统一使用子问题的值）
    AWAY_OVERNIGHT_PENALTY = 0.5    # 外站过夜惩罚（统一使用子问题的值）
    NEW_LAYOVER_PENALTY = 10        # 新停留站点惩罚
    UNCOVERED_FLIGHT_PENALTY = 500  # 未覆盖航班惩罚（主要覆盖率驱动因子）
    UNCOVERED_GROUND_DUTY_PENALTY = 10000  # 未覆盖占位任务惩罚（适中值，确保有覆盖动力但不过度影响目标函数）
    VIOLATION_PENALTY = 10          # 违规惩罚
    
    # === 评分系统参数 ===
    # 用于最终评价的竞赛标准参数
    FLY_TIME_MULTIPLIER = 1000      # 竞赛评分：值勤日日均飞时 * 1000
    UNCOVERED_FLIGHT_SCORE_PENALTY = -5     # 竞赛评分：未覆盖航班 * (-5)
    NEW_LAYOVER_SCORE_PENALTY = -10         # 竞赛评分：新增过夜站点 * (-10)
    AWAY_OVERNIGHT_SCORE_PENALTY = -0.5     # 竞赛评分：外站过夜天数 * (-0.5)
    POSITIONING_SCORE_PENALTY = -0.5        # 竞赛评分：置位次数 * (-0.5)
    VIOLATION_SCORE_PENALTY = -10           # 竞赛评分：违规次数 * (-10)
    
    # === 约束参数 ===
    MAX_DUTY_DAY_HOURS = 12.0
    MAX_FLIGHT_TIME_IN_DUTY_HOURS = 8.0
    MIN_REST_HOURS = 12.0
    MAX_FLIGHTS_IN_DUTY = 4
    MAX_TASKS_IN_DUTY = 6
    
    # === 连接时间参数（根据竞赛规则）===
    MIN_CONNECTION_TIME_FLIGHT_SAME_AIRCRAFT_MINUTES = 30  # 同一飞机最小间隔30分钟（实际可能更短）
    MIN_CONNECTION_TIME_FLIGHT_DIFFERENT_AIRCRAFT_HOURS = 3  # 不同飞机最小间隔3小时
    MIN_CONNECTION_TIME_BUS_HOURS = 2  # 大巴置位与飞行任务最小间隔2小时
    
    # === 算法参数 ===
    MAX_SUBPROBLEM_ITERATIONS = 1000  # 子问题求解最大迭代次数
    BEAM_WIDTH = 10
    
    # === 主程序运行参数 ===
    TIME_LIMIT_SECONDS = 1 * 3600 + 55 * 60  # 1小时55分钟
    DATA_PATH = 'data/'
    MAX_COLUMN_GENERATION_ITERATIONS = 3
    
    @classmethod
    def get_optimization_params(cls):
        """
        获取用于列生成优化的参数（主问题和子问题共用）
        """
        return {
            'flight_time_reward': cls.FLIGHT_TIME_REWARD,
            'positioning_penalty': cls.POSITIONING_PENALTY,
            'away_overnight_penalty': cls.AWAY_OVERNIGHT_PENALTY,
            'new_layover_penalty': cls.NEW_LAYOVER_PENALTY,
            'uncovered_flight_penalty': cls.UNCOVERED_FLIGHT_PENALTY,
            'uncovered_ground_duty_penalty': cls.UNCOVERED_GROUND_DUTY_PENALTY,
            'violation_penalty': cls.VIOLATION_PENALTY
        }
    
    @classmethod
    def get_scoring_params(cls):
        """
        获取用于最终评分的竞赛标准参数
        """
        return {
            'fly_time_multiplier': cls.FLY_TIME_MULTIPLIER,
            'uncovered_flight_penalty': cls.UNCOVERED_FLIGHT_SCORE_PENALTY,
            'new_layover_penalty': cls.NEW_LAYOVER_SCORE_PENALTY,
            'away_overnight_penalty': cls.AWAY_OVERNIGHT_SCORE_PENALTY,
            'positioning_penalty': cls.POSITIONING_SCORE_PENALTY,
            'violation_penalty': cls.VIOLATION_SCORE_PENALTY
        }
    
    @classmethod
    def get_constraint_params(cls):
        """
        获取约束参数
        """
        return {
            'max_duty_day_hours': cls.MAX_DUTY_DAY_HOURS,
            'max_flight_time_in_duty_hours': cls.MAX_FLIGHT_TIME_IN_DUTY_HOURS,
            'min_rest_hours': cls.MIN_REST_HOURS,
            'max_flights_in_duty': cls.MAX_FLIGHTS_IN_DUTY,
            'max_tasks_in_duty': cls.MAX_TASKS_IN_DUTY
        }

# 全局配置实例
config = UnifiedConfig()

# 向后兼容的常量定义
REWARD_PER_FLIGHT_HOUR = -config.FLIGHT_TIME_REWARD  # 子问题中使用负值表示奖励（减少成本）
PENALTY_PER_AWAY_OVERNIGHT = config.AWAY_OVERNIGHT_PENALTY
PENALTY_PER_POSITIONING = config.POSITIONING_PENALTY