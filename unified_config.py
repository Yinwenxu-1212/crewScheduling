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
    FLIGHT_TIME_REWARD = 50       # 飞行时间奖励（大幅提高，激励执行航班）
    POSITIONING_PENALTY = 5.0      # 置位惩罚（大幅提高，抑制过度置位）
    AWAY_OVERNIGHT_PENALTY = 0.5   # 外站过夜惩罚（保持不变）
    NEW_LAYOVER_PENALTY = 10       # 新停留站点惩罚
    UNCOVERED_FLIGHT_PENALTY = 200 # 未覆盖航班惩罚（提高，强化航班覆盖优先级）
    UNCOVERED_GROUND_DUTY_PENALTY = 1000  # 未覆盖占位任务惩罚（降低，平衡航班与占位任务优先级）
    VIOLATION_PENALTY = 10         # 违规惩罚
    
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
    MAX_SUBPROBLEM_ITERATIONS = 2000  # 子问题求解最大迭代次数（增大搜索深度）
    BEAM_WIDTH = 20  # beam search宽度（增大搜索范围）
    MAX_CREWS_PER_FLIGHT = 6  # 每个航班最多被分配给的机组数量
    
    # === 搜索优化参数 ===
    MAX_VISITED_STATES = 200000  # 最大访问状态数
    CLEANUP_INTERVAL = 2000  # 清理间隔
    CONVERGENCE_THRESHOLD = 1e-4  # 收敛阈值
    STAGNATION_LIMIT = 5  # 停滞限制
    MIN_ITERATIONS = 3  # 最小迭代次数
    
    # === 主程序运行参数 ===
    TIME_LIMIT_SECONDS = 1 * 3600 + 55 * 60  # 1小时55分钟
    DATA_PATH = 'data/'
    MAX_COLUMN_GENERATION_ITERATIONS = 3
    PLANNING_START_DATE = (2025, 4, 29)  # 计划开始日期 (年, 月, 日)
    
    # === 机场分类配置 ===
    # 动态机场分类配置 - 基于数据自动分析
    try:
        from dynamic_airport_analyzer import get_dynamic_airport_config
        
        # 获取动态分析的机场分类
        _airport_config = get_dynamic_airport_config()
        
        HUB_AIRPORTS = _airport_config.get('HUB_AIRPORTS', set())
        MAJOR_AIRPORTS = _airport_config.get('MAJOR_AIRPORTS', set())
        IMPORTANT_AIRPORTS = _airport_config.get('IMPORTANT_AIRPORTS', set())
        
        print(f"动态加载机场配置: 枢纽={len(HUB_AIRPORTS)}, 主要={len(MAJOR_AIRPORTS)}, 重要={len(IMPORTANT_AIRPORTS)}")
        
    except Exception as e:
        print(f"动态机场分析失败，使用默认配置: {e}")
        # 降级到基础配置
        HUB_AIRPORTS = {'VIOC'}
        MAJOR_AIRPORTS = {'RRES', 'RTHW'}
        IMPORTANT_AIRPORTS = {
            'VIOC', 'RRES', 'RTHW',  # 枢纽机场
            'ENDP', 'TATC', 'TPWY', 'VWSF', 'XVFW',  # 高频航班机场（200+航班）
            'JFEE', 'BTTC', 'GDHI', 'RTWL'  # 重要航班机场（130+航班）
        }
    
    # === 置位价值评估权重 ===
    POSITIONING_VALUE_WEIGHTS = {
        'base_importance': 0.3,      # 基础重要性权重
        'connection_value': 0.4,     # 连接价值权重（最重要）
        'time_urgency': 0.2,         # 时间紧迫性权重
        'coverage_need': 0.1         # 覆盖需求权重
    }
    
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