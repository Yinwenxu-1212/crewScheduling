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
    FLIGHT_TIME_REWARD = 1000   # 飞行时间奖励（降低，减少负成本问题）
    POSITIONING_PENALTY = 0.5      # 置位惩罚（提高，抑制过度置位）
    AWAY_OVERNIGHT_PENALTY = 0.5   # 外站过夜惩罚（保持不变）
    NEW_LAYOVER_PENALTY = 10       # 新停留站点惩罚
    UNCOVERED_FLIGHT_PENALTY = 5 # 未覆盖航班惩罚（大幅提高，强化航班覆盖优先级）
    UNCOVERED_GROUND_DUTY_PENALTY = 20  # 未覆盖占位任务惩罚（保持高优先级但低于航班）
    VIOLATION_PENALTY = 10         # 违规惩罚
    
    # === 评分系统参数 ===
    # 用于最终评价的竞赛标准参数
    FLY_TIME_MULTIPLIER = 1000    # 竞赛评分：值勤日日均飞时 * 1000
    UNCOVERED_FLIGHT_SCORE_PENALTY = -5     # 竞赛评分：未覆盖航班 * (-5)
    NEW_LAYOVER_SCORE_PENALTY = -10         # 竞赛评分：新增过夜站点 * (-10)
    AWAY_OVERNIGHT_SCORE_PENALTY = -0.5     # 竞赛评分：外站过夜天数 * (-0.5)
    POSITIONING_SCORE_PENALTY = -0.5        # 竞赛评分：置位次数 * (-0.5)
    VIOLATION_SCORE_PENALTY = -10           # 竞赛评分：违规次数 * (-10)
    
    # 任务基础分数参数
    #FLIGHT_BASE_SCORE =1000 # 航班任务基础分数（大幅提高以鼓励航班覆盖）
    #GROUND_DUTY_BASE_SCORE = 500 # 地面任务基础分数（保持高优先级）
    #BUS_TASK_BASE_SCORE = 10.0  # 大巴任务基础分数（进一步降低）
    #POSITIONING_BASE_SCORE = 5.0  # 置位任务基础分数（大幅降低）
    
    # === 约束参数 ===
    # 值勤日约束
    MAX_DUTY_DAY_HOURS = 12.0
    MAX_FLIGHT_TIME_IN_DUTY_HOURS = 8.0
    MIN_REST_HOURS = 12.0
    MAX_FLIGHTS_IN_DUTY = 4
    MAX_TASKS_IN_DUTY = 6
    
    # 飞行值勤日约束
    MAX_FDP_HOURS = 12.0  # 飞行值勤日最大12小时
    MAX_FDP_FLIGHTS = 4   # 飞行值勤日最大4个飞行任务
    MAX_FDP_TASKS = 6     # 飞行值勤日最大6个任务
    MAX_FDP_FLIGHT_TIME = 8.0  # 飞行值勤日最大飞行时间8小时
    
    # 飞行周期约束
    MAX_FLIGHT_CYCLE_DAYS = 4  # 最多横跨4个日历日
    MIN_CYCLE_REST_DAYS = 2    # 开始前需连续休息2个完整日历日
    MAX_TOTAL_FLIGHT_DUTY_HOURS = 60.0  # 总飞行值勤时间不超过60小时
    
    # 休息时间约束
    MIN_OVERNIGHT_HOURS = 12.0  # 最小过夜时间（规则7）
    
    # 工作休息模式约束
    MAX_CONSECUTIVE_DUTY_DAYS = 4  # 值四休二
    MIN_REST_AFTER_CONSECUTIVE_DUTY = 48  # 连续工作后需休息48小时
    
    # 置位任务约束
    MAX_POSITIONING_TASKS = 10  # 最大置位任务数量
    
    # 新飞行周期约束
    MIN_REST_DAYS_FOR_NEW_CYCLE = 2  # 开始新飞行周期前的最小休息天数
    
    # 总飞行时间约束
    MAX_TOTAL_FLIGHT_HOURS = 60.0  # 计划期内总飞行时间上限（小时）
    
    # === 连接时间参数（根据竞赛规则）===
    MIN_CONNECTION_TIME_FLIGHT_SAME_AIRCRAFT_MINUTES = 0  # 同一飞机最小间隔0分钟
    MIN_CONNECTION_TIME_FLIGHT_DIFFERENT_AIRCRAFT_HOURS = 3  # 不同飞机最小间隔3小时
    MIN_CONNECTION_TIME_BUS_HOURS = 2  # 大巴置位与飞行任务最小间隔2小时
    DEFAULT_MIN_CONNECTION_TIME_HOURS = 1  # 默认最小连接时间1小时
    
    # === 算法参数 ===
    MAX_SUBPROBLEM_ITERATIONS = 2500  # 子问题求解最大迭代次数（进一步增大搜索深度）
    BEAM_WIDTH = 25  # beam search宽度（进一步增大搜索范围）
    MAX_CREWS_PER_FLIGHT = 6  # 每个航班最多被分配给的机组数量
    
    # === 搜索优化参数 ===
    MAX_VISITED_STATES = 200000  # 最大访问状态数
    CLEANUP_INTERVAL = 2000  # 清理间隔
    CONVERGENCE_THRESHOLD = 1e-4  # 收敛阈值
    STAGNATION_LIMIT = 5  # 停滞限制
    MIN_ITERATIONS = 3  # 最小迭代次数
    
    # === 路径权重参数 ===
    FLIGHT_PATH_WEIGHT = 10  # 航班路径权重（大幅提高以优先选择航班）
    GROUND_DUTY_PATH_WEIGHT = 6.0  # 地面任务路径权重（保持高优先级）
    BUS_PATH_WEIGHT = 0.3  # 大巴任务路径权重（进一步降低）
    POSITIONING_PATH_WEIGHT = 0.1  # 置位任务路径权重（大幅降低）
    
    # === 主程序运行参数 ===
    TIME_LIMIT_SECONDS = 1 * 3600 + 55 * 60  # 1小时55分钟
    DATA_PATH = 'data/'
    MAX_COLUMN_GENERATION_ITERATIONS = 35  # 设置为35轮以匹配之前的良好结果
    
    @classmethod
    def get_planning_start_date(cls):
        """
        从flight.csv中读取最早的时间作为计划开始日期
        """
        import csv
        import os
        from datetime import datetime
        
        earliest_date = None
        
        # 检查 flight.csv
        try:
            flight_file = os.path.join(cls.DATA_PATH, 'flight.csv')
            if os.path.exists(flight_file):
                with open(flight_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # 解析 std (计划起飞时间) 字段
                        std_str = row['std']
                        flight_datetime = datetime.strptime(std_str, '%Y/%m/%d %H:%M')
                        flight_date = flight_datetime.date()
                        
                        if earliest_date is None or flight_date < earliest_date:
                            earliest_date = flight_date
        except Exception as e:
            print(f"读取航班数据失败: {e}")
        
        if earliest_date:
            return datetime.combine(earliest_date, datetime.min.time())
        else:
            # 如果没有找到任何数据，使用默认值
            print("未找到航班数据文件，使用默认计划开始日期")
            return datetime(2025, 5, 1, 0, 0, 0)
    
    @classmethod
    def get_planning_end_date(cls):
        """
        从flight.csv中读取最晚的时间作为计划结束日期
        """
        import csv
        import os
        from datetime import datetime, timedelta
        
        latest_date = None
        
        # 检查 flight.csv
        try:
            flight_file = os.path.join(cls.DATA_PATH, 'flight.csv')
            if os.path.exists(flight_file):
                with open(flight_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # 解析 sta (计划到达时间) 字段
                        sta_str = row['sta']
                        flight_datetime = datetime.strptime(sta_str, '%Y/%m/%d %H:%M')
                        flight_date = flight_datetime.date()
                        
                        if latest_date is None or flight_date > latest_date:
                            latest_date = flight_date
        except Exception as e:
            print(f"读取航班数据失败: {e}")
        
        if latest_date:
            # 结束日期设为最晚日期的23:59:59
            return datetime.combine(latest_date, datetime.max.time().replace(microsecond=0))
        else:
            # 如果没有找到任何数据，使用默认值
            print("未找到航班数据文件，使用默认计划结束日期")
            return datetime(2025, 5, 7, 23, 59, 59)
    
    # 动态获取计划开始和结束日期
    PLANNING_START_DATE = None  # 将在运行时通过 get_planning_start_date() 获取
    PLANNING_END_DATE = None    # 将在运行时通过 get_planning_end_date() 获取
    
    @classmethod
    def initialize_planning_dates(cls):
        """
        初始化计划开始和结束日期
        """
        if cls.PLANNING_START_DATE is None:
            cls.PLANNING_START_DATE = cls.get_planning_start_date()
        if cls.PLANNING_END_DATE is None:
            cls.PLANNING_END_DATE = cls.get_planning_end_date()
    
    # === 其他全局变量 ===
    GDS = None # ground_duties
    
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
            # 值勤日约束
            'max_duty_day_hours': cls.MAX_DUTY_DAY_HOURS,
            'max_flight_time_in_duty_hours': cls.MAX_FLIGHT_TIME_IN_DUTY_HOURS,
            'min_rest_hours': cls.MIN_REST_HOURS,
            'max_flights_in_duty': cls.MAX_FLIGHTS_IN_DUTY,
            'max_tasks_in_duty': cls.MAX_TASKS_IN_DUTY,
            
            # 飞行值勤日约束
            'max_fdp_hours': cls.MAX_FDP_HOURS,
            'max_fdp_flights': cls.MAX_FDP_FLIGHTS,
            'max_fdp_tasks': cls.MAX_FDP_TASKS,
            'max_fdp_flight_time': cls.MAX_FDP_FLIGHT_TIME,
            
            # 飞行周期约束
            'max_flight_cycle_days': cls.MAX_FLIGHT_CYCLE_DAYS,
            'min_cycle_rest_days': cls.MIN_CYCLE_REST_DAYS,
            'max_total_flight_duty_hours': cls.MAX_TOTAL_FLIGHT_DUTY_HOURS,
            
            # 休息时间约束
            'min_overnight_hours': cls.MIN_OVERNIGHT_HOURS,
            
            # 工作休息模式约束
            'max_consecutive_duty_days': cls.MAX_CONSECUTIVE_DUTY_DAYS,
            'min_rest_after_consecutive_duty': cls.MIN_REST_AFTER_CONSECUTIVE_DUTY,
            
            # 连接时间约束
            'min_connection_time_flight_same_aircraft_minutes': cls.MIN_CONNECTION_TIME_FLIGHT_SAME_AIRCRAFT_MINUTES,
            'min_connection_time_flight_different_aircraft_hours': cls.MIN_CONNECTION_TIME_FLIGHT_DIFFERENT_AIRCRAFT_HOURS,
            'min_connection_time_bus_hours': cls.MIN_CONNECTION_TIME_BUS_HOURS,
            'default_min_connection_time_hours': cls.DEFAULT_MIN_CONNECTION_TIME_HOURS,
            
            # 置位任务约束
            'max_positioning_tasks': cls.MAX_POSITIONING_TASKS,
            
            # 新飞行周期约束
            'min_rest_days_for_new_cycle': cls.MIN_REST_DAYS_FOR_NEW_CYCLE,
            
            # 总飞行时间约束
            'max_total_flight_hours': cls.MAX_TOTAL_FLIGHT_HOURS
        }

# 全局配置实例
config = UnifiedConfig()

# 向后兼容的常量定义
REWARD_PER_FLIGHT_HOUR = -config.FLIGHT_TIME_REWARD  # 子问题中使用负值表示奖励（减少成本）
PENALTY_PER_AWAY_OVERNIGHT = config.AWAY_OVERNIGHT_PENALTY
PENALTY_PER_POSITIONING = config.POSITIONING_PENALTY