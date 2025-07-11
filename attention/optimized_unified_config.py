# optimized_unified_config.py
"""
优化的统一配置文件
调整奖惩机制，鼓励智能置位以提高航班覆盖率
"""

class OptimizedUnifiedConfig:
    """
    优化的配置类，平衡置位与航班执行
    """
    
    # === 核心成本参数（调整以提高覆盖率）===
    FLIGHT_TIME_REWARD = 100        # 大幅提高飞行时间奖励（原50）
    POSITIONING_PENALTY = 1.0       # 大幅降低置位惩罚（原5.0）
    PRODUCTIVE_POSITIONING_REWARD = 20  # 新增：有效置位奖励
    AWAY_OVERNIGHT_PENALTY = 0.3    # 稍微降低外站过夜惩罚（原0.5）
    NEW_LAYOVER_PENALTY = 5         # 降低新停留站点惩罚（原10）
    UNCOVERED_FLIGHT_PENALTY = 500  # 大幅提高未覆盖航班惩罚（原200）
    UNCOVERED_GROUND_DUTY_PENALTY = 500  # 降低占位任务惩罚（原1000）
    VIOLATION_PENALTY = 50          # 提高违规惩罚（原10）
    
    # === 新增：置位策略参数 ===
    POSITIONING_CHAIN_BONUS = 30    # 置位后能执行多个航班的奖励
    RETURN_TO_BASE_BONUS = 10      # 返回基地的置位奖励
    STRATEGIC_POSITIONING_BONUS = 15 # 战略性置位奖励（到高需求机场）
    
    # === 评分系统参数 ===
    FLY_TIME_MULTIPLIER = 1000
    UNCOVERED_FLIGHT_SCORE_PENALTY = -5
    NEW_LAYOVER_SCORE_PENALTY = -10
    AWAY_OVERNIGHT_SCORE_PENALTY = -0.5
    POSITIONING_SCORE_PENALTY = -0.5
    VIOLATION_SCORE_PENALTY = -10
    
    # === 约束参数 ===
    MAX_DUTY_DAY_HOURS = 12.0
    MAX_FLIGHT_TIME_IN_DUTY_HOURS = 8.0
    MIN_REST_HOURS = 12.0
    MAX_FLIGHTS_IN_DUTY = 4
    MAX_TASKS_IN_DUTY = 6
    
    # === 飞行周期约束参数 ===
    MIN_CYCLE_REST_DAYS = 2  # 飞行周期开始前需要完整休息的日历日数
    MAX_CYCLE_HOURS = 168    # 最大飞行周期时间（7天）
    MIN_CYCLE_HOURS = 72     # 最小飞行周期时间（3天）
    CYCLE_END_FLIGHT_BONUS = 50    # 有效结束飞行周期的奖励
    CYCLE_RETURN_BASE_BONUS = 30   # 直接回基地的额外奖励
    CYCLE_POSITIONING_BONUS = 15   # 可通过置位回基地的奖励
    
    # === 连接时间参数 ===
    MIN_CONNECTION_TIME_FLIGHT_SAME_AIRCRAFT_MINUTES = 0
    MIN_CONNECTION_TIME_FLIGHT_DIFFERENT_AIRCRAFT_HOURS = 3
    MIN_CONNECTION_TIME_BUS_HOURS = 2
    
    # === 算法参数（优化）===
    MAX_SUBPROBLEM_ITERATIONS = 3000  # 增加迭代次数
    BEAM_WIDTH = 30  # 增加beam宽度
    MAX_CREWS_PER_FLIGHT = 8  # 增加每个航班的机组候选数
    
    # === 搜索优化参数 ===
    MAX_VISITED_STATES = 500000  # 增加最大访问状态数
    EARLY_STOPPING_PATIENCE = 20  # 提前停止耐心值
    
    # === 新增：动态调整参数 ===
    COVERAGE_THRESHOLD = 0.8  # 覆盖率阈值
    DYNAMIC_PENALTY_FACTOR = 2.0  # 低覆盖率时的惩罚倍数
    
    @classmethod
    def get_dynamic_positioning_penalty(cls, current_coverage_rate, positioning_info):
        """
        动态计算置位惩罚/奖励
        
        Args:
            current_coverage_rate: 当前航班覆盖率
            positioning_info: 置位信息字典，包含：
                - leads_to_flights: 置位后可执行的航班数
                - returns_to_base: 是否返回基地
                - to_high_demand_airport: 是否到高需求机场
        """
        base_penalty = cls.POSITIONING_PENALTY
        
        # 如果覆盖率低于阈值，降低置位惩罚
        if current_coverage_rate < cls.COVERAGE_THRESHOLD:
            base_penalty *= 0.5
            
        # 计算置位价值
        positioning_value = 0
        
        # 如果置位后能执行航班，给予奖励
        if positioning_info.get('leads_to_flights', 0) > 0:
            positioning_value += cls.PRODUCTIVE_POSITIONING_REWARD
            positioning_value += positioning_info['leads_to_flights'] * cls.POSITIONING_CHAIN_BONUS
            
        # 返回基地奖励
        if positioning_info.get('returns_to_base', False):
            positioning_value += cls.RETURN_TO_BASE_BONUS
            
        # 到高需求机场奖励
        if positioning_info.get('to_high_demand_airport', False):
            positioning_value += cls.STRATEGIC_POSITIONING_BONUS
            
        # 最终成本 = 基础惩罚 - 置位价值
        return max(0, base_penalty - positioning_value)
    
    @classmethod
    def get_optimization_params(cls):
        """获取优化参数"""
        return {
            'flight_time_reward': cls.FLIGHT_TIME_REWARD,
            'positioning_penalty': cls.POSITIONING_PENALTY,
            'productive_positioning_reward': cls.PRODUCTIVE_POSITIONING_REWARD,
            'away_overnight_penalty': cls.AWAY_OVERNIGHT_PENALTY,
            'uncovered_flight_penalty': cls.UNCOVERED_FLIGHT_PENALTY,
            'uncovered_ground_duty_penalty': cls.UNCOVERED_GROUND_DUTY_PENALTY,
            'violation_penalty': cls.VIOLATION_PENALTY,
            'positioning_chain_bonus': cls.POSITIONING_CHAIN_BONUS,
            'return_to_base_bonus': cls.RETURN_TO_BASE_BONUS,
            'strategic_positioning_bonus': cls.STRATEGIC_POSITIONING_BONUS
        }
    
    @classmethod
    def get_constraint_params(cls):
        """获取约束参数"""
        return {
            'max_duty_day_hours': cls.MAX_DUTY_DAY_HOURS,
            'max_flight_time_in_duty_hours': cls.MAX_FLIGHT_TIME_IN_DUTY_HOURS,
            'min_rest_hours': cls.MIN_REST_HOURS,
            'max_flights_in_duty': cls.MAX_FLIGHTS_IN_DUTY,
            'max_tasks_in_duty': cls.MAX_TASKS_IN_DUTY,
            'min_cycle_rest_days': cls.MIN_CYCLE_REST_DAYS,
            'max_cycle_hours': cls.MAX_CYCLE_HOURS,
            'min_cycle_hours': cls.MIN_CYCLE_HOURS
        }
        
    @classmethod
    def get_flight_cycle_params(cls):
        """获取飞行周期参数"""
        return {
            'min_cycle_rest_days': cls.MIN_CYCLE_REST_DAYS,
            'max_cycle_hours': cls.MAX_CYCLE_HOURS,
            'min_cycle_hours': cls.MIN_CYCLE_HOURS,
            'cycle_end_flight_bonus': cls.CYCLE_END_FLIGHT_BONUS,
            'cycle_return_base_bonus': cls.CYCLE_RETURN_BASE_BONUS,
            'cycle_positioning_bonus': cls.CYCLE_POSITIONING_BONUS
        }