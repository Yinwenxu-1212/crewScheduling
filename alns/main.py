#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ALNS (Adaptive Large Neighborhood Search) 框架
用于机组排班优化问题

基于现有的约束检查器、覆盖验证器和初始解生成器
实现自适应大邻域搜索算法

Author: Crew Scheduling Team
Date: 2025-01-10
"""

import os
import sys
import time
import random
import math
import copy
from datetime import datetime, timedelta
from typing import List, Dict, Set, Tuple, Optional, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目模块
from constraint_checker import UnifiedConstraintChecker
from coverage_validator import CoverageValidator
from initial_solution_generator import generate_initial_rosters_with_heuristic
from data_loader import load_all_data
from data_models import Flight, Crew, Roster, GroundDuty, BusInfo
# from results_writer import write_results_to_csv  # 使用自定义的简化版本
from unified_config import UnifiedConfig
from scoring_system import ScoringSystem

# traceback
import traceback


class ALNSSolution:
    """ALNS解决方案类"""

    def __init__(self, rosters: List[Roster], flights: List[Flight],
                 ground_duties: List[GroundDuty], crews: List[Crew]):
        self.rosters = rosters
        self.flights = flights
        self.ground_duties = ground_duties
        self.crews = crews
        self.objective_value = None
        self.coverage_rate = None
        self.ground_duty_coverage_rate = None
        self.is_feasible = True
        self.violations = []

        # 计算目标函数值
        self._calculate_objective()

    def _calculate_objective(self):
        """计算目标函数值"""
        # 使用与main.py相同的线性目标函数
        total_flight_hours = 0.0
        total_duty_days = 0.0
        covered_flights = set()
        covered_ground_duties = set()

        roster_cost_sum = 0
        for roster in self.rosters:
            # 计算roster成本（使用统一配置的参数）
            flight_reward = 0
            positioning_penalty = 0
            overnight_penalty = 0

            for duty in roster.duties:
                if isinstance(duty, Flight):
                    covered_flights.add(duty.id)
                    if hasattr(duty, 'flyTime') and duty.flyTime:
                        flight_time_hours = duty.flyTime / 60.0
                        flight_reward += flight_time_hours * UnifiedConfig.FLIGHT_TIME_REWARD
                        total_flight_hours += flight_time_hours
                    total_duty_days += 1
                elif hasattr(duty, 'crewId') and hasattr(duty, 'airport'):
                    covered_ground_duties.add(duty.id)
                    total_duty_days += 1

            # 简化的成本计算（主要组成部分）
            roster_cost = flight_reward - positioning_penalty - overnight_penalty
            roster_cost_sum += roster_cost

        # 计算未覆盖惩罚
        uncovered_flights = len(self.flights) - len(covered_flights)
        uncovered_ground_duties = len(self.ground_duties) - len(covered_ground_duties)

        uncovered_flight_penalty = uncovered_flights * UnifiedConfig.UNCOVERED_FLIGHT_PENALTY
        uncovered_ground_duty_penalty = uncovered_ground_duties * UnifiedConfig.UNCOVERED_GROUND_DUTY_PENALTY

        # 总目标函数值（最小化）
        self.objective_value = roster_cost_sum + uncovered_flight_penalty + uncovered_ground_duty_penalty

        # 计算覆盖率
        self.coverage_rate = len(covered_flights) / len(self.flights) if self.flights else 0.0
        self.ground_duty_coverage_rate = len(covered_ground_duties) / len(self.ground_duties) if self.ground_duties else 0.0

    def copy(self):
        """创建解的深拷贝"""
        new_rosters = [copy.deepcopy(roster) for roster in self.rosters]
        return ALNSSolution(new_rosters, self.flights, self.ground_duties, self.crews)

    def is_better_than(self, other_solution):
        """判断当前解是否优于另一个解"""
        if not self.is_feasible and other_solution.is_feasible:
            return False
        if self.is_feasible and not other_solution.is_feasible:
            return True
        return self.objective_value < other_solution.objective_value

    def __str__(self):
        return (f"ALNSSolution(obj={self.objective_value:.2f}, "
                f"coverage={self.coverage_rate:.2%}, "
                f"rosters={len(self.rosters)}, "
                f"feasible={self.is_feasible})")


class DestroyOperator:
    """破坏算子基类"""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self.usage_count = 0
        self.success_count = 0

    def destroy(self, solution: ALNSSolution, destroy_size: int) -> Tuple[ALNSSolution, List[Any]]:
        """
        破坏解决方案

        Args:
            solution: 当前解决方案
            destroy_size: 破坏的大小（移除的元素数量）

        Returns:
            Tuple[破坏后的解决方案, 被移除的元素列表]
        """
        raise NotImplementedError

    def update_weight(self, success: bool):
        """更新算子权重"""
        self.usage_count += 1
        if success:
            self.success_count += 1


class RepairOperator:
    """修复算子基类"""

    def __init__(self, name: str, weight: float = 1.0, scoring_system: ScoringSystem = None):
        self.name = name
        self.weight = weight
        self.usage_count = 0
        self.success_count = 0
        self.scoring_system = scoring_system
        if scoring_system is None:
            raise ValueError("3. Scoring system must be provided")

    def repair(self, solution: ALNSSolution, removed_elements: List[Any]) -> ALNSSolution:
        """
        修复解决方案

        Args:
            solution: 被破坏的解决方案
            removed_elements: 被移除的元素列表

        Returns:
            修复后的解决方案
        """
        raise NotImplementedError

    def update_weight(self, success: bool):
        """更新算子权重"""
        self.usage_count += 1
        if success:
            self.success_count += 1


class RandomRosterDestroy(DestroyOperator):
    """随机移除roster的破坏算子"""

    def __init__(self):
        super().__init__("RandomRosterDestroy")

    def destroy(self, solution: ALNSSolution, destroy_size: int) -> Tuple[ALNSSolution, List[Roster]]:
        """随机移除指定数量的roster"""
        new_solution = solution.copy()

        if len(new_solution.rosters) <= destroy_size:
            # 如果要移除的数量大于等于总数，保留一个roster
            destroy_size = max(1, len(new_solution.rosters) - 1)

        # 随机选择要移除的roster
        removed_rosters = random.sample(new_solution.rosters, destroy_size)

        # 从解中移除选中的roster
        for roster in removed_rosters:
            new_solution.rosters.remove(roster)

        # 重新计算目标函数
        new_solution._calculate_objective()

        return new_solution, removed_rosters


class WorstRosterDestroy(DestroyOperator):
    """移除最差roster的破坏算子"""

    def __init__(self):
        super().__init__("WorstRosterDestroy")

    def destroy(self, solution: ALNSSolution, destroy_size: int) -> Tuple[ALNSSolution, List[Roster]]:
        """移除成本最高的roster"""
        new_solution = solution.copy()

        if len(new_solution.rosters) <= destroy_size:
            destroy_size = max(1, len(new_solution.rosters) - 1)

        # 按成本排序，选择成本最高的roster
        sorted_rosters = sorted(new_solution.rosters,
                               key=lambda r: getattr(r, 'cost', 0),
                               reverse=True)

        removed_rosters = sorted_rosters[:destroy_size]

        # 从解中移除选中的roster
        for roster in removed_rosters:
            new_solution.rosters.remove(roster)

        new_solution._calculate_objective()

        return new_solution, removed_rosters


class RelatedFlightDestroy(DestroyOperator):
    """移除相关航班的破坏算子"""

    def __init__(self):
        super().__init__("RelatedFlightDestroy")

    def destroy(self, solution: ALNSSolution, destroy_size: int) -> Tuple[ALNSSolution, List[Roster]]:
        """移除包含相关航班的roster"""
        new_solution = solution.copy()

        if not new_solution.rosters:
            return new_solution, []

        # 随机选择一个起始航班
        all_flights_in_rosters = []
        for roster in new_solution.rosters:
            for duty in roster.duties:
                if isinstance(duty, Flight):
                    all_flights_in_rosters.append((duty, roster))

        if not all_flights_in_rosters:
            # 如果没有航班，回退到随机移除
            return RandomRosterDestroy().destroy(solution, destroy_size)

        seed_flight, _ = random.choice(all_flights_in_rosters)

        # 找到相关的航班（相同机场或相近时间）
        related_rosters = set()
        for roster in new_solution.rosters:
            for duty in roster.duties:
                if isinstance(duty, Flight):
                    # 检查是否为相关航班
                    if (duty.depaAirport == seed_flight.depaAirport or
                        duty.arriAirport == seed_flight.arriAirport or
                        abs((duty.std - seed_flight.std).total_seconds()) < 3600):  # 1小时内
                        related_rosters.add(roster)
                        break

        # 限制移除数量
        related_rosters = list(related_rosters)
        if len(related_rosters) > destroy_size:
            related_rosters = random.sample(related_rosters, destroy_size)
        elif len(related_rosters) == 0:
            # 如果没有找到相关roster，随机选择
            related_rosters = random.sample(new_solution.rosters,
                                          min(destroy_size, len(new_solution.rosters)))

        # 移除选中的roster
        for roster in related_rosters:
            if roster in new_solution.rosters:
                new_solution.rosters.remove(roster)

        new_solution._calculate_objective()

        return new_solution, related_rosters


class GreedyRepair(RepairOperator):
    """贪心修复算子"""

    def __init__(self, crews: List[Crew], flights: List[Flight],
                 ground_duties: List[GroundDuty], bus_info: List[BusInfo],
                 crew_leg_match_dict: Dict, layover_stations: Set[str], scoring_system: ScoringSystem=None):
        super().__init__("GreedyRepair", scoring_system=scoring_system)
        self.crews = crews
        self.flights = flights
        self.ground_duties = ground_duties
        self.bus_info = bus_info
        self.crew_leg_match_dict = crew_leg_match_dict
        self.layover_stations = layover_stations
        self.constraint_checker = UnifiedConstraintChecker(layover_stations)

    def repair(self, solution: ALNSSolution, removed_rosters: List[Roster]) -> ALNSSolution:
        """使用贪心策略修复解决方案"""
        new_solution = solution.copy()

        # 获取当前未覆盖的航班和地面任务
        covered_flights = set()
        covered_ground_duties = set()

        for roster in new_solution.rosters:
            for duty in roster.duties:
                if isinstance(duty, Flight):
                    covered_flights.add(duty.id)
                elif hasattr(duty, 'crewId') and hasattr(duty, 'airport'):
                    covered_ground_duties.add(duty.id)

        uncovered_flights = [f for f in self.flights if f.id not in covered_flights]
        uncovered_ground_duties = [gd for gd in self.ground_duties if gd.id not in covered_ground_duties]

        # 获取可用的机组（没有被分配roster的机组）
        assigned_crews = {roster.crew_id for roster in new_solution.rosters}
        available_crews = [crew for crew in self.crews if crew.crewId not in assigned_crews]

        # 为每个可用机组尝试创建新的roster
        for crew in available_crews:
            if not uncovered_flights and not uncovered_ground_duties:
                break

            # 获取该机组可执行的航班
            eligible_flight_ids = self.crew_leg_match_dict.get(crew.crewId, [])
            eligible_flights = [f for f in uncovered_flights if f.id in eligible_flight_ids]

            # 获取该机组的地面任务
            crew_ground_duties = [gd for gd in uncovered_ground_duties if gd.crewId == crew.crewId]

            # 尝试创建一个简单的roster
            new_roster = self._create_simple_roster(crew, eligible_flights, crew_ground_duties)

            if new_roster and new_roster.duties:
                new_solution.rosters.append(new_roster)

                # 更新未覆盖列表
                for duty in new_roster.duties:
                    if isinstance(duty, Flight) and duty in uncovered_flights:
                        uncovered_flights.remove(duty)
                    elif hasattr(duty, 'crewId') and duty in uncovered_ground_duties:
                        uncovered_ground_duties.remove(duty)

        new_solution._calculate_objective()
        return new_solution

    def _create_simple_roster(self, crew: Crew, eligible_flights: List[Flight],
                             crew_ground_duties: List[GroundDuty]) -> Optional[Roster]:
        """为机组创建一个简单的roster"""
        if not eligible_flights and not crew_ground_duties:
            return None

        # 合并所有任务并按时间排序
        all_tasks = []
        all_tasks.extend(eligible_flights)
        all_tasks.extend(crew_ground_duties)

        # 按开始时间排序
        all_tasks.sort(key=lambda x: getattr(x, 'std', getattr(x, 'startTime', datetime.min)))

        # 贪心选择任务，确保满足约束
        selected_tasks = []

        for task in all_tasks:
            # 检查是否可以添加这个任务
            temp_tasks = selected_tasks + [task]

            # 简单的约束检查
            if self._is_valid_task_sequence(temp_tasks):
                selected_tasks.append(task)

                # 限制任务数量，避免过度复杂
                if len(selected_tasks) >= 5:
                    break

        if selected_tasks:
            roster = Roster(crew.crewId, selected_tasks, 0.0)
            cost_details = self.scoring_system.calculate_roster_cost_with_dual_prices(
                roster, crew, {}, 0.0
            )
            roster.cost = cost_details['total_cost']
            return roster

        return None

    def _is_valid_task_sequence(self, tasks: List) -> bool:
        """检查任务序列是否有效"""
        if not tasks:
            return True

        # 简单的时间冲突检查
        for i in range(len(tasks) - 1):
            current_task = tasks[i]
            next_task = tasks[i + 1]

            # 获取任务的结束和开始时间
            current_end = getattr(current_task, 'sta', getattr(current_task, 'endTime', None))
            next_start = getattr(next_task, 'std', getattr(next_task, 'startTime', None))

            if current_end and next_start:
                # 检查最小连接时间
                time_gap = next_start - current_end
                if time_gap < timedelta(minutes=30):  # 最小30分钟间隔
                    return False

        return True


class RandomRepair(RepairOperator):
    """随机修复算子"""

    def __init__(self, crews: List[Crew], flights: List[Flight],
                 ground_duties: List[GroundDuty], bus_info: List[BusInfo],
                 crew_leg_match_dict: Dict, layover_stations: Set[str], scoring_system: ScoringSystem=None):
        super().__init__("RandomRepair", scoring_system= scoring_system)
        self.greedy_repair = GreedyRepair(crews, flights, ground_duties, bus_info,
                                        crew_leg_match_dict, layover_stations, scoring_system=scoring_system)

    def repair(self, solution: ALNSSolution, removed_rosters: List[Roster]) -> ALNSSolution:
        """使用随机策略修复解决方案"""
        # 简单实现：随机选择一部分被移除的roster重新加入
        new_solution = solution.copy()

        if removed_rosters:
            # 随机选择一部分被移除的roster
            num_to_restore = random.randint(1, max(1, len(removed_rosters) // 2))
            rosters_to_restore = random.sample(removed_rosters, num_to_restore)

            for roster in rosters_to_restore:
                # 检查是否与现有roster冲突
                if not self._conflicts_with_existing(roster, new_solution.rosters):
                    new_solution.rosters.append(copy.deepcopy(roster))

        # 使用贪心修复填补剩余空缺
        new_solution = self.greedy_repair.repair(new_solution, [])

        return new_solution

    def _conflicts_with_existing(self, new_roster: Roster, existing_rosters: List[Roster]) -> bool:
        """检查新roster是否与现有roster冲突"""
        # 检查机组冲突
        for existing_roster in existing_rosters:
            if existing_roster.crew_id == new_roster.crew_id:
                return True

        # 检查任务冲突
        new_tasks = {getattr(duty, 'id', str(duty)) for duty in new_roster.duties}
        for existing_roster in existing_rosters:
            existing_tasks = {getattr(duty, 'id', str(duty)) for duty in existing_roster.duties}
            if new_tasks & existing_tasks:  # 有交集
                return True

        return False


class AdaptiveWeightManager:
    """自适应权重管理器"""

    def __init__(self, operators: List, reaction_factor: float = 0.1):
        self.operators = operators
        self.reaction_factor = reaction_factor
        self.segment_size = 100  # 每100次迭代更新一次权重
        self.iteration_count = 0

        # 初始化权重
        for op in self.operators:
            op.weight = 1.0

    def select_operator(self) -> Any:
        """基于权重选择算子"""
        if not self.operators:
            return None

        # 计算权重总和
        total_weight = sum(op.weight for op in self.operators)
        if total_weight <= 0:
            # 如果所有权重都为0，重置为均等权重
            for op in self.operators:
                op.weight = 1.0
            total_weight = len(self.operators)

        # 轮盘赌选择
        rand_val = random.uniform(0, total_weight)
        cumulative_weight = 0

        for op in self.operators:
            cumulative_weight += op.weight
            if rand_val <= cumulative_weight:
                return op

        # 如果没有选中（浮点数精度问题），返回最后一个
        return self.operators[-1]

    def update_weights(self):
        """更新算子权重"""
        self.iteration_count += 1

        if self.iteration_count % self.segment_size == 0:
            for op in self.operators:
                if op.usage_count > 0:
                    success_rate = op.success_count / op.usage_count
                    # 基于成功率调整权重
                    op.weight = (1 - self.reaction_factor) * op.weight + self.reaction_factor * success_rate

                    # 重置计数器
                    op.usage_count = 0
                    op.success_count = 0
                else:
                    # 如果没有使用过，保持当前权重
                    pass

                # 确保权重不会太小
                op.weight = max(op.weight, 0.1)


class ALNSAlgorithm:
    """ALNS主算法类"""

    def __init__(self, flights: List[Flight], crews: List[Crew],
                 ground_duties: List[GroundDuty], bus_info: List[BusInfo],
                 crew_leg_match_dict: Dict, layover_stations: Set[str], scoring_system:ScoringSystem=None):
        self.flights = flights
        self.crews = crews
        self.ground_duties = ground_duties
        self.bus_info = bus_info
        self.crew_leg_match_dict = crew_leg_match_dict
        self.layover_stations = layover_stations
        self.scoring_system = scoring_system

        if scoring_system is None:
            raise ValueError("2. Scoring system must be provided")

        # 初始化算子
        self.destroy_operators = [
            RandomRosterDestroy(),
            WorstRosterDestroy(),
            RelatedFlightDestroy()
        ]

        self.repair_operators = [
            GreedyRepair(crews, flights, ground_duties, bus_info,
                        crew_leg_match_dict, layover_stations, scoring_system=self.scoring_system),
            RandomRepair(crews, flights, ground_duties, bus_info,
                        crew_leg_match_dict, layover_stations, scoring_system=self.scoring_system)
        ]

        # 权重管理器
        self.destroy_weight_manager = AdaptiveWeightManager(self.destroy_operators)
        self.repair_weight_manager = AdaptiveWeightManager(self.repair_operators)

        # 算法参数
        self.max_iterations = 1000**3
        self.time_limit = 3600  # 1小时
        self.destroy_size_min = 1
        self.destroy_size_max = 5

        # 模拟退火参数
        self.initial_temperature = 1000.0
        self.cooling_rate = 0.995
        self.min_temperature = 1.0

        # 统计信息
        self.iteration_count = 0
        self.best_solution = None
        self.current_solution = None
        self.temperature = self.initial_temperature

        # 验证器
        self.coverage_validator = CoverageValidator(min_coverage_rate=0.8)

    def solve(self, initial_solution: ALNSSolution) -> ALNSSolution:
        """执行ALNS算法"""
        print("开始ALNS算法求解...")
        start_time = time.time()

        # 初始化解
        self.current_solution = initial_solution.copy()
        self.best_solution = initial_solution.copy()
        self.temperature = self.initial_temperature

        print(f"初始解: {self.current_solution}")

        # 主循环
        for iteration in range(self.max_iterations):
            self.iteration_count = iteration

            # 检查时间限制
            if time.time() - start_time > self.time_limit:
                print(f"达到时间限制，算法终止")
                break

            # 选择破坏和修复算子
            destroy_op = self.destroy_weight_manager.select_operator()
            repair_op = self.repair_weight_manager.select_operator()

            # 确定破坏大小
            destroy_size = random.randint(self.destroy_size_min,
                                        min(self.destroy_size_max, len(self.current_solution.rosters)))

            try:
                # 破坏
                destroyed_solution, removed_elements = destroy_op.destroy(self.current_solution, destroy_size)

                # 修复
                new_solution = repair_op.repair(destroyed_solution, removed_elements)

                # 评估新解
                accept_solution = self._should_accept_solution(new_solution)

                # 更新算子权重
                improved = new_solution.is_better_than(self.current_solution)
                destroy_op.update_weight(improved)
                repair_op.update_weight(improved)

                # 接受或拒绝新解
                if accept_solution:
                    self.current_solution = new_solution

                    # 更新最优解
                    if new_solution.is_better_than(self.best_solution):
                        self.best_solution = new_solution.copy()
                        print(f"迭代 {iteration}: 找到更好解 {self.best_solution}")

                # 更新温度
                self.temperature = max(self.min_temperature,
                                     self.temperature * self.cooling_rate)

                # 定期输出进度
                if iteration % 100 == 0:
                    print(f"迭代 {iteration}: 当前解={self.current_solution.objective_value:.2f}, "
                          f"最优解={self.best_solution.objective_value:.2f}, "
                          f"温度={self.temperature:.2f}")

                    # 输出算子使用情况
                    self._print_operator_stats()

                # 更新权重
                self.destroy_weight_manager.update_weights()
                self.repair_weight_manager.update_weights()

            except Exception as e:
                print(f"迭代 {iteration} 出错: {e}")
                # print traceback
                traceback.print_exc()
                continue

        print(f"ALNS算法完成，总迭代次数: {self.iteration_count + 1}")
        print(f"最优解: {self.best_solution}")

        return self.best_solution

    def _should_accept_solution(self, new_solution: ALNSSolution) -> bool:
        """判断是否接受新解（模拟退火准则）"""
        if new_solution.is_better_than(self.current_solution):
            return True

        if self.temperature <= self.min_temperature:
            return False

        # 计算接受概率
        delta = new_solution.objective_value - self.current_solution.objective_value
        probability = math.exp(-delta / self.temperature)

        return random.random() < probability

    def _print_operator_stats(self):
        """输出算子使用统计"""
        print("破坏算子统计:")
        for op in self.destroy_operators:
            success_rate = op.success_count / op.usage_count if op.usage_count > 0 else 0
            print(f"  {op.name}: 权重={op.weight:.3f}, 使用={op.usage_count}, 成功率={success_rate:.3f}")

        print("修复算子统计:")
        for op in self.repair_operators:
            success_rate = op.success_count / op.usage_count if op.usage_count > 0 else 0
            print(f"  {op.name}: 权重={op.weight:.3f}, 使用={op.usage_count}, 成功率={success_rate:.3f}")

def main():
    """ALNS主函数"""
    print("=== ALNS机组排班优化系统 ===")

    # 数据加载
    print("正在加载数据...")
    data_path = UnifiedConfig.DATA_PATH
    all_data = load_all_data(data_path)

    if not all_data:
        print("数据加载失败，程序退出。")
        return

    flights = all_data["flights"]
    crews = all_data["crews"]
    bus_info = all_data["bus_info"]
    ground_duties = all_data["ground_duties"]
    crew_leg_match_list = all_data["crew_leg_matches"]
    layover_stations = all_data["layover_stations"]

    # 预处理机长-航班资质数据
    crew_leg_match_dict = {}
    for match in crew_leg_match_list:
        flight_id, crew_id = match.flightId, match.crewId
        if crew_id not in crew_leg_match_dict:
            crew_leg_match_dict[crew_id] = []
        crew_leg_match_dict[crew_id].append(flight_id)

    print(f"数据加载完成: 航班{len(flights)}个, 机组{len(crews)}个, 地面任务{len(ground_duties)}个")

    import pickle
    if not os.path.exists('initial_solution'):
        os.makedirs('initial_solution')
    # 从文件加载initial_rosters和scoring_system
    try:
        with open('initial_solution/initial_rosters.pkl', 'rb') as f:
            initial_rosters = pickle.load(f)
        with open('initial_solution/scoring_system.pkl', 'rb') as f:
            scoring_system = pickle.load(f)
        print("初始解和评分系统加载成功")
    except FileNotFoundError:
        print("初始解和评分系统未找到，将重新生成")
        # 生成初始解
        print("正在生成初始解...")
        initial_rosters = generate_initial_rosters_with_heuristic(
            flights, crews, bus_info, ground_duties, crew_leg_match_dict, layover_stations
        )
        # 定义scoring system
        scoring_system = ScoringSystem(flights, crews, layover_stations)
        if scoring_system is None:
            raise ValueError("1. Scoring system must be provided")
        try:
            # 将initial_rosters和scoring_system保存到文件，下次直接加载
            with open('initial_solution/initial_rosters.pkl', 'wb') as f:
                pickle.dump(initial_rosters, f)
            with open('initial_solution/scoring_system.pkl', 'wb') as f:
                pickle.dump(scoring_system, f)
        except:
            pass
    

    if not initial_rosters:
        print("初始解生成失败，程序退出。")
        return

    # 创建ALNS解决方案对象
    initial_solution = ALNSSolution(initial_rosters, flights, ground_duties, crews)
    print(f"初始解生成完成: {initial_solution}")

    # 验证初始解
    coverage_validator = CoverageValidator(min_coverage_rate=0.8)
    coverage_result = coverage_validator.validate_coverage(flights, initial_rosters)
    print(f"初始解航班覆盖率: {coverage_result['coverage_rate']:.2%}")

    # 创建ALNS算法实例
    alns = ALNSAlgorithm(flights, crews, ground_duties, bus_info,
                        crew_leg_match_dict, layover_stations, scoring_system=scoring_system)

    # 执行ALNS算法
    best_solution = alns.solve(initial_solution)

    # 验证最终解
    final_coverage_result = coverage_validator.validate_coverage(flights, best_solution.rosters)
    print(f"\n=== 最终结果 ===")
    print(f"最优解: {best_solution}")
    print(f"航班覆盖率: {final_coverage_result['coverage_rate']:.2%}")
    print(f"地面任务覆盖率: {best_solution.ground_duty_coverage_rate:.2%}")

    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"output/alns_result_{timestamp}.csv"

    # 更新initial_solution文件夹中的initial_rosters和scoring_system
    try:
        with open('initial_solution/initial_rosters.pkl', 'wb') as f:
            pickle.dump(best_solution.rosters, f)
        with open('initial_solution/scoring_system.pkl', 'wb') as f:
            pickle.dump(scoring_system, f)
    except:
        pass

    # 确保输出目录存在
    os.makedirs("output", exist_ok=True)

    # 使用简化的结果写入函数
    write_alns_results_to_csv(best_solution.rosters, output_file)
    print(f"结果已保存到: {output_file}")


def write_alns_results_to_csv(rosters: List[Roster], output_file: str):
    """简化的结果写入函数，专门用于ALNS"""
    import csv

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # 写入表头
        writer.writerow(['crewId', 'dutyId', 'dutyType', 'startTime', 'endTime', 'airport'])

        # 写入每个roster的任务
        for roster in rosters:
            for duty in roster.duties:
                if isinstance(duty, Flight):
                    writer.writerow([
                        roster.crew_id,
                        duty.id,
                        'Flight',
                        duty.std.strftime('%Y-%m-%d %H:%M:%S'),
                        duty.sta.strftime('%Y-%m-%d %H:%M:%S'),
                        f"{duty.depaAirport}-{duty.arriAirport}"
                    ])
                elif hasattr(duty, 'crewId') and hasattr(duty, 'airport'):
                    # 地面任务
                    writer.writerow([
                        roster.crew_id,
                        duty.id,
                        'GroundDuty',
                        duty.startTime.strftime('%Y-%m-%d %H:%M:%S'),
                        duty.endTime.strftime('%Y-%m-%d %H:%M:%S'),
                        duty.airport
                    ])
                elif hasattr(duty, 'depaAirport') and hasattr(duty, 'arriAirport'):
                    # 大巴任务
                    writer.writerow([
                        roster.crew_id,
                        getattr(duty, 'id', 'BUS'),
                        'Bus',
                        duty.startTime.strftime('%Y-%m-%d %H:%M:%S'),
                        duty.endTime.strftime('%Y-%m-%d %H:%M:%S'),
                        f"{duty.depaAirport}-{duty.arriAirport}"
                    ])


if __name__ == "__main__":
    main()