#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分支定价算法实现
Branch-and-Price Algorithm Implementation

实现完整的分支定价算法，包括分支定界树和节点管理。
"""

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from gurobipy import GRB
import time
from datetime import datetime

from master_problem import MasterProblem
from data_models import Roster
from unified_config import UnifiedConfig


@dataclass
class BranchingDecision:
    """分支决策"""
    crew_id: str
    roster_index: int  # roster在roster_vars中的索引
    value: int  # 0 或 1
    roster_signature: str = ""  # roster的唯一签名，用于子问题识别
    
    def __str__(self):
        return f"x[{self.crew_id},{self.roster_index}] = {self.value}"


@dataclass
class BAPNode:
    """分支定价树节点"""
    node_id: int
    parent_id: Optional[int] = None
    branching_decisions: List[BranchingDecision] = field(default_factory=list)
    lower_bound: float = float('inf')  # LP松弛的目标值
    is_integer: bool = False
    is_pruned: bool = False
    is_infeasible: bool = False
    solution: Optional[Dict] = None  # 存储LP解
    depth: int = 0
    
    def __lt__(self, other):
        """用于优先队列比较（最小化问题，选择lower_bound最小的）"""
        return self.lower_bound < other.lower_bound


class BranchAndPriceAlgorithm:
    """分支定价算法主类"""
    
    def __init__(self, flights, crews, ground_duties, initial_rosters, 
                 crew_leg_match_dict, layover_stations, bus_info, 
                 max_nodes=50, time_limit=1800, optimality_gap=0.01,
                 branching_strategy='most_fractional', node_selection='best_bound',
                 node_cg_max_iter=20, node_cg_time_limit=60,
                 pruning_tolerance=1e-6, integer_tolerance=1e-6):
        self.flights = flights
        self.crews = crews
        self.ground_duties = ground_duties
        self.initial_rosters = initial_rosters
        self.crew_leg_match_dict = crew_leg_match_dict
        self.layover_stations = layover_stations
        self.bus_info = bus_info
        
        # 算法参数（从配置文件获取）
        self.max_nodes = max_nodes
        self.max_time = time_limit
        self.column_generation_max_iter = node_cg_max_iter
        self.branching_tolerance = pruning_tolerance
        self.optimality_gap = optimality_gap
        self.branching_strategy = branching_strategy
        self.node_selection = node_selection
        self.integer_tolerance = integer_tolerance
        
        # 算法状态
        self.nodes_explored = 0
        self.incumbent_value = float('inf')  # 当前最好的整数解
        self.incumbent_solution = None
        self.start_time = time.time()
        
        # 节点管理
        self.node_counter = 0
        self.active_nodes = []  # 最小堆
        self.all_nodes = {}  # node_id -> node
        
        # 全局列池（避免重复生成）
        self.global_roster_pool = {}  # roster_signature -> roster
        
        # 改进的roster索引机制
        self.roster_index_map = {}  # roster_signature -> roster_index
        self.index_roster_map = {}  # roster_index -> roster
        self.next_roster_index = 0
        
        # 注册所有初始roster并添加到全局池
        for roster in self.initial_rosters:
            self._register_roster(roster)
            sig = self._get_roster_signature(roster)
            self.global_roster_pool[sig] = roster
        
    def _register_roster(self, roster):
        """注册roster并分配唯一索引"""
        sig = self._get_roster_signature(roster)
        if sig not in self.roster_index_map:
            idx = self.next_roster_index
            self.roster_index_map[sig] = idx
            self.index_roster_map[idx] = roster
            self.next_roster_index += 1
            return idx
        return self.roster_index_map[sig]
        
    def solve(self, verbose=True) -> Tuple[Optional[List[Roster]], float]:
        """
        执行分支定价算法
        
        Returns:
            (selected_rosters, objective_value)
        """
        import time
        self.start_time = time.time()  # 设置为实例变量
        
        if verbose:
            print("\n=== 开始分支定价算法 ===")
            print(f"最大节点数: {self.max_nodes}")
            print(f"时间限制: {self.max_time}秒")
        
        print(f"DEBUG: 分支定价算法初始化完成")
        print(f"DEBUG: 初始roster数量: {len(self.initial_rosters)}")
        print(f"DEBUG: 全局roster池大小: {len(self.global_roster_pool)}")
        print(f"DEBUG: 机组数量: {len(self.crews)}")
        print(f"DEBUG: 航班数量: {len(self.flights)}")
        
        # 创建根节点
        print(f"DEBUG: 创建根节点...")
        root = self._create_node(parent_id=None, branching_decisions=[])
        print(f"DEBUG: 根节点创建完成，ID: {root.node_id}")
        heapq.heappush(self.active_nodes, root)
        
        # 主循环
        print(f"DEBUG: 开始主循环，活跃节点数: {len(self.active_nodes)}")
        while self.active_nodes and not self._should_stop():
            print(f"DEBUG: 主循环迭代开始，活跃节点数: {len(self.active_nodes)}")
            # 检查时间限制
            elapsed_time = time.time() - self.start_time
            if elapsed_time > self.max_time:
                if verbose:
                    print(f"\n达到时间限制 ({elapsed_time:.1f}秒)，终止算法")
                break
            
            # 选择最有希望的节点（lower_bound最小）
            current_node = heapq.heappop(self.active_nodes)
            print(f"DEBUG: 选择节点 {current_node.node_id}, lower_bound: {current_node.lower_bound}, incumbent: {self.incumbent_value}")
            
            # 只有非根节点且已经求解过的节点才进行剪枝检查
            if current_node.is_pruned or (current_node.node_id != 0 and current_node.lower_bound >= self.incumbent_value):
                print(f"DEBUG: 节点 {current_node.node_id} 被剪枝，is_pruned: {current_node.is_pruned}, bound check: {current_node.lower_bound >= self.incumbent_value}")
                continue
                
            self.nodes_explored += 1
            print(f"DEBUG: 开始处理节点 {current_node.node_id}")
            
            if verbose:
                print(f"\n处理节点 {current_node.node_id} (深度: {len(current_node.branching_decisions)}, 已用时: {elapsed_time:.1f}s)")
            
            # 在当前节点求解列生成
            is_feasible, is_integer, obj_value, solution = self._solve_node_with_column_generation(
                current_node, verbose=False
            )
            
            if not is_feasible:
                current_node.is_infeasible = True
                if verbose:
                    print(f"  节点 {current_node.node_id} 不可行，剪枝")
                continue
            
            current_node.lower_bound = obj_value
            current_node.solution = solution
            current_node.is_integer = is_integer
            
            # 剪枝检查
            if obj_value >= self.incumbent_value:
                current_node.is_pruned = True
                continue
            
            # 如果是整数解，更新incumbent
            if is_integer:
                print(f"DEBUG: 节点 {current_node.node_id} 是整数解，目标值: {obj_value:.6f}")
                print(f"DEBUG: 当前incumbent_value: {self.incumbent_value}")
                print(f"DEBUG: 比较结果: {obj_value} < {self.incumbent_value} = {obj_value < self.incumbent_value}")
                if obj_value < self.incumbent_value:
                    self.incumbent_value = obj_value
                    self.incumbent_solution = solution
                    print(f"DEBUG: 更新incumbent成功！新的最优值: {obj_value:.6f}")
                    if verbose:
                        print(f"  找到新的最优整数解: {obj_value:.2f}")
                else:
                    print(f"DEBUG: 目标值不优于当前incumbent，未更新")
                continue
            
            # 如果不是整数解，进行分支
            self._branch(current_node)
            
            # 每10个节点报告一次进度
            if self.nodes_explored % 10 == 0 and verbose:
                print(f"  进度: 已处理 {self.nodes_explored} 个节点，活跃节点 {len(self.active_nodes)} 个")
        
        # 返回结果
        if self.incumbent_solution:
            selected_rosters = self._extract_rosters_from_solution(self.incumbent_solution)
            return selected_rosters, self.incumbent_value
        else:
            if verbose:
                print("\n未找到可行的整数解")
            return None, float('inf')
    
    def _create_node(self, parent_id: Optional[int], 
                     branching_decisions: List[BranchingDecision]) -> BAPNode:
        """创建新节点"""
        node = BAPNode(
            node_id=self.node_counter,
            parent_id=parent_id,
            branching_decisions=branching_decisions.copy(),
            depth=len(branching_decisions)
        )
        self.node_counter += 1
        self.all_nodes[node.node_id] = node
        return node
    
    def _should_stop(self) -> bool:
        """检查是否应该停止算法"""
        if self.nodes_explored >= self.max_nodes:
            print(f"\n达到最大节点数限制 ({self.max_nodes})")
            return True
        
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.max_time:
            print(f"\n达到时间限制 ({self.max_time}秒)")
            return True
            
        return False
    
    def _solve_node_with_column_generation(self, node: BAPNode, verbose=False) -> Tuple[bool, bool, float, Optional[Dict]]:
        """
        在给定节点运行列生成算法
        
        Returns:
            (is_feasible, is_integer, objective_value, solution)
        """
        # 创建该节点的主问题
        master_problem = MasterProblem(
            self.flights, self.crews, self.ground_duties, self.layover_stations
        )
        
        # 添加初始解
        print(f"DEBUG: 添加 {len(self.initial_rosters)} 个初始roster")
        # 注册所有初始roster并添加到全局池
        for roster in self.initial_rosters:
            self._register_roster(roster)
            sig = self._get_roster_signature(roster)
            self.global_roster_pool[sig] = roster
        
        # 添加全局列池中的所有列到主问题
        for roster in self.global_roster_pool.values():
            master_problem.add_roster(roster)
        
        # 应用分支决策（固定某些变量）
        self._apply_branching_decisions(master_problem, node.branching_decisions)
        
        # 动态调整列生成参数
        node_depth = len(node.branching_decisions)
        
        # 检查初始模型状态
        if node_depth == 0:  # 根节点
            print(f"DEBUG: 根节点模型状态检查")
            print(f"  - 总航班数: {len(self.flights)}")
            print(f"  - 总机组数: {len(self.crews)}")
            print(f"  - 总占位任务数: {len(self.ground_duties)}")
            print(f"  - 初始roster数: {len(self.initial_rosters)}")
            print(f"  - 全局roster池大小: {len(self.global_roster_pool)}")
            print(f"  - 主问题roster变量数: {len(master_problem.roster_vars)}")
            
            # 检查航班覆盖情况
            covered_flights = set()
            for roster in master_problem.roster_vars.keys():
                for duty in roster.duties:
                    if hasattr(duty, 'flightNo') and not getattr(duty, 'is_positioning', False):
                        covered_flights.add(duty.id)
            
            uncovered_flights = set(f.id for f in self.flights) - covered_flights
            print(f"  - 被初始roster覆盖的航班数: {len(covered_flights)}")
            print(f"  - 未被覆盖的航班数: {len(uncovered_flights)}")
            
            if len(uncovered_flights) > 0:
                print(f"  - 前10个未覆盖航班: {list(uncovered_flights)[:10]}")
        
        # 根据节点深度动态调整迭代限制
        if node_depth == 0:  # 根节点
            max_iterations = min(self.column_generation_max_iter, 20)  # 保守设置为20以内
            no_improvement_threshold = 3
        elif node_depth <= 2:  # 浅层节点
            max_iterations = min(self.column_generation_max_iter, 15)
            no_improvement_threshold = 2
        else:  # 深层节点
            max_iterations = min(self.column_generation_max_iter, 10)
            no_improvement_threshold = 1
        
        if verbose:
            print(f"  节点深度: {node_depth}, 最大迭代: {max_iterations}, 收敛阈值: {no_improvement_threshold}")
        
        # 列生成循环
        no_improvement_rounds = 0
        previous_obj = float('inf')
        
        for iteration in range(max_iterations):
            if node_depth == 0:  # 根节点
                print(f"DEBUG: 根节点列生成迭代 {iteration+1}/{max_iterations}")
            
            # 求解LP松弛
            pi_duals, sigma_duals, ground_duty_duals, current_obj = master_problem.solve_lp(verbose=False)
            
            # --- 添加调试代码 ---
            if iteration == 0 and node_depth == 0:  # 根节点的第一次LP求解
                print(f"DEBUG: Root node, first LP solve status: {master_problem.model.status}")
                if master_problem.model.status != 2:  # GRB.OPTIMAL = 2
                    print("DEBUG: Root node RMP is not optimal! Might be infeasible or unbounded.")
                    print(f"DEBUG: Model status code: {master_problem.model.status}")
                    
                    # 尝试计算IIS（不可行子系统）
                    try:
                        master_problem.model.computeIIS()
                        print("DEBUG: IIS computed, writing to file...")
                        master_problem.model.write("debug_infeasible_model.ilp")
                    except Exception as e:
                        print(f"DEBUG: Failed to compute IIS: {e}")
                    
                    # 检查约束状态
                    print("DEBUG: Checking constraint status...")
                    for constr in master_problem.model.getConstrs():
                        if constr.Slack < -1e-6:  # 违反的约束
                            print(f"  Violated constraint: {constr.ConstrName}, slack: {constr.Slack}")
                else:
                    print(f"DEBUG: Root node LP optimal, objective: {current_obj:.6f}")
            # --- 调试代码结束 ---
            
            if pi_duals is None:
                if node_depth == 0:
                    print("DEBUG: Root node LP solve failed, returning infeasible")
                return False, False, float('inf'), None
            
            # 检查收敛
            if abs(current_obj - previous_obj) < 1e-6:
                no_improvement_rounds += 1
                if no_improvement_rounds >= no_improvement_threshold:
                    break
            else:
                no_improvement_rounds = 0
            previous_obj = current_obj
            
            # 为每个机组求解子问题
            new_rosters_found = 0
            
            for crew in self.crews:
                # 检查该机组是否被分支决策固定
                if self._is_crew_fixed(crew.crewId, node.branching_decisions):
                    continue
                
                # 求解子问题
                try:
                    if hasattr(self, 'solve_subproblem_func'):
                        # 收集该机组的分支约束
                        crew_branching_constraints = []
                        for decision in node.branching_decisions:
                            if decision.crew_id == crew.crewId:
                                crew_branching_constraints.append(decision)
                        
                        new_roster = self.solve_subproblem_func(
                            crew, pi_duals, sigma_duals, ground_duty_duals,
                            self.flights, self.ground_duties, self.bus_info,
                            self.crew_leg_match_dict, self.layover_stations,
                            branching_constraints=crew_branching_constraints
                        )
                        
                        if new_roster and hasattr(new_roster, 'reduced_cost') and new_roster.reduced_cost < -1e-6:
                            roster_sig = self._get_roster_signature(new_roster)
                            if roster_sig not in self.global_roster_pool:
                                # 注册新roster并添加到全局池
                                self._register_roster(new_roster)
                                self.global_roster_pool[roster_sig] = new_roster
                                master_problem.add_roster(new_roster)
                                new_rosters_found += 1
                                if verbose:
                                    print(f"  添加新列: crew {crew.crewId}, reduced_cost = {new_roster.reduced_cost:.4f}")
                except Exception as e:
                    if verbose:
                        print(f"子问题求解失败 (crew {crew.crewId}): {e}")
            
            if new_rosters_found == 0:
                break
        
        # 获取解
        solution = self._get_lp_solution(master_problem)
        is_integer = self._check_integer_solution(solution)
        
        # 添加调试信息
        if node_depth == 0:  # 根节点
            print(f"DEBUG: 根节点列生成完成")
            print(f"  - 最终目标值: {current_obj:.6f}")
            print(f"  - 解的变量数: {len(solution)}")
            print(f"  - 是否为整数解: {is_integer}")
            if not is_integer:
                print(f"  - 分数变量:")
                for var_key, value in solution.items():
                    if isinstance(var_key, tuple) and len(var_key) == 2:
                        key1, key2 = var_key
                        if 'roster' in str(key2):
                            if abs(value - round(value)) > self.branching_tolerance:
                                print(f"    {var_key}: {value:.6f}")
        
        return True, is_integer, current_obj, solution
    
    def _apply_branching_decisions(self, master_problem, decisions: List[BranchingDecision]):
        """应用分支决策到主问题"""
        # 应用分支决策：固定某些变量的值
        for decision in decisions:
            # 使用roster索引而不是id()进行匹配
            if decision.roster_index in self.index_roster_map:
                roster = self.index_roster_map[decision.roster_index]
                if roster in master_problem.roster_vars:
                    var = master_problem.roster_vars[roster]
                    var.lb = decision.value
                    var.ub = decision.value
                    if hasattr(self, 'verbose') and self.verbose:
                        print(f"  应用分支决策: crew={decision.crew_id}, roster_idx={decision.roster_index}, value={decision.value}")
                else:
                    if hasattr(self, 'verbose') and self.verbose:
                        print(f"  警告: roster不在主问题变量中 {decision}")
            else:
                if hasattr(self, 'verbose') and self.verbose:
                    print(f"  警告: 未找到对应的roster索引 {decision}")
    
    def _get_roster_signature(self, roster):
        """获取roster的唯一、稳健的标识"""
        duties_sig = []
        # 按任务开始时间排序，确保签名唯一性
        sorted_duties = sorted(roster.duties, key=lambda d: getattr(d, 'startTime', getattr(d, 'std', datetime.min)))
        for duty in sorted_duties:
            # 使用ID和时间戳来唯一标识一个任务实例
            start_time = getattr(duty, 'startTime', getattr(duty, 'std', None))
            end_time = getattr(duty, 'endTime', getattr(duty, 'sta', None))
            if start_time and end_time:
                start_time_str = start_time.strftime('%Y%m%d%H%M')
                end_time_str = end_time.strftime('%Y%m%d%H%M')
                duties_sig.append(f"{duty.id}-{start_time_str}-{end_time_str}")
            else:
                duties_sig.append(f"{duty.id}")
        return f"{roster.crew_id}_{'|'.join(duties_sig)}"
     
    def _branch(self, node: BAPNode):
        """对节点进行分支"""
        # 选择分支变量（最接近0.5的分数变量）
        branch_var = self._select_branching_variable(node.solution)
        
        if branch_var is None:
            return
        
        crew_id, roster_idx, value = branch_var
        
        # 获取roster签名
        roster_signature = ""
        if roster_idx in self.index_roster_map:
            roster = self.index_roster_map[roster_idx]
            roster_signature = self._get_roster_signature(roster)
        
        # 创建两个子节点
        # 左分支: x = 0
        left_decisions = node.branching_decisions + [
            BranchingDecision(crew_id, roster_idx, 0, roster_signature)
        ]
        left_node = self._create_node(node.node_id, left_decisions)
        heapq.heappush(self.active_nodes, left_node)
        
        # 右分支: x = 1
        right_decisions = node.branching_decisions + [
            BranchingDecision(crew_id, roster_idx, 1, roster_signature)
        ]
        right_node = self._create_node(node.node_id, right_decisions)
        heapq.heappush(self.active_nodes, right_node)
    
    def _select_branching_variable(self, solution: Dict) -> Optional[Tuple[str, int, float]]:
        """选择分支变量"""
        best_var = None
        best_distance = float('inf')
        
        for (crew_id, roster_key), value in solution.items():
            if 'roster' not in str(roster_key):  # 只考虑roster变量
                continue
                
            # 计算到0.5的距离
            distance = abs(value - 0.5)
            
            # 选择最接近0.5的分数变量
            if self.branching_tolerance < value < 1 - self.branching_tolerance:
                if distance < best_distance:
                    best_distance = distance
                    # 提取roster索引
                    try:
                        roster_idx = int(roster_key.replace('roster_', ''))
                        best_var = (crew_id, roster_idx, value)
                    except ValueError:
                        continue
        
        return best_var
    
    def _check_integer_solution(self, solution: Dict) -> bool:
        """检查解是否为整数解"""
        for var_key, value in solution.items():
            # var_key是tuple格式: (crew_id, roster_key) 或 ('uncovered', flight_id)
            if isinstance(var_key, tuple) and len(var_key) == 2:
                key1, key2 = var_key
                if 'roster' in str(key2):  # 检查第二个元素是否包含'roster'
                    if abs(value - round(value)) > self.branching_tolerance:
                        return False
        return True
    
    def _get_lp_solution(self, master_problem) -> Dict:
        """从主问题获取LP解"""
        solution = {}
        
        # 获取roster变量的值
        for roster, var in master_problem.roster_vars.items():
            if var.X > self.branching_tolerance:
                # 使用roster索引而不是id()
                roster_idx = self._register_roster(roster)
                solution[(roster.crew_id, f"roster_{roster_idx}")] = var.X
        
        # 获取未覆盖变量的值
        for flight_id, var in master_problem.uncovered_vars.items():
            if var.X > self.branching_tolerance:
                solution[('uncovered', flight_id)] = var.X
                
        return solution
    
    def _extract_rosters_from_solution(self, solution: Dict) -> List[Roster]:
        """从解中提取被选中的roster"""
        selected_rosters = []
        
        # 从solution中恢复roster对象
        # solution的格式: {(crew_id, roster_key): value}
        for (key1, key2), value in solution.items():
            if 'roster' in str(key2) and value > 0.5:
                # 使用roster索引而不是id()进行匹配
                try:
                    roster_idx = int(key2.replace('roster_', ''))
                    if roster_idx in self.index_roster_map:
                        roster = self.index_roster_map[roster_idx]
                        selected_rosters.append(roster)
                except ValueError:
                    # 如果解析索引失败，跳过
                    continue
        
        # 如果没有找到足够的roster，从初始解中补充
        if not selected_rosters:
            print("警告：未能从解中提取roster，使用初始解")
            selected_rosters = self.initial_rosters[:]
        
        return selected_rosters
    
    def _is_crew_fixed(self, crew_id: str, decisions: List[BranchingDecision]) -> bool:
        """检查机组是否被分支决策固定"""
        for decision in decisions:
            if decision.crew_id == crew_id and decision.value == 1:
                return True
        return False
    
    def _get_roster_signature_legacy(self, roster: Roster) -> str:
        """获取roster的唯一标识（已弃用，保留以兼容性）"""
        duty_ids = sorted([duty.id for duty in roster.duties])
        return f"{roster.crew_id}_{hash(tuple(duty_ids))}"
    
    def _print_progress(self):
        """打印算法进度"""
        elapsed = time.time() - self.start_time
        print(f"\r节点: {self.nodes_explored}, 活跃: {len(self.active_nodes)}, "
              f"最优: {self.incumbent_value:.2f}, 时间: {elapsed:.1f}s", end='')
    
    def set_subproblem_solver(self, solver_func):
        """设置子问题求解函数"""
        self.solve_subproblem_func = solver_func


# 配置文件扩展
class BAPConfig:
    """分支定价算法配置"""
    MAX_BAP_NODES = 1000  # 最大探索节点数
    BAP_TIME_LIMIT = 3600  # 时间限制（秒）
    BRANCHING_STRATEGY = 'most_fractional'  # 分支策略
    NODE_SELECTION = 'best_bound'  # 节点选择策略
