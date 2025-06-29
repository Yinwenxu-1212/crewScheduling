# file: master_problem.py

import gurobipy as gp
from gurobipy import GRB
from typing import List
from datetime import timedelta
from data_models import Flight, Roster, Crew
from dinkelbach_optimizer import DinkelbachOptimizer

class MasterProblem:
    def __init__(self, flights: List[Flight], crews: List[Crew]):
        self.flights = flights
        self.crews = crews
        
        # 使用Dinkelbach优化器替代原来的简单模型
        self.dinkelbach_optimizer = DinkelbachOptimizer(flights, crews)
        
        # 保持向后兼容的接口
        self.model = self.dinkelbach_optimizer.model
        self.rosters = self.dinkelbach_optimizer.rosters
        self.roster_vars = self.dinkelbach_optimizer.roster_vars
        self.uncovered_vars = self.dinkelbach_optimizer.uncovered_vars
        self.flight_constraints = self.dinkelbach_optimizer.flight_constraints
        self.crew_constraints = self.dinkelbach_optimizer.crew_constraints
        self.UNCOVERED_FLIGHT_PENALTY = self.dinkelbach_optimizer.UNCOVERED_FLIGHT_PENALTY
    
    # _setup_model方法已移除，现在由DinkelbachOptimizer处理模型设置

    def add_roster_column(self, roster: Roster):
        """添加新的roster列，委托给DinkelbachOptimizer处理"""
        self.dinkelbach_optimizer.add_roster_column(roster)
        
        # 注意：不要在这里设置目标函数，让DinkelbachOptimizer来管理
        # Dinkelbach算法会在solve_dinkelbach方法中动态更新目标函数

    def solve_lp(self, verbose=False, use_multiple_starts=True, num_starts=3) -> tuple[dict, dict, float]:
        """使用Dinkelbach算法求解主问题"""
        if use_multiple_starts:
            pi_duals, sigma_duals, obj_val = self.dinkelbach_optimizer.solve_dinkelbach_with_multiple_starts(
                num_starts=num_starts, verbose=verbose)
        else:
            pi_duals, sigma_duals, obj_val = self.dinkelbach_optimizer.solve_dinkelbach(verbose=verbose)
        
        if pi_duals is not None and verbose:
            # 输出解的详细信息
            solution_summary = self.dinkelbach_optimizer.get_solution_summary()
            print("\n=== Dinkelbach算法求解结果 ===")
            print(f"最终得分: {solution_summary.get('final_score', 0):.6f}")
            print(f"总飞行时间: {solution_summary.get('total_flight_hours', 0):.2f} 小时")
            print(f"总值勤天数: {solution_summary.get('total_duty_days', 0):.2f} 天")
            print(f"日均飞时: {solution_summary.get('avg_daily_flight_hours', 0):.4f} 小时/天")
            print(f"未覆盖航班: {solution_summary.get('uncovered_flights', 0)} 个")
            print(f"最终lambda: {solution_summary.get('lambda_final', 0):.6f}")
        
        return pi_duals, sigma_duals, obj_val
    
    def get_current_lambda(self):
        """获取当前Dinkelbach算法的lambda值"""
        return self.dinkelbach_optimizer.lambda_k
        

    def solve_bip(self):
        print("正在求解最终的BIP模型...")
        self.model.optimize()
        return self.model

    def get_selected_rosters(self):
        """获取被选中的排班方案"""
        import csv
        from datetime import datetime
        
        selected = []
        print("=== 调试：排班方案变量值 ===")
        print(f"总共有 {len(self.roster_vars)} 个排班方案变量")
        print(f"目标函数值: {self.model.ObjVal:.2f}")
        
        # 创建CSV文件记录所有方案的详细信息
        csv_filename = f"debug/debug_rosters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['方案编号', '变量值', '成本', '机组ID', '是否选中', '任务详情']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, (roster, var) in enumerate(self.roster_vars.items()):
                var_value = var.X
                is_selected = var_value > 0.5
                
                # 如果被选中，添加到selected列表
                if is_selected:
                    selected.append(roster)
                
                # 构建任务详情字符串
                task_details = []
                for duty in roster.duties:
                    if hasattr(duty, 'flightNo'):
                        task_details.append(f"Flight:{duty.flightNo}")
                    elif hasattr(duty, 'task'):
                        task_details.append(f"Ground:{duty.task}")
                    elif type(duty).__name__ == 'BusInfo':
                        task_details.append(f"Bus:{duty.id}")
                    elif type(duty).__name__ == 'GroundDuty':
                        task_details.append(f"Ground:{duty.id}")
                    else:
                        task_details.append(f"Other:{type(duty).__name__}")
                
                task_details_str = "; ".join(task_details)
                
                # 写入CSV
                writer.writerow({
                    '方案编号': i + 1,
                    '变量值': f"{var_value:.6f}",
                    '成本': f"{roster.cost:.2f}",
                    '机组ID': roster.crew_id,
                    '是否选中': '是' if is_selected else '否',
                    '任务详情': task_details_str
                })
        
        print(f"总共选中了 {len(selected)} 个排班方案")
        print(f"详细信息已保存到: {csv_filename}")
        return selected
