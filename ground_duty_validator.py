# ground_duty_validator.py
# 地面值勤验证器

from typing import List, Dict, Set, Tuple, Optional
from data_models import GroundDuty, Roster, Crew
import logging
from datetime import datetime, timedelta

class GroundDutyValidator:
    """地面值勤验证器
    
    验证排班方案是否满足地面值勤要求
    注意：groundDuty是占位任务，不是置位任务
    """
    
    def __init__(self, ground_duties: List[GroundDuty], crews: Optional[List[Crew]] = None):
        """
        初始化验证器
        
        Args:
            ground_duties: 地面值勤任务列表
            crews: 机组人员列表（可选）
        """
        self.ground_duties = ground_duties
        self.crews = crews or []
        self.logger = logging.getLogger(__name__)
        
        # 按机组人员分组地面值勤任务
        self.crew_ground_duties = {}
        for duty in ground_duties:
            crew_id = duty.crewId
            if crew_id not in self.crew_ground_duties:
                self.crew_ground_duties[crew_id] = []
            self.crew_ground_duties[crew_id].append(duty)
    
    def validate_solution(self, rosters: List[Roster], master_problem=None) -> Dict:
        """
        验证排班方案的地面值勤分配
        
        Args:
            rosters: 排班方案列表
            master_problem: 主问题实例，用于获取uncovered_ground_duty_vars信息
            
        Returns:
            Dict: 包含验证结果的字典
        """
        
        # 获取所有地面值勤ID
        all_ground_duty_ids = {duty.id for duty in self.ground_duties}
        total_ground_duties = len(all_ground_duty_ids)
        
        # 🔧 修复：从master_problem获取正确的覆盖信息
        if master_problem and hasattr(master_problem, 'uncovered_ground_duty_vars'):
            # 从master_problem的uncovered_ground_duty_vars获取覆盖信息
            uncovered_count = 0
            unassigned_duty_ids = set()
            
            try:
                for duty_id, var in master_problem.uncovered_ground_duty_vars.items():
                    if hasattr(var, 'X') and var.X > 0.5:  # 未覆盖
                        uncovered_count += 1
                        unassigned_duty_ids.add(duty_id)
                
                assigned_count = total_ground_duties - uncovered_count
                assigned_ground_duty_ids = all_ground_duty_ids - unassigned_duty_ids
                
            except Exception as e:
                self.logger.warning(f"从master_problem获取地面值勤覆盖信息失败: {e}")
                # 回退到原有逻辑
                assigned_ground_duty_ids, assigned_count, unassigned_duty_ids = self._fallback_validation(rosters)
        else:
            # 回退到原有逻辑（但已知有问题）
            assigned_ground_duty_ids, assigned_count, unassigned_duty_ids = self._fallback_validation(rosters)
        
        # 计算覆盖率
        coverage_rate = assigned_count / total_ground_duties if total_ground_duties > 0 else 0.0
        
        # 找出未分配的地面值勤
        unassigned_duties = [duty for duty in self.ground_duties if duty.id in unassigned_duty_ids]
        
        # 🔧 增强验证逻辑 - 检查覆盖率和机组分配正确性
        violations = []
        crew_violations = {}
        assignment_details = {}
        
        # 1. 为每个有未覆盖地面值勤的机组创建违规记录
        for duty in unassigned_duties:
            crew_id = duty.crewId
            if crew_id not in crew_violations:
                crew_violations[crew_id] = []
            
            duty_type = "值勤" if duty.isDuty == 1 else "休息"
            violation = {
                'type': 'missing_ground_duty',
                'crew_id': crew_id,
                'duty_id': duty.id,
                'message': f'机组 {crew_id} 缺少必须执行的地面{duty_type} {duty.id}'
            }
            crew_violations[crew_id].append(violation)
            violations.append(violation)
        
        # 2. 🔧 新增：检查机组分配正确性（地面值勤是否被正确的机组执行）
        wrong_assignments = self._check_crew_assignment_correctness(rosters)
        for wrong_assignment in wrong_assignments:
            crew_id = wrong_assignment['actual_crew']
            if crew_id not in crew_violations:
                crew_violations[crew_id] = []
            
            violation = {
                'type': 'wrong_crew_assignment',
                'crew_id': crew_id,
                'duty_id': wrong_assignment['duty_id'],
                'expected_crew': wrong_assignment['expected_crew'],
                'message': f'地面值勤 {wrong_assignment["duty_id"]} 应由 {wrong_assignment["expected_crew"]} 执行，但被分配给了 {crew_id}'
            }
            crew_violations[crew_id].append(violation)
            violations.append(violation)
        
        # 判断是否有效
        is_valid = coverage_rate >= 0.80  # 要求80%以上覆盖率
        
        # 记录日志
        self.logger.info(f"地面值勤验证结果: {coverage_rate:.2%} ({assigned_count}/{total_ground_duties})")
        if not is_valid:
            self.logger.warning(f"地面值勤验证失败！违规数量: {len(violations)}")
            self.logger.warning(f"未分配地面值勤数量: {len(unassigned_duties)}")
        
        return {
            'is_valid': is_valid,
            'total_ground_duties': total_ground_duties,
            'assigned_ground_duties': assigned_count,
            'coverage_rate': coverage_rate,
            'violations': violations,
            'crew_violations': crew_violations,
            'unassigned_duties': unassigned_duties,
            'unassigned_duty_ids': list(unassigned_duty_ids),
            'assignment_details': assignment_details
        }
    
    def _fallback_validation(self, rosters: List[Roster]):
        """
        回退验证逻辑 - 通过检查roster中的地面值勤任务
        注意：这个方法已知有问题，仅作为备用
        """
        all_ground_duty_ids = {duty.id for duty in self.ground_duties}
        assigned_ground_duty_ids = set()
        
        for roster in rosters:
            crew_id = roster.crew_id
            crew_ground_duties = self.crew_ground_duties.get(crew_id, [])
            
            for duty in roster.duties:
                # 检查是否是地面值勤任务 - 通过ID前缀识别
                if hasattr(duty, 'id') and duty.id.startswith('Grd_'):
                    assigned_ground_duty_ids.add(duty.id)
                # 或者检查是否在该机组的地面值勤列表中
                elif hasattr(duty, 'id') and any(gd.id == duty.id for gd in crew_ground_duties):
                    assigned_ground_duty_ids.add(duty.id)
        
        assigned_count = len(assigned_ground_duty_ids)
        unassigned_duty_ids = all_ground_duty_ids - assigned_ground_duty_ids
        
        return assigned_ground_duty_ids, assigned_count, unassigned_duty_ids
    
    def _check_crew_assignment_correctness(self, rosters: List[Roster]) -> List[dict]:
        """
        检查地面值勤是否被正确的机组执行
        
        Returns:
            List[dict]: 错误分配的列表，每个元素包含duty_id, expected_crew, actual_crew
        """
        wrong_assignments = []
        
        # 创建地面值勤ID到预期机组的映射
        duty_to_expected_crew = {duty.id: duty.crewId for duty in self.ground_duties}
        
        # 检查每个roster中的地面值勤分配
        for roster in rosters:
            crew_id = roster.crew_id
            
            for duty in roster.duties:
                # 检查是否是地面值勤任务
                if hasattr(duty, 'id') and duty.id.startswith('Grd_'):
                    duty_id = duty.id
                    expected_crew = duty_to_expected_crew.get(duty_id)
                    
                    # 如果这个地面值勤应该由其他机组执行
                    if expected_crew and expected_crew != crew_id:
                        wrong_assignments.append({
                            'duty_id': duty_id,
                            'expected_crew': expected_crew,
                            'actual_crew': crew_id
                        })
        
        return wrong_assignments
    
    def _validate_crew_ground_duties(self, crew_id: str, crew_ground_duties: List[GroundDuty], 
                                   assigned_duty_ids: List[str]) -> Dict:
        """
        验证单个机组的地面值勤分配
        
        Args:
            crew_id: 机组ID
            crew_ground_duties: 该机组的地面值勤任务
            assigned_duty_ids: 已分配的地面值勤ID
            
        Returns:
            Dict: 验证结果
        """
        violations = []
        
        # 检查所有地面值勤（占位任务）：isDuty=0代表休息，isDuty=1代表值勤，但都需要执行
        # 根据业务逻辑，所有分配给机组的地面值勤（占位任务）都必须执行
        for duty in crew_ground_duties:
            if duty.id not in assigned_duty_ids:
                duty_type = "值勤" if duty.isDuty == 1 else "休息"
                violations.append({
                    'type': 'missing_ground_duty',
                    'crew_id': crew_id,
                    'duty_id': duty.id,
                    'message': f'机组 {crew_id} 缺少必须执行的地面{duty_type} {duty.id}'
                })
        
        # 检查是否分配了不属于该机组的地面值勤
        crew_duty_ids = {duty.id for duty in crew_ground_duties}
        for assigned_id in assigned_duty_ids:
            if assigned_id not in crew_duty_ids:
                violations.append({
                    'type': 'invalid_ground_duty_assignment',
                    'crew_id': crew_id,
                    'duty_id': assigned_id,
                    'message': f'机组 {crew_id} 被分配了不属于其的地面值勤 {assigned_id}'
                })
        
        return {
            'violations': violations,
            'total_duties_count': len(crew_ground_duties),
            'assigned_duties_count': len([d for d in crew_ground_duties if d.id in assigned_duty_ids])
        }
    
    def get_validation_report(self, validation_result: Dict) -> str:
        """
        生成地面值勤验证报告
        
        Args:
            validation_result: validate_solution返回的结果
            
        Returns:
            str: 格式化的验证报告
        """
        result = validation_result
        
        report = []
        report.append("=== 地面值勤验证报告 ===")
        report.append(f"总地面值勤数: {result['total_ground_duties']}")
        report.append(f"已分配地面值勤数: {result['assigned_ground_duties']}")
        report.append(f"未分配地面值勤数: {len(result['unassigned_duties'])}")
        report.append(f"覆盖率: {result['coverage_rate']:.2%}")
        report.append(f"违规数量: {len(result['violations'])}")
        
        if result['is_valid']:
            report.append("✅ 验证通过：满足地面值勤要求")
        else:
            report.append("❌ 验证失败：存在地面值勤违规")
            
            # 显示违规详情
            if result['violations']:
                report.append("\n违规详情:")
                violation_types = {}
                for violation in result['violations']:
                    v_type = violation['type']
                    if v_type not in violation_types:
                        violation_types[v_type] = []
                    violation_types[v_type].append(violation)
                
                for v_type, violations in violation_types.items():
                    type_name = {
                        'missing_ground_duty': '缺少地面值勤',
                        'wrong_crew_assignment': '错误机组分配',
                        'invalid_ground_duty_assignment': '无效地面值勤分配'
                    }.get(v_type, v_type)
                    
                    report.append(f"  {type_name}: {len(violations)} 个")
                    for violation in violations[:3]:  # 只显示前3个
                        report.append(f"    - {violation['message']}")
                    if len(violations) > 3:
                        report.append(f"    ... 还有 {len(violations) - 3} 个类似违规")
            
            # 显示未分配的地面值勤
            if result['unassigned_duties']:
                report.append("\n未分配地面值勤（前10个）:")
                for duty in result['unassigned_duties'][:10]:
                    duty_type = "值勤" if duty.isDuty == 1 else "休息"
                    report.append(f"  - {duty.id}: {duty.crewId} at {duty.airport} ({duty_type})")
                if len(result['unassigned_duties']) > 10:
                    report.append(f"  ... 还有 {len(result['unassigned_duties']) - 10} 个未分配地面值勤")
        
        return "\n".join(report)
    
    def suggest_improvements(self, validation_result: Dict) -> List[str]:
        """
        根据验证结果提供改进建议
        
        Args:
            validation_result: validate_solution返回的结果
            
        Returns:
            List[str]: 改进建议列表
        """
        suggestions = []
        
        if not validation_result['is_valid']:
            # 分析违规类型
            violation_types = {}
            for violation in validation_result['violations']:
                v_type = violation['type']
                violation_types[v_type] = violation_types.get(v_type, 0) + 1
            
            if 'missing_ground_duty' in violation_types:
                count = violation_types['missing_ground_duty']
                suggestions.append(f"需要为 {count} 个机组分配缺失的地面值勤")
            
            if 'invalid_ground_duty_assignment' in violation_types:
                count = violation_types['invalid_ground_duty_assignment']
                suggestions.append(f"需要修正 {count} 个错误的地面值勤分配")
            
            # 覆盖率建议
            if validation_result['coverage_rate'] < 0.80:
                missing_count = int((0.80 - validation_result['coverage_rate']) * validation_result['total_ground_duties']) + 1
                suggestions.append(f"需要额外分配至少 {missing_count} 个地面值勤以达到80%覆盖率")
            elif validation_result['coverage_rate'] < 0.95:
                missing_count = int((0.95 - validation_result['coverage_rate']) * validation_result['total_ground_duties']) + 1
                suggestions.append(f"建议额外分配 {missing_count} 个地面值勤以达到更高的95%覆盖率（可选）")
        
        return suggestions

def validate_ground_duty_solution(ground_duties: List[GroundDuty], rosters: List[Roster], 
                                crews: Optional[List[Crew]] = None) -> Tuple[bool, Dict]:
    """
    便捷函数：验证解决方案的地面值勤分配
    
    Args:
        ground_duties: 地面值勤列表
        rosters: 排班方案列表
        crews: 机组人员列表（可选）
        
    Returns:
        Tuple[bool, Dict]: (是否有效, 详细验证结果)
    """
    validator = GroundDutyValidator(ground_duties, crews)
    result = validator.validate_solution(rosters)
    return result['is_valid'], result

def print_ground_duty_summary(ground_duties: List[GroundDuty], rosters: List[Roster], 
                            crews: Optional[List[Crew]] = None) -> None:
    """
    便捷函数：打印地面值勤摘要
    
    Args:
        ground_duties: 地面值勤列表
        rosters: 排班方案列表
        crews: 机组人员列表（可选）
    """
    validator = GroundDutyValidator(ground_duties, crews)
    result = validator.validate_solution(rosters)
    
    print(validator.get_validation_report(result))
    
    if not result['is_valid']:
        print("\n改进建议:")
        for suggestion in validator.suggest_improvements(result):
            print(suggestion)

if __name__ == "__main__":
    # 测试代码
    print("地面值勤验证器测试")
    
    # 这里可以添加测试代码
    validator = GroundDutyValidator([])
    print("地面值勤验证器初始化完成")