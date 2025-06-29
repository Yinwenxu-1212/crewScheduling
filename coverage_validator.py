# coverage_validator.py
# 航班覆盖率验证器

from typing import List, Dict, Set, Tuple
from data_models import Flight, Roster
import logging

class CoverageValidator:
    """航班覆盖率验证器
    
    验证排班方案是否满足航班覆盖率要求
    """
    
    def __init__(self, min_coverage_rate: float = 0.8):
        """
        初始化验证器
        
        Args:
            min_coverage_rate: 最小覆盖率要求，默认0.8（80%）
        """
        self.min_coverage_rate = min_coverage_rate
        self.logger = logging.getLogger(__name__)
    
    def validate_coverage(self, flights: List[Flight], rosters: List[Roster]) -> Dict:
        """
        验证航班覆盖率
        
        Args:
            flights: 所有航班列表
            rosters: 排班方案列表
            
        Returns:
            Dict: 包含验证结果的字典
                {
                    'is_valid': bool,           # 是否满足覆盖率要求
                    'coverage_rate': float,     # 实际覆盖率
                    'total_flights': int,       # 总航班数
                    'covered_flights': int,     # 已覆盖航班数
                    'uncovered_flights': List,  # 未覆盖航班列表
                    'coverage_details': Dict    # 详细覆盖信息
                }
        """
        
        # 获取所有航班ID
        all_flight_ids = {flight.id for flight in flights}
        total_flights = len(all_flight_ids)
        
        # 获取已覆盖的航班ID
        covered_flight_ids = set()
        coverage_details = {}
        
        for roster in rosters:
            crew_id = roster.crew_id
            crew_covered_flights = []
            
            for duty in roster.duties:
                if isinstance(duty, Flight):  # 这是一个航班任务
                    flight_id = duty.id
                    covered_flight_ids.add(flight_id)
                    crew_covered_flights.append(flight_id)
            
            if crew_covered_flights:
                coverage_details[crew_id] = crew_covered_flights
        
        # 计算覆盖率
        covered_flights = len(covered_flight_ids)
        coverage_rate = covered_flights / total_flights if total_flights > 0 else 0.0
        
        # 找出未覆盖的航班
        uncovered_flight_ids = all_flight_ids - covered_flight_ids
        uncovered_flights = [flight for flight in flights if flight.id in uncovered_flight_ids]
        
        # 判断是否满足要求
        is_valid = coverage_rate >= self.min_coverage_rate
        
        # 记录日志
        self.logger.info(f"航班覆盖率验证结果: {coverage_rate:.2%} ({covered_flights}/{total_flights})")
        if not is_valid:
            self.logger.warning(f"覆盖率不足！要求: {self.min_coverage_rate:.2%}, 实际: {coverage_rate:.2%}")
            self.logger.warning(f"未覆盖航班数量: {len(uncovered_flights)}")
        
        return {
            'is_valid': is_valid,
            'coverage_rate': coverage_rate,
            'total_flights': total_flights,
            'covered_flights': covered_flights,
            'uncovered_flights': uncovered_flights,
            'uncovered_flight_ids': list(uncovered_flight_ids),
            'coverage_details': coverage_details,
            'min_required_rate': self.min_coverage_rate
        }
    
    def get_coverage_report(self, validation_result: Dict) -> str:
        """
        生成覆盖率报告
        
        Args:
            validation_result: validate_coverage返回的结果
            
        Returns:
            str: 格式化的覆盖率报告
        """
        result = validation_result
        
        report = []
        report.append("=== 航班覆盖率验证报告 ===")
        report.append(f"总航班数: {result['total_flights']}")
        report.append(f"已覆盖航班数: {result['covered_flights']}")
        report.append(f"未覆盖航班数: {len(result['uncovered_flights'])}")
        report.append(f"覆盖率: {result['coverage_rate']:.2%}")
        report.append(f"要求覆盖率: {result['min_required_rate']:.2%}")
        
        if result['is_valid']:
            report.append("✅ 验证通过：满足覆盖率要求")
        else:
            report.append("❌ 验证失败：覆盖率不足")
            report.append("\n未覆盖航班列表:")
            for flight in result['uncovered_flights'][:10]:  # 只显示前10个
                report.append(f"  - {flight.id}: {flight.std} {flight.depaAirport}->{flight.arriAirport}")
            if len(result['uncovered_flights']) > 10:
                report.append(f"  ... 还有 {len(result['uncovered_flights']) - 10} 个未覆盖航班")
        
        return "\n".join(report)
    
    def suggest_improvements(self, validation_result: Dict) -> List[str]:
        """
        根据验证结果提供改进建议
        
        Args:
            validation_result: validate_coverage返回的结果
            
        Returns:
            List[str]: 改进建议列表
        """
        suggestions = []
        
        if not validation_result['is_valid']:
            coverage_gap = self.min_coverage_rate - validation_result['coverage_rate']
            needed_flights = int(coverage_gap * validation_result['total_flights']) + 1
            
            suggestions.append(f"需要额外覆盖至少 {needed_flights} 个航班才能达到要求")
            suggestions.append("建议检查以下方面:")
            suggestions.append("1. 机组资质匹配 - 确保有足够的合格机组")
            suggestions.append("2. 时间约束 - 检查FDP和休息时间限制")
            suggestions.append("3. 地点衔接 - 优化机组的地理位置安排")
            suggestions.append("4. 任务分配策略 - 调整贪心算法的优先级")
            suggestions.append("5. 增加机组数量 - 如果约束允许")
            
            # 分析未覆盖航班的特征
            uncovered = validation_result['uncovered_flights']
            if uncovered:
                # 按机场分组
                airport_count = {}
                for flight in uncovered:
                    dep = flight.depaAirport
                    arr = flight.arriAirport
                    airport_count[dep] = airport_count.get(dep, 0) + 1
                    airport_count[arr] = airport_count.get(arr, 0) + 1
                
                # 找出问题机场
                problem_airports = sorted(airport_count.items(), key=lambda x: x[1], reverse=True)[:3]
                if problem_airports:
                    suggestions.append(f"\n重点关注机场: {', '.join([f'{airport}({count}次)' for airport, count in problem_airports])}")
        
        return suggestions

def validate_solution_coverage(flights: List[Flight], rosters: List[Roster], 
                             min_coverage_rate: float = 0.8) -> Tuple[bool, Dict]:
    """
    便捷函数：验证解决方案的航班覆盖率
    
    Args:
        flights: 航班列表
        rosters: 排班方案列表
        min_coverage_rate: 最小覆盖率要求
        
    Returns:
        Tuple[bool, Dict]: (是否有效, 详细验证结果)
    """
    validator = CoverageValidator(min_coverage_rate)
    result = validator.validate_coverage(flights, rosters)
    return result['is_valid'], result

def print_coverage_summary(flights: List[Flight], rosters: List[Roster], 
                          min_coverage_rate: float = 0.8) -> None:
    """
    便捷函数：打印覆盖率摘要
    
    Args:
        flights: 航班列表
        rosters: 排班方案列表
        min_coverage_rate: 最小覆盖率要求
    """
    validator = CoverageValidator(min_coverage_rate)
    result = validator.validate_coverage(flights, rosters)
    
    print(validator.get_coverage_report(result))
    
    if not result['is_valid']:
        print("\n改进建议:")
        for suggestion in validator.suggest_improvements(result):
            print(suggestion)

if __name__ == "__main__":
    # 测试代码
    print("航班覆盖率验证器测试")
    
    # 这里可以添加测试代码
    validator = CoverageValidator(0.8)
    print(f"最小覆盖率要求: {validator.min_coverage_rate:.2%}")