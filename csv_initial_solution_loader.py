# file: csv_initial_solution_loader.py

import pandas as pd
import os
from typing import List, Dict, Set, Tuple
from data_models import Flight, Crew, GroundDuty, BusInfo, Roster, LayoverStation
from data_loader import load_all_data
from scoring_system import ScoringSystem
from datetime import datetime


class CSVInitialSolutionLoader:
    """从CSV文件加载初始解并统计新的layover_stations的类"""
    
    def __init__(self, data_path: str = './data/'):
        """
        初始化加载器
        
        Args:
            data_path: 数据文件夹路径
        """
        self.data_path = data_path
        self.all_data = None
        self.flights_dict = {}
        self.ground_duties_dict = {}
        self.bus_info_dict = {}
        
    def load_base_data(self) -> Dict:
        """加载基础数据"""
        print("正在加载基础数据...")
        self.all_data = load_all_data(self.data_path)
        
        if not self.all_data:
            raise ValueError("基础数据加载失败")
        
        # 构建任务字典，便于快速查找
        self.flights_dict = {flight.id: flight for flight in self.all_data["flights"]}
        self.ground_duties_dict = {gd.id: gd for gd in self.all_data["ground_duties"]}
        self.bus_info_dict = {bus.id: bus for bus in self.all_data["bus_info"]}
        
        print(f"基础数据加载完成: 航班{len(self.flights_dict)}个, "
              f"地面任务{len(self.ground_duties_dict)}个, "
              f"大巴任务{len(self.bus_info_dict)}个")
        
        return self.all_data
    
    def load_roster_from_csv(self, csv_file_path: str) -> List[Roster]:
        """
        从CSV文件加载排班方案并转换为Roster对象
        
        Args:
            csv_file_path: CSV文件路径，格式应为 ['crewId', 'taskId', 'isDDH']
            
        Returns:
            List[Roster]: 排班方案列表
        """
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV文件不存在: {csv_file_path}")
        
        if self.all_data is None:
            self.load_base_data()
        
        print(f"正在从CSV文件加载排班方案: {csv_file_path}")
        
        # 读取CSV文件
        roster_df = pd.read_csv(csv_file_path)
        
        # 验证CSV格式
        required_columns = ['crewId', 'taskId', 'isDDH']
        if not all(col in roster_df.columns for col in required_columns):
            raise ValueError(f"CSV文件格式错误，需要包含列: {required_columns}")
        
        print(f"CSV文件包含 {len(roster_df)} 条任务分配记录")
        
        # 按机组ID分组
        crew_tasks = roster_df.groupby('crewId')
        
        rosters = []
        task_not_found_count = 0
        
        for crew_id, crew_tasks_df in crew_tasks:
            duties = []
            
            # 为每个机组收集任务
            for _, row in crew_tasks_df.iterrows():
                task_id = row['taskId']
                is_ddh = int(row['isDDH']) if pd.notna(row['isDDH']) else 0
                
                # 查找任务对象
                task_obj = self._find_task_object(task_id)
                
                if task_obj is None:
                    task_not_found_count += 1
                    print(f"警告: 未找到任务 {task_id}")
                    continue
                
                # 如果是置位任务，标记任务对象
                if is_ddh == 1:
                    if hasattr(task_obj, 'type'):
                        task_obj.type = 'positioning'
                    else:
                        # 为任务对象添加type属性
                        setattr(task_obj, 'type', 'positioning')
                
                duties.append(task_obj)
            
            if duties:  # 只有当机组有任务时才创建Roster
                # 按时间排序任务
                duties.sort(key=lambda x: self._get_task_start_time(x))
                
                # 创建Roster对象，初始成本为0，后续可以重新计算
                roster = Roster(crew_id=crew_id, duties=duties, cost=0.0)
                rosters.append(roster)
        
        if task_not_found_count > 0:
            print(f"警告: 共有 {task_not_found_count} 个任务未在基础数据中找到")
        
        print(f"成功加载 {len(rosters)} 个机组的排班方案")
        return rosters

    def _find_task_object(self, task_id: str):
        """
        根据任务ID查找对应的任务对象

        Args:
            task_id: 任务ID

        Returns:
            任务对象 (Flight, GroundDuty, 或 BusInfo)
        """
        # 先尝试在航班中查找
        if task_id in self.flights_dict:
            return self.flights_dict[task_id]

        # 再尝试在地面任务中查找
        if task_id in self.ground_duties_dict:
            return self.ground_duties_dict[task_id]

        # 最后尝试在大巴任务中查找
        if task_id in self.bus_info_dict:
            return self.bus_info_dict[task_id]

        return None

    def _get_task_start_time(self, task):
        """
        获取任务的开始时间

        Args:
            task: 任务对象

        Returns:
            datetime: 开始时间
        """
        if isinstance(task, Flight):
            return task.std
        elif isinstance(task, GroundDuty):
            return task.startTime
        elif isinstance(task, BusInfo):
            return task.td
        else:
            return datetime.min

    def analyze_layover_stations_from_rosters(self, rosters: List[Roster]) -> Tuple[Set[str], Dict]:
        """
        从排班方案中分析并统计新的layover_stations

        Args:
            rosters: 排班方案列表

        Returns:
            Tuple[Set[str], Dict]: (新的layover_stations集合, 统计信息)
        """
        print("正在分析排班方案中的过夜机场...")

        # 原始layover_stations
        original_layover_stations = self.all_data["layover_stations"] if self.all_data else set()

        # 统计信息
        stats = {
            'original_layover_count': len(original_layover_stations),
            'crew_overnight_locations': {},
            'new_layover_stations': set(),
            'overnight_frequency': {},
            'crew_duty_day_analysis': {}
        }

        new_layover_stations = set(original_layover_stations)

        for roster in rosters:
            crew_id = roster.crew_id
            duties = roster.duties

            if not duties:
                continue

            # 分析机组的值勤日和过夜位置
            crew_analysis = self._analyze_crew_overnight_pattern(crew_id, duties)
            stats['crew_duty_day_analysis'][crew_id] = crew_analysis

            # 收集过夜位置
            overnight_locations = crew_analysis.get('overnight_locations', [])
            stats['crew_overnight_locations'][crew_id] = overnight_locations

            # 统计过夜位置频率
            for location in overnight_locations:
                if location not in stats['overnight_frequency']:
                    stats['overnight_frequency'][location] = 0
                stats['overnight_frequency'][location] += 1

                # 如果不在原始layover_stations中，添加到新的集合
                if location not in original_layover_stations:
                    new_layover_stations.add(location)
                    stats['new_layover_stations'].add(location)

        stats['new_layover_count'] = len(new_layover_stations)
        stats['added_layover_count'] = len(stats['new_layover_stations'])

        print(f"过夜机场分析完成:")
        print(f"  原始过夜机场数量: {stats['original_layover_count']}")
        print(f"  新增过夜机场数量: {stats['added_layover_count']}")
        print(f"  总过夜机场数量: {stats['new_layover_count']}")

        if stats['new_layover_stations']:
            print(f"  新增的过夜机场: {sorted(list(stats['new_layover_stations']))}")

        return new_layover_stations, stats

    def _analyze_crew_overnight_pattern(self, crew_id: str, duties: List) -> Dict:
        """
        分析单个机组的过夜模式

        Args:
            crew_id: 机组ID
            duties: 任务列表

        Returns:
            Dict: 分析结果
        """
        analysis = {
            'crew_id': crew_id,
            'total_duties': len(duties),
            'flight_duties': 0,
            'ground_duties': 0,
            'bus_duties': 0,
            'duty_days': [],
            'overnight_locations': [],
            'duty_day_count': 0
        }

        if not duties:
            return analysis

        # 按日期分组任务
        from collections import defaultdict
        daily_tasks = defaultdict(list)

        for duty in duties:
            start_time = self._get_task_start_time(duty)
            date = start_time.date()
            daily_tasks[date].append(duty)

            # 统计任务类型
            if isinstance(duty, Flight):
                analysis['flight_duties'] += 1
            elif isinstance(duty, GroundDuty):
                analysis['ground_duties'] += 1
            elif isinstance(duty, BusInfo):
                analysis['bus_duties'] += 1

        # 分析每个值勤日
        sorted_dates = sorted(daily_tasks.keys())
        analysis['duty_day_count'] = len(sorted_dates)

        for i, date in enumerate(sorted_dates):
            day_duties = daily_tasks[date]
            day_duties.sort(key=lambda x: self._get_task_start_time(x))

            day_analysis = {
                'date': date,
                'task_count': len(day_duties),
                'start_location': None,
                'end_location': None,
                'flight_tasks': [],
                'is_overnight_day': False
            }

            # 确定当天的开始和结束位置
            if day_duties:
                first_task = day_duties[0]
                last_task = day_duties[-1]

                day_analysis['start_location'] = self._get_task_start_location(first_task)
                day_analysis['end_location'] = self._get_task_end_location(last_task)

                # 收集飞行任务信息
                for duty in day_duties:
                    if isinstance(duty, Flight):
                        day_analysis['flight_tasks'].append({
                            'id': duty.id,
                            'route': f"{duty.depaAirport}-{duty.arriAirport}",
                            'is_positioning': getattr(duty, 'type', '') == 'positioning'
                        })

            analysis['duty_days'].append(day_analysis)

            # 判断是否需要过夜
            if i < len(sorted_dates) - 1:  # 不是最后一天
                next_date = sorted_dates[i + 1]
                if (next_date - date).days == 1:  # 连续的日期
                    next_day_duties = daily_tasks[next_date]
                    if next_day_duties:
                        next_day_start_location = self._get_task_start_location(next_day_duties[0])

                        # 如果今天结束位置和明天开始位置相同，且不是基地，则可能是过夜
                        if (day_analysis['end_location'] == next_day_start_location and
                            day_analysis['end_location'] is not None):
                            day_analysis['is_overnight_day'] = True
                            analysis['overnight_locations'].append(day_analysis['end_location'])

        return analysis

    def _get_task_start_location(self, task) -> str:
        """获取任务的开始位置"""
        if isinstance(task, Flight):
            return task.depaAirport
        elif isinstance(task, GroundDuty):
            return task.airport
        elif isinstance(task, BusInfo):
            return task.depaAirport
        return None

    def _get_task_end_location(self, task) -> str:
        """获取任务的结束位置"""
        if isinstance(task, Flight):
            return task.arriAirport
        elif isinstance(task, GroundDuty):
            return task.airport
        elif isinstance(task, BusInfo):
            return task.arriAirport
        return None

    def save_new_layover_stations(self, new_layover_stations: Set[str],
                                 output_path: str = None) -> str:
        """
        保存新的layover_stations到CSV文件

        Args:
            new_layover_stations: 新的过夜机场集合
            output_path: 输出文件路径，如果为None则自动生成

        Returns:
            str: 保存的文件路径
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"output/new_layover_stations_{timestamp}.csv"

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 创建DataFrame并保存
        layover_df = pd.DataFrame({
            'airport': sorted(list(new_layover_stations))
        })

        layover_df.to_csv(output_path, index=False)
        print(f"新的layover_stations已保存到: {output_path}")

        return output_path

    def calculate_roster_costs(self, rosters: List[Roster]) -> List[Roster]:
        """
        使用评分系统重新计算排班方案的成本

        Args:
            rosters: 排班方案列表

        Returns:
            List[Roster]: 更新成本后的排班方案列表
        """
        if not self.all_data:
            raise ValueError("基础数据未加载")

        print("正在重新计算排班方案成本...")

        # 创建评分系统
        scoring_system = ScoringSystem(
            self.all_data["flights"],
            self.all_data["crews"],
            self.all_data["layover_stations"]
        )

        # 创建机组字典
        crews_dict = {crew.crewId: crew for crew in self.all_data["crews"]}

        updated_rosters = []
        for roster in rosters:
            crew = crews_dict.get(roster.crew_id)
            if crew:
                # 使用评分系统计算成本
                cost_details = scoring_system.calculate_roster_cost_with_dual_prices(
                    roster, crew, {}, 0.0
                )
                roster.cost = cost_details['total_cost']

            updated_rosters.append(roster)

        print(f"成本计算完成，共处理 {len(updated_rosters)} 个排班方案")
        return updated_rosters


def load_initial_solution_from_csv(csv_file_path: str, data_path: str = './data/',
                                  calculate_costs: bool = True,
                                  save_new_layovers: bool = True) -> Tuple[List[Roster], Set[str], Dict]:
    """
    便利函数：从CSV文件加载初始解并分析layover_stations

    Args:
        csv_file_path: CSV文件路径
        data_path: 数据文件夹路径
        calculate_costs: 是否重新计算成本
        save_new_layovers: 是否保存新的layover_stations到文件

    Returns:
        Tuple[List[Roster], Set[str], Dict]: (排班方案列表, 新的layover_stations集合, 统计信息)
    """
    print("=== 从CSV文件加载初始解并分析layover_stations ===")

    # 创建加载器
    loader = CSVInitialSolutionLoader(data_path)

    # 加载基础数据
    loader.load_base_data()

    # 从CSV加载排班方案
    rosters = loader.load_roster_from_csv(csv_file_path)

    # 分析layover_stations
    new_layover_stations, stats = loader.analyze_layover_stations_from_rosters(rosters)

    # 重新计算成本（可选）
    if calculate_costs:
        rosters = loader.calculate_roster_costs(rosters)

    # 保存新的layover_stations（可选）
    if save_new_layovers and stats['added_layover_count'] > 0:
        loader.save_new_layover_stations(new_layover_stations)

    # 打印详细统计信息
    print("\n=== 统计信息 ===")
    print(f"加载的排班方案数量: {len(rosters)}")
    print(f"原始过夜机场数量: {stats['original_layover_count']}")
    print(f"新增过夜机场数量: {stats['added_layover_count']}")
    print(f"总过夜机场数量: {stats['new_layover_count']}")

    if stats['new_layover_stations']:
        print(f"新增过夜机场: {sorted(list(stats['new_layover_stations']))}")

    # 过夜频率统计
    if stats['overnight_frequency']:
        print("\n过夜机场使用频率 (前10个):")
        sorted_freq = sorted(stats['overnight_frequency'].items(),
                           key=lambda x: x[1], reverse=True)[:10]
        for airport, freq in sorted_freq:
            marker = " (新增)" if airport in stats['new_layover_stations'] else ""
            print(f"  {airport}: {freq} 次{marker}")

    return rosters, new_layover_stations, stats


def save_analysis_report(stats: Dict, output_path: str = None) -> str:
    """
    保存详细的分析报告到文件

    Args:
        stats: 统计信息字典
        output_path: 输出文件路径

    Returns:
        str: 保存的文件路径
    """
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"output/layover_analysis_report_{timestamp}.txt"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=== Layover Stations 分析报告 ===\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("=== 总体统计 ===\n")
        f.write(f"原始过夜机场数量: {stats['original_layover_count']}\n")
        f.write(f"新增过夜机场数量: {stats['added_layover_count']}\n")
        f.write(f"总过夜机场数量: {stats['new_layover_count']}\n\n")

        if stats['new_layover_stations']:
            f.write("=== 新增过夜机场 ===\n")
            for airport in sorted(stats['new_layover_stations']):
                f.write(f"  {airport}\n")
            f.write("\n")

        f.write("=== 过夜机场使用频率 ===\n")
        sorted_freq = sorted(stats['overnight_frequency'].items(),
                           key=lambda x: x[1], reverse=True)
        for airport, freq in sorted_freq:
            marker = " (新增)" if airport in stats['new_layover_stations'] else ""
            f.write(f"  {airport}: {freq} 次{marker}\n")
        f.write("\n")

        f.write("=== 机组过夜模式分析 ===\n")
        for crew_id, analysis in stats['crew_duty_day_analysis'].items():
            f.write(f"\n机组 {crew_id}:\n")
            f.write(f"  总任务数: {analysis['total_duties']}\n")
            f.write(f"  值勤日数: {analysis['duty_day_count']}\n")
            f.write(f"  过夜位置: {analysis['overnight_locations']}\n")

    print(f"分析报告已保存到: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="从CSV文件加载初始解并分析layover_stations")
    parser.add_argument("csv_file", help="CSV文件路径，格式应为 ['crewId', 'taskId', 'isDDH']")
    parser.add_argument("--data-path", default="./data/", help="数据文件夹路径")
    parser.add_argument("--no-cost", action="store_true", help="不重新计算成本")
    parser.add_argument("--no-save", action="store_true", help="不保存新的layover_stations")
    parser.add_argument("--report", action="store_true", help="生成详细分析报告")
    parser.add_argument("--output", help="输出文件路径")

    args = parser.parse_args()

    # 加载初始解并分析
    rosters, new_layover_stations, stats = load_initial_solution_from_csv(
        args.csv_file,
        args.data_path,
        not args.no_cost,
        not args.no_save
    )

    # 生成详细报告（可选）
    if args.report:
        save_analysis_report(stats, args.output)

    # 如果指定了输出路径但没有生成报告，则保存新的layover_stations
    elif args.output and not args.no_save:
        loader = CSVInitialSolutionLoader(args.data_path)
        loader.save_new_layover_stations(new_layover_stations, args.output)

    print("\n处理完成！")
