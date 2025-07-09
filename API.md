# API 文档 (API Documentation)

机组排班优化系统 API 参考文档

## 目录

- [核心模块](#核心模块)
- [数据模型](#数据模型)
- [求解器](#求解器)
- [工具类](#工具类)
- [配置](#配置)
- [使用示例](#使用示例)

## 核心模块

### main.py

主程序入口模块，包含完整的优化流程。

#### 函数

##### `main() -> None`

执行完整的机组排班优化流程。

**流程步骤：**
1. 数据加载与预处理
2. 初始解生成
3. 列生成算法优化
4. 整数规划求解
5. 结果验证与输出

**异常：**
- `SystemExit`: 当关键模块不可用或数据加载失败时

**示例：**
```python
from main import main

if __name__ == "__main__":
    main()
```

## 数据模型

### data_models.py

定义了系统中使用的所有数据结构。

#### 类

##### `Flight`

航班信息数据模型。

**属性：**
- `flightId: str` - 航班号
- `depAirport: str` - 起飞机场
- `arrAirport: str` - 到达机场
- `depTime: datetime` - 起飞时间
- `arrTime: datetime` - 到达时间
- `acType: str` - 机型
- `is_positioning: bool` - 是否为置位航班（可选）

**示例：**
```python
from datetime import datetime
from data_models import Flight

flight = Flight(
    flightId="CA1234",
    depAirport="PEK",
    arrAirport="SHA",
    depTime=datetime(2025, 1, 10, 8, 0),
    arrTime=datetime(2025, 1, 10, 10, 30),
    acType="A320"
)
```

##### `Crew`

机组人员信息数据模型。

**属性：**
- `crewId: str` - 机组ID
- `crewType: str` - 机组类型（如Captain, FirstOfficer）
- `baseAirport: str` - 基地机场

**示例：**
```python
from data_models import Crew

crew = Crew(
    crewId="CREW001",
    crewType="Captain",
    baseAirport="PEK"
)
```

##### `Roster`

排班方案数据模型。

**属性：**
- `crew_id: str` - 机组ID
- `duties: List[Union[Flight, GroundDuty, BusInfo]]` - 任务列表
- `cost: float` - 方案成本

**示例：**
```python
from data_models import Roster

roster = Roster(
    crew_id="CREW001",
    duties=[flight1, flight2],
    cost=150.0
)
```

##### `GroundDuty`

地面值勤任务数据模型。

**属性：**
- `id: str` - 任务ID
- `crewId: str` - 机组ID
- `airport: str` - 机场
- `startTime: datetime` - 开始时间
- `endTime: datetime` - 结束时间
- `task: str` - 任务描述

##### `BusInfo`

班车信息数据模型。

**属性：**
- `id: str` - 班车ID
- `depAirport: str` - 起点机场
- `arrAirport: str` - 终点机场
- `depTime: datetime` - 出发时间
- `arrTime: datetime` - 到达时间

##### `LayoverStation`

过夜站点信息数据模型。

**属性：**
- `airport: str` - 机场代码
- `isBase: bool` - 是否为基地

## 求解器

### master_problem.py

主问题求解器，实现集合覆盖模型。

#### 类

##### `MasterProblem`

主问题求解器类。

**方法：**

###### `__init__(flights, crews, ground_duties, layover_stations)`

初始化主问题求解器。

**参数：**
- `flights: List[Flight]` - 航班列表
- `crews: List[Crew]` - 机组列表
- `ground_duties: List[GroundDuty]` - 地面任务列表
- `layover_stations: List[LayoverStation]` - 过夜站点列表

###### `add_roster(roster: Roster, is_initial_roster: bool = False) -> None`

添加排班方案到主问题。

**参数：**
- `roster: Roster` - 排班方案
- `is_initial_roster: bool` - 是否为初始解

###### `solve_lp(verbose: bool = True) -> Tuple[Dict, Dict, Dict, float]`

求解线性规划松弛问题。

**参数：**
- `verbose: bool` - 是否输出详细信息

**返回：**
- `Tuple[Dict, Dict, Dict, float]` - (航班对偶价格, 机组对偶价格, 地面任务对偶价格, 目标函数值)

###### `solve_bip(verbose: bool = True) -> gurobipy.Model`

求解二进制整数规划问题。

**参数：**
- `verbose: bool` - 是否输出详细信息

**返回：**
- `gurobipy.Model` - Gurobi模型对象

###### `get_selected_rosters() -> List[Roster]`

获取选中的排班方案。

**返回：**
- `List[Roster]` - 选中的排班方案列表

### attention_guided_subproblem_solver.py

注意力引导的子问题求解器。

#### 函数

##### `solve_subproblem_for_crew_with_attention(...) -> List[Roster]`

使用注意力机制求解子问题。

**参数：**
- `crew: Crew` - 机组信息
- `flights: List[Flight]` - 可执行航班列表
- `pi_duals: Dict[str, float]` - 航班对偶价格
- `sigma_dual: float` - 机组对偶价格
- `ground_duties: List[GroundDuty]` - 地面任务列表
- `ground_duty_duals: Dict[str, float]` - 地面任务对偶价格
- `layover_stations: List[LayoverStation]` - 过夜站点列表
- `bus_info: List[BusInfo]` - 班车信息列表

**返回：**
- `List[Roster]` - 生成的排班方案列表

## 工具类

### scoring_system.py

排班方案评分系统。

#### 类

##### `ScoringSystem`

排班方案评分和成本计算系统。

**方法：**

###### `calculate_roster_cost(roster: Roster) -> float`

计算排班方案成本。

**参数：**
- `roster: Roster` - 排班方案

**返回：**
- `float` - 方案成本

###### `calculate_competition_score(rosters: List[Roster]) -> float`

计算竞赛评分（日均飞行时间）。

**参数：**
- `rosters: List[Roster]` - 排班方案列表

**返回：**
- `float` - 竞赛评分

### coverage_validator.py

覆盖率验证器。

#### 类

##### `CoverageValidator`

航班覆盖率验证器。

**方法：**

###### `validate_coverage(flights: List[Flight], rosters: List[Roster]) -> Dict`

验证航班覆盖率。

**参数：**
- `flights: List[Flight]` - 航班列表
- `rosters: List[Roster]` - 排班方案列表

**返回：**
- `Dict` - 验证结果字典

### ground_duty_validator.py

地面值勤验证器。

#### 类

##### `GroundDutyValidator`

地面值勤规则验证器。

**方法：**

###### `validate_solution(rosters: List[Roster], master_problem: MasterProblem) -> Dict`

验证地面值勤规则。

**参数：**
- `rosters: List[Roster]` - 排班方案列表
- `master_problem: MasterProblem` - 主问题对象

**返回：**
- `Dict` - 验证结果字典

## 配置

### unified_config.py

统一配置管理。

#### 类

##### `UnifiedConfig`

系统配置类，包含所有可配置参数。

**类属性：**

**算法参数：**
- `MAX_COLUMN_GENERATION_ITERATIONS: int = 10` - 最大列生成迭代次数
- `TIME_LIMIT_SECONDS: int = 3600` - 时间限制（秒）
- `CONVERGENCE_THRESHOLD: float = 1e-6` - 收敛阈值

**搜索参数：**
- `MAX_SUBPROBLEM_ITERATIONS: int = 2000` - 子问题最大迭代次数
- `BEAM_WIDTH: int = 10` - 束搜索宽度
- `MAX_CANDIDATES_PER_EXPANSION: int = 8` - 每次扩展的最大候选数

**业务约束：**
- `MAX_FLIGHT_HOURS: float = 100.0` - 最大飞行小时数
- `MAX_DUTY_DAYS: int = 20` - 最大值勤天数
- `MIN_REST_TIME: float = 12.0` - 最小休息时间（小时）

**惩罚系数：**
- `FLIGHT_TIME_REWARD: float = 50.0` - 飞行时间奖励系数
- `UNCOVERED_FLIGHT_PENALTY: float = -500.0` - 未覆盖航班惩罚
- `POSITIONING_PENALTY: float = -0.5` - 置位惩罚
- `AWAY_OVERNIGHT_PENALTY: float = -0.5` - 外站过夜惩罚
- `NEW_LAYOVER_PENALTY: float = -10.0` - 新增过夜站点惩罚
- `VIOLATION_PENALTY: float = -10.0` - 违规惩罚

**路径配置：**
- `DATA_PATH: str = "data"` - 数据文件路径
- `OUTPUT_PATH: str = "output"` - 输出文件路径
- `LOG_PATH: str = "log"` - 日志文件路径

## 使用示例

### 基本使用

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from main import main

if __name__ == "__main__":
    main()
```

### 自定义配置

```python
from unified_config import UnifiedConfig
from main import main

# 修改配置
UnifiedConfig.MAX_COLUMN_GENERATION_ITERATIONS = 20
UnifiedConfig.FLIGHT_TIME_REWARD = 100.0
UnifiedConfig.UNCOVERED_FLIGHT_PENALTY = -1000.0

# 运行优化
main()
```

### 单独使用模块

```python
from data_loader import load_all_data
from master_problem import MasterProblem
from scoring_system import ScoringSystem

# 加载数据
data = load_all_data("data")
flights = data["flights"]
crews = data["crews"]
ground_duties = data["ground_duties"]
layover_stations = data["layover_stations"]

# 创建主问题
master = MasterProblem(flights, crews, ground_duties, layover_stations)

# 创建评分系统
scoring = ScoringSystem(flights, crews, layover_stations)

# 计算方案成本
cost = scoring.calculate_roster_cost(roster)
```

### 验证解决方案

```python
from coverage_validator import CoverageValidator
from ground_duty_validator import GroundDutyValidator

# 验证航班覆盖率
validator = CoverageValidator(min_coverage_rate=0.8)
result = validator.validate_coverage(flights, rosters)
print(validator.get_coverage_report(result))

# 验证地面值勤规则
gd_validator = GroundDutyValidator(ground_duties, crews)
gd_result = gd_validator.validate_solution(rosters, master_problem)
print(gd_validator.get_validation_report(gd_result))
```

## 错误处理

### 常见异常

- `ImportError`: 缺少必要的依赖包
- `FileNotFoundError`: 数据文件不存在
- `ValueError`: 参数值不合法
- `RuntimeError`: 求解器运行时错误

### 调试信息

系统会在以下位置生成调试信息：
- `debug/` - 调试日志和中间结果
- `log/` - 系统运行日志
- `output/` - 最终结果文件

## 性能优化

### 参数调优建议

1. **减少迭代次数**：降低 `MAX_COLUMN_GENERATION_ITERATIONS`
2. **调整束搜索宽度**：根据问题规模调整 `BEAM_WIDTH`
3. **优化惩罚系数**：平衡各种约束的重要性
4. **设置时间限制**：避免过长的求解时间

### 内存优化

- 使用生成器处理大数据集
- 及时释放不需要的对象
- 监控内存使用情况

---

更多详细信息请参考源代码注释和README文档。