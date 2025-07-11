# CSV初始解加载器使用说明

## 功能概述

这个模块提供了从CSV文件加载机组排班初始解并分析layover_stations（过夜机场）的功能。主要特性包括：

1. **从CSV文件加载排班方案**：支持标准的rosterResult.csv格式（crewId, taskId, isDDH）
2. **自动分析过夜机场**：从排班方案中识别新的过夜机场
3. **统计分析**：提供详细的过夜模式分析和使用频率统计
4. **成本重新计算**：使用评分系统重新计算排班方案的成本
5. **报告生成**：生成详细的分析报告

## 文件结构

- `csv_initial_solution_loader.py` - 主要功能模块
- `create_test_roster.py` - 测试数据生成工具
- `test_csv_loader.py` - 测试脚本
- `CSV_LOADER_README.md` - 本说明文档

## 快速开始

### 1. 准备CSV文件

CSV文件需要包含以下三列：
- `crewId`: 机组ID
- `taskId`: 任务ID
- `isDDH`: 是否为置位任务（0=执行任务，1=置位任务）

示例：
```csv
crewId,taskId,isDDH
Crew_10003,Flt_15787,0
Crew_10003,Flt_15783,0
Crew_10004,Bus_12345,1
```

### 2. 基本使用

#### 命令行使用：
```bash
# 基本加载和分析
python csv_initial_solution_loader.py your_roster.csv

# 生成详细报告
python csv_initial_solution_loader.py your_roster.csv --report

# 不重新计算成本
python csv_initial_solution_loader.py your_roster.csv --no-cost

# 指定数据路径
python csv_initial_solution_loader.py your_roster.csv --data-path ./data/
```

#### Python代码使用：
```python
from csv_initial_solution_loader import load_initial_solution_from_csv

# 加载初始解并分析
rosters, new_layover_stations, stats = load_initial_solution_from_csv(
    'your_roster.csv',
    data_path='./data/',
    calculate_costs=True,
    save_new_layovers=True
)

print(f"加载了 {len(rosters)} 个排班方案")
print(f"发现 {stats['added_layover_count']} 个新的过夜机场")
```

## 输出文件

### 1. 新的layover_stations文件
- 文件名：`output/new_layover_stations_YYYYMMDD_HHMMSS.csv`
- 格式：单列CSV文件，包含所有过夜机场（原有+新增）
- 用途：可以替换原始的`data/layoverStation.csv`文件

### 2. 分析报告
- 文件名：`output/layover_analysis_report_YYYYMMDD_HHMMSS.txt`
- 内容：详细的统计信息和机组过夜模式分析

## 测试功能

### 生成测试数据
```bash
python create_test_roster.py
```
这会生成：
- `test_rosterResult.csv` - 从现有ALNS结果转换的测试文件
- `simple_test_roster.csv` - 简单的测试数据

### 运行测试
```bash
python test_csv_loader.py
```

## 功能详解

### 1. 过夜机场分析算法

系统通过以下逻辑识别过夜机场：
1. 按机组和日期分组任务
2. 分析连续值勤日之间的位置连接
3. 如果今天结束位置 = 明天开始位置，且不是基地，则认为是过夜
4. 统计所有过夜位置的使用频率

### 2. 成本计算

使用项目的ScoringSystem重新计算每个排班方案的成本，考虑：
- 飞行时间成本
- 置位任务惩罚
- 过夜惩罚
- 新过站惩罚

### 3. 统计信息

提供以下统计数据：
- 原始vs新增过夜机场数量
- 过夜机场使用频率排序
- 每个机组的过夜模式分析
- 值勤日分布统计

## 实际应用场景

### 1. 初始解导入
```python
# 从外部优化器的结果导入初始解
rosters, _, _ = load_initial_solution_from_csv('external_solution.csv')

# 在ALNS算法中使用
initial_solution = ALNSSolution(rosters, flights, ground_duties, crews)
```

### 2. 过夜机场配置更新
```python
# 分析现有排班方案，更新过夜机场配置
_, new_layovers, stats = load_initial_solution_from_csv('current_roster.csv')

# 保存新的配置
loader = CSVInitialSolutionLoader()
loader.save_new_layover_stations(new_layovers, 'data/layoverStation.csv')
```

### 3. 排班方案评估
```python
# 评估不同排班方案的过夜机场使用情况
for roster_file in ['solution_A.csv', 'solution_B.csv']:
    _, _, stats = load_initial_solution_from_csv(roster_file)
    print(f"{roster_file}: {stats['added_layover_count']} 个新过夜机场")
```

## 注意事项

1. **数据一致性**：确保CSV文件中的taskId在基础数据中存在
2. **时间格式**：系统会自动处理时间格式转换
3. **内存使用**：大型排班方案可能需要较多内存
4. **NumPy兼容性**：可能会有NumPy版本警告，但不影响功能

## 错误处理

常见错误及解决方案：

1. **CSV格式错误**：确保包含必需的三列
2. **任务未找到**：检查taskId是否在基础数据中存在
3. **数据加载失败**：确认data/目录下有完整的数据文件
4. **成本计算失败**：检查ScoringSystem的依赖是否满足

## 扩展功能

可以通过继承CSVInitialSolutionLoader类来扩展功能：
- 自定义过夜机场识别逻辑
- 添加新的统计指标
- 支持其他CSV格式
- 集成到现有的优化流程中
