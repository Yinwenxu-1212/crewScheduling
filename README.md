# 机组排班优化系统 (Crew Scheduling Optimization)

基于列生成算法和Dinkelbach算法的机组排班优化系统，用于解决航空公司机组人员的排班问题。

## 项目概述

本项目实现了一个完整的机组排班优化解决方案，主要特点包括：

- **列生成算法**：高效求解大规模排班问题
- **Dinkelbach算法**：处理分数规划目标函数
- **注意力机制**：智能引导子问题求解
- **多约束处理**：支持复杂的航空业务规则

## 核心算法

### 1. 列生成框架
- 主问题：集合覆盖模型，选择最优排班组合
- 子问题：生成具有负reduced cost的新排班方案
- 迭代优化：直到无法找到改进方案

### 2. Dinkelbach算法
- 处理飞行时间与值勤天数的比值优化
- 参数化目标函数：max(1000×飞行时间 - λ×值勤天数)
- 自适应λ更新：确保算法收敛

### 3. 注意力引导求解
- 深度学习模型预测最优决策
- Beam Search策略探索解空间
- 启发式剪枝提高求解效率

## 项目结构

```
crewSchedule_cg/
├── main.py                              # 主程序入口
├── data_loader.py                       # 数据加载模块
├── data_models.py                       # 数据模型定义
├── master_problem.py                    # 主问题求解器
├── dinkelbach_optimizer.py              # Dinkelbach算法实现
├── subproblem_solver.py                 # 传统子问题求解器
├── attention_guided_subproblem_solver.py # 注意力引导子问题求解器
├── scoring_system.py                    # 排班方案评分系统
├── initial_solution_generator.py        # 初始解生成器
├── coverage_validator.py                # 覆盖率验证器
├── results_writer.py                    # 结果输出模块
├── attention/                           # 注意力模型相关
│   ├── model.py                        # 神经网络模型
│   ├── environment.py                  # 强化学习环境
│   ├── config.py                       # 模型配置
│   └── utils.py                        # 工具函数
├── data/                               # 输入数据
│   ├── flight.csv                     # 航班信息
│   ├── crew.csv                       # 机组信息
│   ├── crewLegMatch.csv               # 机组航段匹配
│   ├── busInfo.csv                    # 班车信息
│   ├── groundDuty.csv                 # 地面值勤
│   └── layoverStation.csv             # 过夜站点
├── models/                             # 预训练模型
└── requirements.txt                    # 依赖包列表
```

## 安装要求

### Python版本
- Python 3.8+

### 依赖包
```bash
pip install -r requirements.txt
```

主要依赖：
- `gurobipy`: 优化求解器
- `pandas`: 数据处理
- `numpy`: 数值计算
- `torch`: 深度学习框架
- `scikit-learn`: 机器学习工具

## 使用方法

### 1. 数据准备
将输入数据文件放置在 `data/` 目录下：
- `flight.csv`: 航班信息
- `crew.csv`: 机组信息
- `crewLegMatch.csv`: 机组航段匹配关系
- 其他辅助数据文件

### 2. 运行优化
```bash
python main.py
```

### 3. 查看结果
优化结果将保存在 `output/` 目录下：
- `rosterResult_YYYYMMDD_HHMMSS.csv`: 最终排班方案
- `initial_solution.csv`: 初始解

## 配置参数

### 算法参数
- `MAX_ITERATIONS`: 最大迭代次数 (默认: 100)
- `CONVERGENCE_THRESHOLD`: 收敛阈值 (默认: 1e-6)
- `TIME_LIMIT`: 时间限制 (默认: 3600秒)

### 业务约束
- `MAX_FLIGHT_HOURS`: 最大飞行小时数 (默认: 100)
- `MAX_DUTY_DAYS`: 最大值勤天数 (默认: 20)
- `MIN_REST_TIME`: 最小休息时间 (默认: 12小时)

### 惩罚系数
- `FLY_TIME_MULTIPLIER`: 飞行时间奖励系数 (默认: 1000)
- `UNCOVERED_FLIGHT_PENALTY`: 未覆盖航班惩罚 (默认: -5)
- `POSITIONING_PENALTY`: 定位惩罚 (默认: -0.5)
- `AWAY_OVERNIGHT_PENALTY`: 外站过夜惩罚 (默认: -0.5)

## 算法特性

### 优势
1. **高效性**: 列生成算法处理大规模问题
2. **智能性**: 注意力机制提升求解质量
3. **灵活性**: 支持多种业务约束和目标
4. **稳定性**: Dinkelbach算法保证收敛

### 适用场景
- 航空公司机组排班
- 大规模人员调度
- 资源分配优化
- 分数规划问题

## 开发说明

### 代码规范
- 遵循PEP 8编码规范
- 使用类型注解提高代码可读性
- 详细的文档字符串说明

### 调试功能
- 详细的日志记录
- 中间结果保存
- 性能监控指标

### 扩展性
- 模块化设计便于功能扩展
- 插件式求解器架构
- 可配置的约束和目标函数

## 许可证

本项目仅供学术研究和教育用途使用。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 项目仓库: [https://github.com/Yinwenxu-1212/crewScheduling]
- 邮箱: [2151102@tongji.edu.cn]

---

*最后更新: 2025年*