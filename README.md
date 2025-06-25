# 机组排班优化系统 (Crew Scheduling Optimization System)

基于列生成算法和注意力机制的航空公司机组排班优化解决方案。

## 📋 项目概述

本项目是一个先进的机组排班优化系统，采用列生成算法结合强化学习和注意力机制，为航空公司提供高效的机组排班解决方案。系统能够在满足各种运营约束的前提下，最大化飞行时间利用率并最小化运营成本。

### 🎯 主要特性

- **列生成算法**: 采用经典的列生成方法求解大规模机组排班问题
- **注意力机制**: 集成基于Transformer的注意力模型指导子问题求解
- **智能约束处理**: 支持90%航班覆盖率约束和多种运营规则
- **实时监控**: 提供详细的求解过程监控和性能分析
- **灵活配置**: 支持多种参数调优和约束配置

## 🏗️ 系统架构

```
crewSchedule_cg/
├── main.py                              # 主程序入口
├── master_problem.py                    # 主问题求解器
├── subproblem_solver.py                 # 子问题求解器
├── attention_guided_subproblem_solver.py # 注意力引导的子问题求解器
├── data_loader.py                       # 数据加载模块
├── data_models.py                       # 数据模型定义
├── scoring_system.py                    # 评分系统
├── initial_solution_generator.py        # 初始解生成器
├── results_writer.py                    # 结果输出模块
├── attention/                           # 注意力模型模块
│   ├── config.py                        # 模型配置
│   ├── model.py                         # 神经网络模型
│   ├── environment.py                   # 强化学习环境
│   ├── main.py                          # 模型训练入口
│   └── utils.py                         # 工具函数
├── data/                                # 数据文件
├── output/                              # 输出结果
├── debug/                               # 调试日志
└── models/                              # 训练好的模型
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- Gurobi Optimizer 9.0+
- PyTorch 1.8+
- CUDA (可选，用于GPU加速)

### 安装依赖

```bash
pip install gurobipy torch pandas numpy datetime
```

### 数据准备

将以下CSV文件放置在 `data/` 目录下：

- `flight.csv` - 航班信息
- `crew.csv` - 机组信息
- `crewLegMatch.csv` - 机组航段匹配
- `layoverStation.csv` - 过夜站点信息
- `busInfo.csv` - 班车信息
- `groundDuty.csv` - 地面任务信息

### 运行系统

```bash
python main.py
```

## 📊 核心算法

### 列生成算法

系统采用经典的列生成方法：

1. **主问题**: 在已有roster集合中选择最优组合
2. **子问题**: 为每个机组生成新的可行roster
3. **迭代优化**: 通过对偶价格指导新列生成

### 注意力机制

集成了基于Transformer的注意力模型：

- **Actor-Critic架构**: 策略网络和价值网络
- **多头注意力**: 捕获航班间的复杂依赖关系
- **强化学习训练**: PPO算法优化决策策略

### 约束处理

- **机组唯一性**: 每个机组最多分配一个roster
- **时间窗约束**: 满足航班时间和机组可用性
- **规则约束**: 符合民航局相关规定

## 📈 评分系统

系统采用多维度评分机制：

- **飞行时间得分**: 值勤日日均飞时 × 1000
- **未覆盖航班惩罚**: 未覆盖航班数 × (-5)
- **过夜站点惩罚**: 新增过夜站点数 × (-10)
- **外站过夜惩罚**: 外站过夜天数 × (-0.5)
- **置位惩罚**: 置位次数 × (-0.5)
- **违规惩罚**: 违规次数 × (-10)

## 🔧 配置选项

### 主要参数

- `TIME_LIMIT_SECONDS`: 求解时间限制
- `MAX_ITERATIONS`: 最大迭代次数
- `MIN_COVERAGE_RATIO`: 最小航班覆盖率 (默认0.77)
- `UNCOVERED_FLIGHT_PENALTY`: 未覆盖航班惩罚系数

### 注意力模型参数

- `LEARNING_RATE`: 学习率 (3e-5)
- `GAMMA`: 折扣因子 (0.998)
- `PPO_EPSILON`: PPO裁剪参数 (0.2)
- `ENTROPY_COEF`: 熵系数 (0.01)

## 📁 输出文件

- `rosterResult.csv`: 最终排班结果
- `initial_solution.csv`: 初始解
- `roster_cost_debug_*.log`: 调试日志
- `debug_rosters_*.csv`: 调试用roster信息

## 🔍 监控与调试

系统提供详细的运行监控：

- **实时进度**: 显示当前迭代和求解状态
- **性能指标**: 目标函数值、未覆盖航班数、求解时间
- **调试日志**: 详细的成本计算和约束检查信息
- **错误处理**: 智能的异常处理和恢复机制

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 项目Issues: [GitHub Issues](https://github.com/Yinwenxu-1212/crewScheduling)
- 邮箱: 2151102@tongji.edu.cn

## 🙏 致谢

感谢所有为本项目做出贡献的开发者和研究人员。

---

**注意**: 本项目仅用于学术研究和教育目的。在生产环境中使用前，请确保充分测试并符合相关法规要求。