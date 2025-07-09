# 机组排班优化系统 (Crew Scheduling Optimization)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

基于列生成算法和注意力机制的智能机组排班优化系统，专为航空公司机组人员排班问题设计。

## ✨ 核心特性

- 🚀 **列生成算法**：高效求解大规模排班问题
- 🧠 **注意力机制**：AI引导的智能子问题求解
- 📊 **线性目标函数**：简化且稳定的覆盖率优化
- 🔧 **多约束处理**：支持复杂航空业务规则
- 📈 **实时监控**：详细的求解过程可视化

## 核心算法

### 1. 列生成框架
- 主问题：集合覆盖模型，选择最优排班组合
- 子问题：生成具有负reduced cost的新排班方案
- 迭代优化：直到无法找到改进方案

### 2. 注意力引导求解
- 深度学习模型预测最优决策
- Beam Search策略探索解空间
- 启发式剪枝提高求解效率

## 📁 项目结构

```
crewSchedule_cg/
├── 🚀 核心模块
│   ├── main.py                              # 主程序入口
│   ├── master_problem.py                    # 主问题求解器
│   └── attention_guided_subproblem_solver.py # AI引导子问题求解器
├── 📊 数据处理
│   ├── data_loader.py                       # 数据加载
│   ├── data_models.py                       # 数据模型
│   └── unified_config.py                    # 统一配置
├── 🔍 验证与评估
│   ├── coverage_validator.py                # 覆盖率验证
│   ├── ground_duty_validator.py             # 地面值勤验证
│   └── scoring_system.py                    # 评分系统
├── 🧠 AI模块
│   └── attention/                           # 注意力机制
│       ├── model.py                        # 神经网络模型
│       ├── environment.py                  # 强化学习环境
│       └── config.py                       # 模型配置
├── 📂 数据文件
│   └── data/                               # CSV数据文件
└── 📋 配置文件
    ├── requirements.txt                    # Python依赖
    └── .gitignore                         # Git忽略规则
```

## 🚀 快速开始

### 环境要求
- Python 3.8+
- Gurobi Optimizer (需要许可证)

### 安装依赖
```bash
# 克隆项目
git clone https://github.com/Yinwenxu-1212/crewScheduling.git
cd crewSchedule_cg

# 安装依赖
pip install -r requirements.txt
```

### 运行示例
```bash
# 直接运行（使用默认数据）
python main.py

# 查看结果
ls output/  # 查看生成的排班方案
```

### 输出文件
- `rosterResult_YYYYMMDD_HHMMSS.csv` - 最终排班方案
- `initial_solution.csv` - 初始解
- `optimization_*.log` - 求解日志

## ⚙️ 配置参数

### 核心算法参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MAX_ITERATIONS` | 10 | 列生成最大迭代次数 |
| `TIME_LIMIT` | 3600s | 求解时间限制 |
| `beam_width` | 10 | AI搜索束宽度 |
| `max_iterations` | 2000 | 子问题最大迭代次数 |

### 业务约束参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MAX_FLIGHT_HOURS` | 100h | 最大飞行小时数 |
| `MAX_DUTY_DAYS` | 20天 | 最大值勤天数 |
| `MIN_REST_TIME` | 12h | 最小休息时间 |

### 目标函数系数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `FLY_TIME_MULTIPLIER` | 50 | 飞行时间奖励系数 |
| `UNCOVERED_FLIGHT_PENALTY` | -500 | 未覆盖航班惩罚 |
| `POSITIONING_PENALTY` | -0.5 | 调机惩罚系数 |

> 💡 **提示**: 所有参数可在 `unified_config.py` 中修改

## 🎯 算法特性

### ✅ 核心优势
- **🚀 高效性**: 列生成算法处理大规模问题
- **🧠 智能性**: AI注意力机制提升求解质量  
- **🔧 灵活性**: 支持多种业务约束和目标
- **📈 稳定性**: 线性目标函数保证收敛
- **🛠️ 可维护性**: 模块化设计便于扩展

### 🎯 适用场景
- ✈️ 航空公司机组排班
- 👥 大规模人员调度
- 📊 资源分配优化
- 🏭 生产计划排程

## 🔧 开发指南

### 代码规范
- 遵循 PEP 8 编码规范
- 使用类型注解和文档字符串
- 模块化设计，职责分离

### 调试功能
- 详细的日志记录系统
- 中间结果自动保存
- 实时性能监控指标

### 扩展开发
```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 代码质量检查
pre-commit install
python scripts/check_code_quality.py

# 运行测试
pytest tests/ -v --cov
```

## 许可证

本项目仅供学术研究和教育用途使用。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 项目仓库: [https://github.com/Yinwenxu-1212/crewScheduling]
- 邮箱: [2151102@tongji.edu.cn]

---

*最后更新: 2025年*