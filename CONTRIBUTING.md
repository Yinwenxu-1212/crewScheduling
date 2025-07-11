# 贡献指南 (Contributing Guide)

感谢您对机组排班优化系统项目的关注！我们欢迎各种形式的贡献。

## 🚀 快速开始

### 开发环境设置

1. **Fork 项目**
   ```bash
   git clone https://github.com/YOUR_USERNAME/crewScheduling.git
   cd crewSchedule_cg
   ```

2. **创建虚拟环境**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或
   venv\Scripts\activate     # Windows
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # 开发依赖
   ```

4. **安装 pre-commit 钩子**
   ```bash
   pre-commit install
   ```

## 📝 贡献类型

### 🐛 Bug 报告
- 使用 GitHub Issues 报告 bug
- 提供详细的复现步骤
- 包含错误信息和环境信息
- 使用 bug 标签

### ✨ 功能请求
- 在 Issues 中描述新功能
- 解释功能的用途和价值
- 提供使用场景示例
- 使用 enhancement 标签

### 📚 文档改进
- 修复文档错误
- 添加使用示例
- 改进 API 文档
- 翻译文档

### 🔧 代码贡献
- 修复 bug
- 实现新功能
- 性能优化
- 代码重构

## 🛠️ 开发流程

### 1. 创建分支
```bash
git checkout -b feature/your-feature-name
# 或
git checkout -b bugfix/issue-number
```

### 2. 编写代码
- 遵循项目代码规范
- 添加必要的测试
- 更新相关文档
- 确保代码通过所有检查

### 3. 提交代码
```bash
git add .
git commit -m "feat: 添加新功能描述"
# 或
git commit -m "fix: 修复bug描述"
```

### 4. 推送并创建 PR
```bash
git push origin your-branch-name
```

然后在 GitHub 上创建 Pull Request。

## 📋 代码规范

### Python 代码风格
- 遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 规范
- 使用 [Black](https://github.com/psf/black) 进行代码格式化
- 使用 [isort](https://github.com/PyCQA/isort) 排序导入
- 使用 [flake8](https://flake8.pycqa.org/) 进行代码检查

### 提交信息规范
使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**类型 (type):**
- `feat`: 新功能
- `fix`: bug 修复
- `docs`: 文档更新
- `style`: 代码格式化
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

**示例:**
```
feat(solver): 添加新的列生成算法实现

fix(data): 修复CSV文件读取编码问题

docs: 更新README中的安装说明
```

### 代码质量检查

运行以下命令确保代码质量：

```bash
# 代码格式化
black .
isort .

# 代码检查
flake8 .
mypy . --ignore-missing-imports

# 安全检查
bandit -r .

# 运行测试
pytest tests/ -v --cov
```

## 🧪 测试指南

### 编写测试
- 为新功能编写单元测试
- 确保测试覆盖率 > 80%
- 使用有意义的测试名称
- 测试边界条件和异常情况

### 测试结构
```
tests/
├── unit/           # 单元测试
├── integration/    # 集成测试
├── fixtures/       # 测试数据
└── conftest.py     # pytest 配置
```

### 运行测试
```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_solver.py

# 运行带覆盖率的测试
pytest --cov=. --cov-report=html
```

## 📖 文档规范

### 代码文档
- 使用 Google 风格的 docstring
- 为所有公共函数和类添加文档
- 包含参数、返回值和异常说明

```python
def calculate_score(roster: List[Duty], weights: Dict[str, float]) -> float:
    """计算排班方案的总得分。
    
    Args:
        roster: 排班方案列表
        weights: 各项指标的权重字典
        
    Returns:
        float: 总得分
        
    Raises:
        ValueError: 当权重字典缺少必要键时
    """
    pass
```

### Markdown 文档
- 使用清晰的标题结构
- 添加代码示例
- 包含必要的链接和引用
- 保持中英文对照

## 🔍 Pull Request 指南

### PR 标题
- 使用清晰描述性的标题
- 包含相关的 issue 编号
- 使用适当的标签

### PR 描述
包含以下内容：
- **变更摘要**: 简要描述所做的更改
- **相关 Issue**: 链接到相关的 issue
- **测试**: 描述如何测试这些更改
- **截图**: 如果有 UI 变更，提供截图
- **检查清单**: 确认所有必要步骤已完成

### PR 模板
```markdown
## 变更摘要
简要描述此 PR 的目的和主要变更。

## 相关 Issue
- Closes #123
- Related to #456

## 变更类型
- [ ] Bug 修复
- [ ] 新功能
- [ ] 文档更新
- [ ] 性能优化
- [ ] 代码重构

## 测试
描述如何测试这些更改：
- [ ] 单元测试通过
- [ ] 集成测试通过
- [ ] 手动测试完成

## 检查清单
- [ ] 代码遵循项目规范
- [ ] 添加了必要的测试
- [ ] 更新了相关文档
- [ ] 所有 CI 检查通过
```

## 🎯 开发最佳实践

### 代码组织
- 保持函数和类的单一职责
- 使用有意义的变量和函数名
- 避免深层嵌套和复杂逻辑
- 适当使用设计模式

### 性能考虑
- 避免不必要的计算和内存分配
- 使用适当的数据结构
- 考虑算法复杂度
- 进行性能测试和分析

### 错误处理
- 使用适当的异常类型
- 提供有用的错误信息
- 记录重要的错误和警告
- 优雅地处理边界情况

## 📞 获取帮助

如果您在贡献过程中遇到问题：

1. **查看文档**: 首先查看项目文档和 FAQ
2. **搜索 Issues**: 查看是否有类似问题已被讨论
3. **创建 Issue**: 如果找不到答案，创建新的 issue
4. **联系维护者**: 通过邮件或其他方式联系项目维护者

## 🏆 贡献者认可

我们重视每一个贡献，所有贡献者都会在项目中得到认可：

- 贡献者列表会在 README 中展示
- 重要贡献会在 CHANGELOG 中记录
- 优秀贡献者可能被邀请成为项目维护者

## 📄 许可证

通过贡献代码，您同意您的贡献将在与项目相同的 [MIT 许可证](LICENSE) 下发布。

---

再次感谢您的贡献！🎉