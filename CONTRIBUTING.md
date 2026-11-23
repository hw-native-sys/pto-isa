# 贡献指南

本项目欢迎广大开发者体验并参与贡献。在参与社区贡献之前，请参见 [cann-community](https://gitcode.com/cann/community) 了解社区行为准则、完成 CLA 签署流程，并熟悉源码仓的贡献方式。

本文档分为以下 3 个部分：

1. **常见贡献场景**：了解哪些类型的修改适合参与贡献，以及如何从 Issue 开始。
2. **本地开发**：准备本地实现、交付文件和本地检查。
3. **提交与合入 PR**：完成提交前检查、提交 PR，并跟进评审与合入流程。

## 常见贡献场景

### 算子 Bug 修复

如果您在本项目中发现某些算子实现存在缺陷，并希望对其进行修复，欢迎通过 Issue 进行反馈、跟踪与处理。

您可以参考 [提交 Issue / 处理 Issue 任务](https://gitcode.com/cann/community#提交Issue处理Issue任务) 创建 `Bug-Report|缺陷反馈` 类 Issue，对问题进行描述，然后在评论区输入 `/assign` 或 `/assign @yourself` 将该 Issue 分配给自己处理。

### 算子优化

如果您对本项目中的某些算子实现有泛化增强或性能优化思路，并希望着手实现这些优化点，欢迎贡献相关改进。

您可以参考 [提交 Issue / 处理 Issue 任务](https://gitcode.com/cann/community#提交Issue处理Issue任务) 创建 `Requirement|需求建议` 类 Issue，说明优化点并提供设计方案，然后在评论区输入 `/assign` 或 `/assign @yourself` 将该 Issue 分配给自己持续跟进。

### 贡献新算子

如果您希望基于 NPU 设计并实现一个全新的算子，欢迎在 Issue 中提出您的想法与设计方案。

#### 1. 新建 Issue

请按照 [提交 Issue / 处理 Issue 任务](https://gitcode.com/cann/community#提交Issue处理Issue任务) 指引，创建 `Requirement|需求建议` 类 Issue，并在其中说明新增算子的设计方案。

Issue 建议至少包含以下内容：

- **背景信息**
- **价值 / 作用**
- **设计方案**

同时，请在提交的 Issue 中评论 `/assign` 或 `/assign @yourself` 认领该任务，以便后续完成算子上库。

#### 2. 需求评审

SIG 成员会对您提交的 Issue 进行评审并反馈修改意见。完成修改后，请在 Issue 中回复：

> “完成意见修改，申请复审”

如果需求被接纳，SIG 成员会为您分配合适的算子分类路径（例如 `include/pto/npu/a5`），便于您将新增算子提交到对应目录。

如果在 Issue 交流中未能达成共识，建议申请在 SIG 双周例会上进一步讨论。

### 文档纠错

如果您发现算子文档中存在描述错误，欢迎通过 Issue 反馈并提交修复。

您可以参考 [提交 Issue / 处理 Issue 任务](https://gitcode.com/cann/community#提交Issue处理Issue任务) 创建 `Documentation|文档反馈` 类 Issue，指出对应文档的问题，然后在评论区输入 `/assign` 或 `/assign @yourself` 将该 Issue 分配给自己处理。

### 帮助解决他人 Issue

如果社区中他人遇到的问题您有合适的解决思路，欢迎在对应 Issue 中发表评论，帮助一起解决问题、优化易用性。

如果对应 Issue 需要代码修改，您也可以在评论区输入 `/assign` 或 `/assign @yourself` 将任务分配给自己，持续跟进并推动问题解决。

## 本地开发

方案确认后，就可以开始本地开发了。对于新增算子，交付内容可以在满足最小要求的基础上按需调整。下面给出一个更贴近当前仓库结构的参考布局，其中 `${op_name}` 为算子名称，`${op_class}` 为评审过程中分配的算子分类路径：

```text
    docs/
    ├── isa/
    │   └── ${op_name}.md                                # 算子说明文档
    include/
    └── pto/
        ├── common/
        │   ├── pto_instr_impl.hpp                       # 算子实现汇总
        │   └── pto_instr.hpp                            # 对外接口
        └── ${op_class}/
            └── ${op_name}.hpp                           # 算子实现文件、注释、结构与逻辑
    tests/
    ├── ${op_class}/src/st/testcase/
    │   ├── ${op_name}/
    │   │   ├── ${op_name}.cpp                           # 调用接口文件
    │   │   ├── main.cpp                                 # 测试入口文件
    │   │   ├── gen_data.py                              # 输入 / 期望结果生成脚本
    │   │   └── CMakeLists.txt                           # 构建文件
    │   └── CMakeLists.txt                               # 测试集合构建文件
    ├── run_st.sh                                        # ST 执行脚本入口
    └── README.md                                        # 测试说明
```

在准备提交 PR 之前，建议先完成以下本地开发检查：

- 交付文件是否完整，是否包含 ST 测试用例
- 代码是否符合 `.clang-format` 与 `pyproject.toml` 规范；提交前请使用 `clang-format -i -style=file <file>` 和 `ruff format <file>` 进行格式修复

## 提交与合入 PR

在准备本地代码与提交 PR 时，请重点关注以下几点：

1. 提交 PR 时，请按照 PR 模板仔细填写本次变更的业务背景、目标和设计方案等信息。
2. 若您的修改不是简单的 bug 修复，而是涉及新增特性、新增接口、新增配置参数，或修改既有流程，建议先通过 Issue 进行方案讨论，以避免后续评审中出现返工或无法合入的情况。如果您不确定本次修改是否属于“简单的 bug 修复”，也建议优先提交 Issue 进行确认。

### 提交前本地检查

社区贡献者可以使用 `pre-commit` 能力在提交前执行本地检查。

#### 步骤 1：安装 `pre-commit` 框架

```bash
# 使用 pip（推荐）
pip install pre-commit

# 验证安装
pre-commit --version
# 输出: pre-commit 3.x.x
```

Windows 用户：请确保已安装 Python 和 pip。

#### 步骤 2：进入项目目录

```bash
cd /path/to/your/pto-isa

# 例如
cd d:\complianceRepo\CANN\pto-isa
```

#### 步骤 3：安装 Git Hooks

```bash
# 在项目根目录运行
pre-commit install
```

#### 步骤 4：验证安装（可选）

该命令用于验证 `pre-commit` hook 是否已正确安装并能正常执行。若检查通过，它会创建一个空提交，因此如果只是用于测试，验证完成后可以将该提交撤销。安装 `pre-commit` 后，后续执行 `git commit` 时会自动触发同样的检查。

```bash
# 使用空提交测试 hook
git commit --allow-empty -m "test pre-commit"
```

#### PR 提交与评审

在正式提交 PR 前，还请确认以下事项：

- PR 是否已关联对应 Issue
- 是否已完成 CLA 签署
- 是否通过评论中的 `compile` 指令触发 CI 检查，并根据 CI 结果修复问题；若 `codecheck` 存在误报，请联系 SIG 成员处理

门禁通过后，请在关联的 Issue 中回复：

> “该 Issue 关联的 PR：XXX，请尽快评审”

收到 SIG 成员的评审意见后，请完成所有修改并回复：

> “该 Issue 关联的 PR：XXX，已完成 PR 问题整改，请尽快评审”

Committer 检视通过后，Maintainer 会进行最终审核。确认无误后，将通过 `/lgtm` 和 `/approve` 标签合入 PR。
