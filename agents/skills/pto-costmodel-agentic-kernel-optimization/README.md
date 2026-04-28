# AKO4PTO

这是一个用于 PTO 算子调优的多项目工作区。

目标：

- 不污染原始 `pto-kernels` 目录
- 不同算子使用独立项目目录
- 在各自项目的隔离副本中修改代码
- 通过远程 Ascend 环境做 benchmark 验证
- 对每一轮调优保留完整记录

## 建议阅读顺序

1. `README.md`
2. `task.md`
3. `remote.md`
4. `project_template/`

## 目录说明

- `projects/`：每个算子一个独立项目目录
- `project_template/`：新建算子项目时使用的模板
- `task.md`：所有项目共用的任务流程
- `remote.md`：共享的远程登录、同步、执行 benchmark 流程
- `context/`：可选的共享参考资料

## 最小使用流程

1. 为当前算子在 `projects/` 下创建一个独立项目目录。
2. 从 `project_template/` 复制 `TASK_REQUIREMENTS.md`、`ITERATIONS.md`、`workspace/`、`runs/`、`context/`。
3. 在该项目目录下将原始 `pto-kernels` 复制到 `projects/<operator_name>/workspace/pto-kernels`。
4. 具体执行统一按顶层 `task.md`，远程连接和环境配置按 `remote.md`。
5. 在该项目自己的 `projects/<operator_name>/workspace/pto-kernels` 中修改代码。
6. 每轮结束后，把记录写入该项目自己的 `runs/iter-XXX/`，并同步更新该项目自己的 `ITERATIONS.md`。

## 使用 Agent 自动寻优

如果使用 agent 自动执行调优，用户进入 `AKO4PTO` 根目录后，应直接让 agent 按顶层 `task.md` 行事。
