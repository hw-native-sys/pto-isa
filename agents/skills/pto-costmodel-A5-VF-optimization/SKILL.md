---
name: pto-costmodel-A5-VF-optimization
description: 用于优化 Ascend A5 代际 CCE/PTO 向量算子或 VF 代码，要求按 correctness-first 的迭代轮次执行，可使用 PTO costmodel、trace、profiler 或静态指令证据进行瓶颈分析，并记录结构化性能日志。
metadata:
  short-description: 基于证据优化 A5 CCE/PTO VF 算子
license: CANN Open Software License Agreement Version 2.0
---

# PTO A5 VF 优化 Skill

当需要优化 Ascend A5 代际 CCE、PTO-ISA、DSL 或向量算子，并且必须保持正确性、性能结论必须由当前任务可用的测量链路支撑时，使用本 skill。除非当前任务明确提供，否则不要假设算子类型、tensor shape、数据流、仓库结构、运行器或 costmodel 一定存在。

## 核心流程

1. **先读任务本地文档和代码**
   - 阅读目标 kernel/DSL 和 workload 定义。
   - 如果任务没有提供可工作的 baseline，先构造 correctness-first baseline，再开始优化。
   - 如果仓库中存在 A5 ISA、machine、costmodel、profiler、runner 或测试文档，先阅读这些资料。
   - 如果仓库没有 costmodel 或 profiler，优先使用代码分析和 runner 证据，不要臆造 model-only 结论。

2. **建立测量契约**
   - 任务的 golden correctness runner 是强制要求。
   - 任务声明的 performance runner 是最终性能依据。
   - PTO costmodel、指令级 simulator trace、profiler 输出或静态指令证据只能作为形成假设的证据源，除非用户明确要求 model-only 分析。
   - 如果存在 CAModel 风格 VF dump 文件，多 VF 总时间统一按：`instr_log.dump` 中最后一个 VF 结束时间减去 `instr_popped_log.dump` 中第一个 VF 开始时间。
   - correctness validation 通过之前，不得给出任何有效性能结论。

3. **一轮一轮执行**
   - 每轮只验证一个主要 hypothesis。
   - 即使用户要求多轮优化，也不要提前安排多个无关轮次。
   - 当前轮次完成 build、correctness validation、performance parsing 和 round log 记录前，不得开启下一轮。
   - 如果 hypothesis 是某一个 loop/fusion factor/order 维度的参数调优，可以在提前声明候选集合后，把有限候选作为同一轮 bounded scan。

4. **记录每一轮**
   - 保留每一轮 candidate source。
   - 每轮结束后立即追加到 `perf_log.md` 或仓库等价日志。
   - 同时记录 `candidate vs previous round` 和 `candidate vs best-so-far`。

5. **根据证据切换方向**
   - 多轮持平或回退后，重读日志并切换 bottleneck class。
   - 当 local reorder/unroll 等局部尝试停止产生收益时，优先尝试结构性优化。
   - 每完成 10 轮优化后，必须重读本 `SKILL.md` 和相关 references，再提出下一轮。

## 何时读取哪些文档

- 开始优化前，或用户询问流程是否合规时，读取 [references/rules.md](references/rules.md)。
- 没有 baseline、需要生成 CCE/DSL 代码、或不确定 intrinsic/ISA 语法语义时，读取 [references/source_construction.md](references/source_construction.md)。
- 需要确认 barrier 语义或 vector ISA 行为时，读取 [references/isa.md](references/isa.md)。
- 需要解析 trace/profiler 日志、报告 speedup 或画图时，读取 [references/metrics.md](references/metrics.md)。
- 选择优化 hypothesis 前，读取 [references/bottlenecks.md](references/bottlenecks.md)。
- 选择具体优化候选时，读取 [references/patterns.md](references/patterns.md)。

## 工具策略

- 优先使用仓库自带的 build、correctness、profiling、trace 和 costmodel 工具。
- 如果本 skill 所在仓库没有辅助脚本，只在必要时创建小型 task-local parser，并保持 metric 定义不变。
- 不要依赖其他机器、其他仓库中的路径、虚拟环境、运行目录或脚本。

## 不可违反的规则

- 不得为了制造 speedup 而弱化 correctness threshold、workload size、build flags 或 benchmark logic。
- build failed 或 correctness failed 的结果不得作为有效性能。
- 除非有明确 ordering proof 并重新验证 correctness，否则不得删除 `mem_bar`、`wait` 或 `set_flag`。
- 除非用户明确把 costmodel 或 simulator 定义为 benchmark，否则不得把其输出当作最终性能。
- 单轮不得混合无关修改。
