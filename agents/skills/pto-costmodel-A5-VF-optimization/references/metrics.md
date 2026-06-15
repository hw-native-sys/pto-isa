# 指标与 Trace 解析

## 权威边界

- **Golden correctness runner**：任何性能结论之前都必须通过。
- **任务声明的 benchmark runner**：最终性能依据。它可以是 CAModel、CPU simulator、NPU run、profiler 输出、仓库 costmodel 或其他 task-local runner。
- **PTO costmodel、指令级 simulator trace、profiler 输出和静态指令证据**：用于理解微架构、指令类别、依赖/issue 行为，并形成 hypothesis；除非用户明确将其定义为 benchmark。

除非用户明确要求 model-only 分析，或明确将某个 costmodel/simulator runner 定义为 benchmark，否则不要把 costmodel 或 simulator timing 当作最终 kernel performance。

## VF 总时间

若使用CAModel仿真对于 multi-VF trace，统一使用如下定义：

```text
VF total = core0.veccore0.instr_log.dump 中最后一个 VF end
         - core0.veccore0.instr_popped_log.dump 中第一个 VF start
```

这不同于累加 `vf_execute_time`，也不同于只看某一个 VF 的局部 latency。该定义覆盖从第一个 VF issue/pop 到最后一个 VF 完成的总窗口。

## Trace 文件

如果存在 CAModel 风格 VF dump 文件，典型文件包括：

- `core0.veccore0.instr_popped_log.dump`：VF start/pop cycle。
- `core0.veccore0.instr_log.dump`：VF completion cycle、`vf_execute_time`，通常也包含 `instr_num`。

如果仓库提供 parser，优先使用仓库 parser。否则编写小型 task-local parser，并记录精确解析规则。

## 必须记录的性能字段

至少记录：

- first VF start
- last VF end
- total VF latency 或任务声明的 latency/cycle metric
- VF 数量
- per-VF execute time，如果可用
- VF instruction count，如果可用
- benchmark reference，如果任务提供
- 精确 runner、command、device/model target 和相关 build flags

## 必须记录的正确性字段

使用任何性能数字之前，必须记录 task-local golden 结果。可用时包括：

- 最终 `PASS` / `FAIL` 状态
- `mismatches`
- `max_abs_err`
- `max_rel_err`
- runner 使用的未修改绝对/相对容差

只有任务原始 golden check 通过后，性能才有效。如果 runner 同时报告 PASS 标记和 mismatch/error 字段，都保留到日志中，便于后续轮次诊断精度漂移。

## Speedup 与 Benchmark Percent

对于 latency 类指标，越低越好。

```text
speedup_vs_baseline = baseline_cycles / candidate_cycles
performance_vs_benchmark_percent = benchmark_cycles / candidate_cycles * 100
```

如果 benchmark 是 latency 值，超过 100% 表示 candidate 快于 benchmark。

## 稳定性

如果仓库 timing 路径是确定性的，可以接受单次 run。如果 rebuild/rerun 之间 timing 波动，记录 noisy，并重复足够次数以确认代表性结果。

## 无效指标

以下情况下性能无效：

- build failed
- correctness failed
- trace 文件缺失或格式错误
- build flags 与测量契约不同
- workload 或 threshold 未经用户批准被修改
