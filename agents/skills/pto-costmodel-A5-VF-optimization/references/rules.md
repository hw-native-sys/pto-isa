# A5 VF 通用优化规则

这些规则与具体算子无关。除非用户给出更严格的 task-specific rule，否则适用于任何 A5 代际 CCE、PTO-ISA、DSL 或 vector-kernel 优化任务。

## Baseline 要求

如果任务没有提供可工作的 baseline，先创建并验证 correctness-first baseline，再开始优化轮次。baseline 构造是独立阶段，不要和性能优化混在一起。PTO/CCE 查询和 baseline 构造规则见 `source_construction.md`。

## 固定 Build 与 Runner 契约

任何 build/run validation 都必须固定 runner、workload、correctness threshold、target device/model 和相关 build flags，让收益能归因到 candidate change。

对于直接 CCE VF 代码，如果任务测量 source-level scheduling change，且工具链支持以下选项，默认测量契约为：

- 必须开启 `-mllvm -cce-aicore-vec-misched=0`
- 必须关闭 VF fusion：`--cce-simd-vf-fusion=false`

当仓库 runner 不暴露这些 flags、代码是 PTO-ISA 而非直接 CCE VF、或用户明确指定不同 compiler-policy experiment 时，不要强制加入这些 flags。此时记录仓库实际 build flags，并在 baseline 和 candidate 之间保持不变。

## 轮次纪律

一轮是一个主要 hypothesis 及其验证。hypothesis 必须说明：

- 当前 bottleneck class
- bottleneck 证据
- 计划代码修改
- 预期 metric movement
- correctness argument
- 主要风险和需要监控的 failure signal

pre-edit hypothesis 明确前，不要编辑 candidate source。如果 bottleneck、证据、预期 metric movement 或 correctness argument 不清楚，先收集更多证据。

允许的单 hypothesis 示例：

- 一个 VF fusion/split 策略
- 一个 loop unroll 策略
- 一个 loop split/fusion 策略
- 一个 loop 或 stage 内的局部 instruction reorder 策略
- 一个 algorithmic instruction replacement 策略
- 一个有证明的冗余 synchronization 删除或 relocation

单轮禁止：

- 修改多个无关 stage
- 将 cleanup/refactor 与优化混合
- 修改 workload、threshold 或 benchmark script
- 不记录结果却静默保留 candidate
- 使用 correctness/build failed case 的性能数据

## Pre-Edit Checklist

每次源码修改前，写下：

- 当前 bottleneck class
- 该 bottleneck 的 trace 或 cost-model 证据
- 计划修改为何能解决该问题
- 任务声明 performance metric 的预期变化
- correctness 和 precision preservation 论证
- 主要风险和第一个需要关注的 failure signal

每个 candidate 后，将实际证据和预测对比。不要从未解决的“maybe”结论直接开始下一轮。

## 单轮内候选扫描（Candidate Scan）

只有当所有 candidate 都是同一 hypothesis 的变体时，才允许 bounded candidate scan。例如：

- 某一个 loop 的 unroll factor `{1,2,4,8}`
- 某一个 VF fusion factor family
- 某一个 independent processing order family

可行时，同一个局部 hypothesis 的变体应作为一轮 bounded scan，而不是拆成多轮。例如同一 stage 的 group2/group4、unroll2/unroll4、store-first/reduce-first。

round log 必须明确 chosen candidate，并拒绝其他 candidate。如果 candidate 暴露出不同 bottleneck，停止 scan，并开启新一轮。

## Plateau Reset

如果连续 4 个有效轮次对 best-so-far task latency/cycle metric 的累计提升低于 3%，停止局部调优，并在提出下一轮前重读优化 references。

reset 必须记录：

- 当前 best round 和 latency/cycle metric
- 这 4 轮累计 best improvement
- 已尝试和已拒绝的 hypotheses
- 当前 bottleneck classification
- 搜索是否被 local tuning 主导
- 下一步尝试的 structural direction

plateau reset 后，除非 trace evidence 强烈支持局部瓶颈，否则优先 structural change，而不是继续 instruction-order、pragma、buffer-placement 或 parameter tuning。

## Local-Tuning Budget

local tuning 包括 instruction reorder、仅 pragma 的 unroll change、等价 output grouping、不改变 dataflow 的 buffer placement swap，以及其他不改变 main dataflow 或 stage structure 的修改。

从同一个 base 开始，如果没有至少一轮产生可测 best-so-far improvement，不要连续运行超过 3 轮 local tuning。local-tuning budget 用尽后，切换到 structural hypothesis，或总结为什么当前没有可用 structural candidate。

如果 compiler 忽略或抵消了 pragma / scheduling hint，不要反复尝试等价 compiler-hint 变体。下一次相关尝试必须使用 manual expansion、loop-structure change，或带明确证据的不同 hypothesis。

## 正确性门禁（Correctness Gate）

对于生成的 baseline，记录任何 baseline performance 前也必须通过同样 correctness gate。

对每个 candidate：

1. build candidate
2. 运行 correctness/golden validation
3. correctness 通过后，才解析 performance
4. 追加 round log

如果仓库 runner 没有目标算子的 golden check，实现 task-local golden checker，但不得弱化 tolerance 或改变 workload。

当 runner 报告 `mismatches`、`max_abs_err`、`max_rel_err` 和 `PASS/FAIL` 等字段时，全部记录。严格使用 task-local tolerance；不得放宽 threshold 让 candidate 变 valid。如果 correctness 失败，先识别并记录原因，再把修复版本作为有效 candidate。

## Shape Contract 与有状态输出

优化 tiled、streaming、online 或 pipelined 算子前，先判断任务是 shape-specialized 还是 shape-general。

- **Shape-specialized**：如果用户明确只要求给定 fixed shape 或 tile count 正确，以该 shape 的端到端 correctness 作为语义契约。只要 correctness 仍通过，该 fixed shape 下端到端结果不需要的 state、store、FIFO 或 intermediate output 可以删除。
- **Shape-general**：如果用户要求通用算子、未来合法 shape，或没有明确将优化限制到一个 fixed shape，必须保持算子族语义。不得仅因为当前 benchmark shape 不消费，就删除 stateful store、running state、FIFO output 或 output-like 参数。
- 如果 shape-general benchmark 没有覆盖可能消费 state 的 multi-tile 或 downstream path，需要选择覆盖该路径的 review shape，并在保留 state-production、update-logic、FIFO 或 pipeline-synchronization 修改前运行 correctness。
- review shape timing 只作为语义护栏，不作为 benchmark，除非用户明确改变 benchmark contract。

## 每轮 CCE/PTO 检查清单

每个 candidate 前后检查：

- UB capacity 和 lifetime，包括目标特定限制，例如适用时 248 KiB UB
- dependency structure：没有无意引入 long chain、serialization 或改变 producer/consumer logic
- scheduling 和 dual-issue 行为：使用 IPC、issue gap 或 trace evidence，而不是只凭视觉上的 instruction parallelism
- synchronization：每个 `mem_bar`、`wait` 或 `set_flag` 都是必要的、有移动证明的，或明确保持不变
- resource side effects：识别 bottleneck movement，例如 data movement 转 sync、compute 转 register pressure、loop overhead 转 scalar address work

将相关检查记录到 round log；仅有更快但无法解释的 trace 不是完整 round result。

## Rollback 与 Search Policy

- 将每轮与前一个有效轮次比较，分类 positive/negative local movement。
- 另外单独维护 best-so-far。
- 如果两个连续相关 hypothesis 回退或持平且没有 insight，回滚到最后有用 base，并切换 sub-direction。
- 如果从同一个 base 连续三次未提升，优先尝试该 base round 中保存的 best rejected/second-best candidate，前提是它仍有 coherent hypothesis。
- 如果没有有用 second-best candidate，再回滚一个有效 base，重读 round log，并选择不同 bottleneck class。
- 当 rollback round 建立了未来可用的干净 base，应保留该记录。

## 可编辑范围

默认可编辑：

- 目标 CCE/DSL/PTO source
- 直接 kernel-generation helper scripts
- task-local validation 或 plotting scripts

默认只读，除非用户明确要求：

- costmodel implementation
- optimizer framework internals
- shared references 或 installed skills
- benchmark workload definitions
- correctness thresholds
- result parser semantics

## 必须记录的 Round Log 字段

每轮应记录：

- timestamp
- round id
- kernel/source path
- trace/run directory
- changed files
- current bottleneck
- evidence
- hypothesis
- planned change
- expected metric change
- main risk
- correctness result：mismatches、max_abs_err、max_rel_err
- costmodel/profiler/trace evidence used，如果有
- performance metric：任务声明 total，或可用时 CAModel-style first VF start、last VF end、total、per-VF execute、instr counts
- candidate vs previous round
- candidate vs best-so-far
- resource side effects：UB、register pressure、sync count、movement pattern
- decision：keep、reject、rollback、preserve
- next action

Round log 的轮次编号格式应保持一致，便于后续检索、解析和绘图。

## Commit Gate

只有同时满足以下条件时，才提交优化：

- 在未修改的 task-local golden check 下 correctness 通过
- 任务声明的 latency/cycle metric 有可复现提升
- source change 和 measured gain 有清晰因果解释
- build flags、workload、threshold 和 parser semantics 未变
- 代码可维护，足以支持后续轮次

如果仓库没有更严格约定，推荐 commit message 格式：

```text
[kernel_name] <optimization_type>: <cce_gain>
```
