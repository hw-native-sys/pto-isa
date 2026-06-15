# 通用优化模式

这些是可复用模式，不是针对某个算子的 recipe。只有当 bottleneck evidence 支持时才应用。

## VF 编程上下文

在 Ascend CCE vector kernel 中，`__VEC_SCOPE__` 通常从 UB 载入 vector-sized block，使用 `vlds`，在 vector-side register context 中执行 vector operation，再用 `vsts` 存回。若多个操作可保留在同一个 VF scope 中，这可以减少反复 UB-to-vector movement。

硬件细节随目标变化，但常见优化信号包括：

- 当 active ISA 使用相应宽度时，vector block granularity 例如每个 operation group 256 bytes
- 很多 vector ALU 指令具备 dual EXU issue 潜力
- reduction、broadcast、transcendental、divide 或其他 long instruction 可能带来 EXU0-only 或 SFU-like 压力
- 简单且无 branch 的 loop form 可能符合 hardware-loop 条件

使用 active ISA/uarch evidence 判断精确 instruction class；不要从算子名推断目标行为。

## 优化优先级

除非 trace evidence 指向不同瓶颈，默认按以下顺序优先考虑：

1. remove redundant data movement
2. remove redundant computation
3. adjust VF fusion/split structure
4. improve algorithmic structure，减少 dependency depth、long operations、reductions 或 intermediate traffic
5. replace instruction patterns，避免过度使用 long、reduction 或 EXU0-only operations
6. restructure loops，通过 merge、split、unroll 或 hardware-loop-friendly forms
7. tune local instruction order inside one loop or stage

该顺序只是默认 search bias，不能替代每轮证据。

plateau 后，不要继续做 local instruction-order 或 parameter tuning，除非已经根据当前 trace evidence 重新检查过优先级 1-6。

structural optimization 指改变 dataflow、stage boundary、loop nest、VF scope layout 或 synchronization placement，使主执行结构发生变化。例如：

- 将 data movement 或 synchronization 跨 outer loop 移动
- 合批 work，使多个 VF 共享更少 copy/barrier phase
- 减少对同一数据的完整 pass 次数
- 改变 algorithmic dataflow，缩短 dependency depth 或减少 intermediate traffic
- 当 head/tail overhead、UB pressure 或 live range 证据支持时，merge 或 split VF scope
- 改写 loop nest，使能 hardware loop 或减少 scalar address work

local tuning 指不改变主 dataflow，只改变 instruction order、pragma hint、小 grouping factor 或 buffer placement。它应在 structural options 已检查后使用，或在 trace evidence 直接指向局部 stage 时使用。

## VF Fusion / Split

当大量小 VF 产生明显 launch/head-tail overhead，且 UB/register resource 能承受 fusion 时使用。

流程：

1. 建立 baseline VF count、`instr_num` 和 total window
2. 先测试保守 fusion
3. 监控 `instr_num`、per-VF execute、last-end drain 和 correctness
4. larger VF 回退时停止继续增大 fusion
5. 如果总问题规模不能被最佳 factor 整除，考虑 nonuniform grouping

不要假设 fusion 越大越好。过大的 VF 可能碰到 queue、register 或 drain 限制。

## Shared Barrier Grouping

当多个独立 store phase 后跟需要 store-to-load ordering 的 load phase 时使用。

模式：

1. 完成若干独立 store-producing segment
2. 发出一个必要的 `mem_bar(VST_VLD)` 或等价 barrier
3. 执行 load-consuming segments

要求：

- UB 区域不能错误重叠
- barrier type 必须匹配 ordering requirement
- reordering 后 correctness 必须通过
- 必须考虑跨 shared barrier 保持 live 的 values

## Reduction Restructuring

当 reduction instruction 频繁或 EXU0-only 压力高时使用。

候选形式：

- vector accumulation 后只做一次 final reduction
- 多个 independent accumulator 以 tree 合并
- 数学等价时，用 vector ops 替代重复 block reduction

风险：

- 更多 live accumulator 会增加 register pressure
- merge order 会改变 dependency depth
- numerical associativity 可能变化；必须满足 correctness tolerance

## Invariant Hoisting / Long-Op Replacement

当相同 expensive operation 在很多元素或 block 上重复出现时使用。

通用变换示例：

- invariant scale 只计算一次，再用更便宜的 multiply/add 应用
- 输入不变时，把 broadcast/setup hoist 到 loop 外
- 用一次 long operation 加 vector ALU ops 替代重复 long operation

要求：

- 证明 hoisted value 在变换 scope 内 invariant
- 确认 precision/rounding 仍可接受

## Loop Unroll Tuning

当 loop overhead 或 independent work 不足可能是瓶颈时使用。

原则：

- 一次只调一个 loop
- 优先尝试有界 factor，例如 1、2、4、8
- 如果 compiler 没有按预期 unroll，可将 manual unrolling 作为单独 candidate
- 监控 `instr_num`、register pressure 和 long-op clustering
- 多次持平/回退后，不要继续 unroll tuning

即使 instruction count 下降，unroll 也可能回退，因为 loop structure、queue behavior 和 issue spacing 可能变化。

## Loop Split / Fusion

horizontal split 可缩短 dependency chain，但可能增加 store 和 barrier。vertical split 可改善 awkward loop count 的 unroll compatibility。

证据满足以下情况时使用：

- long dependent stage 阻塞后续 independent work
- 一个 loop 中包含 optimal unroll/order 需求不同的 stages
- loop bound 不利于 unroll

如果拆分后的 loop 变成 memory-bound 或 barrier-heavy，避免 split。

## Hardware-Loop-Oriented Restructuring

当怀疑 loop structure 本身限制性能，而不仅是 loop 内 operation 限制性能时使用。

hardware-loop-friendly loop 通常要求：

- 简单 canonical loop form
- loop variable 从 0 开始，步长为 1
- static 或 simple bound，可被目标 compiler/hardware-loop mechanism 表示
- loop body 内无 branch、early exit 或 irregular control flow
- nesting depth 在目标 hardware-loop limit 内

候选变换：

- 多个重复 identical loop 合并为两层 nested loop，以减少 loop head overhead
- 当 nested-loop overhead 或 scalar offset calculation 主导时，将 nested loop flatten 成单层 loop
- 当不同 section 需要不同 unroll/order/fusion 选择时 split loop
- 将 loop variable 和 bound 规范化为目标友好的 integer type 和 form

风险：

- 看起来更简单的结构可能生成更差的 scalar address arithmetic
- flattening 可能掩盖 regular address pattern
- nesting 可能超过 hardware-loop limit 或增加 scalar offset work
- 移除 branch 可能需要 predication 或单独 boundary loop；必须仔细验证 correctness

## POST_UPDATE Addressing

当 vector load/store offset calculation 可能是 scalar 瓶颈时使用，尤其是较深 nested loop 或复杂 address expression。

通用变换模式：

```c
vlds(vec_x, data_x, computed_offset, NORM);
vsts(vec_x, data_y, computed_offset, NORM_B32, mask);
```

当 intrinsic 支持时，可变为 post-update 风格：

```c
vlds(vec_x, data_x, stride, NORM, POST_UPDATE);
vsts(vec_x, data_y, stride, NORM_B32, mask, POST_UPDATE);
```

原则：

- 使用 intrinsic 文档或 active CANN headers 确认精确语法和 pointer update 语义。
- 仅当 address progression 规则，且 pointer mutation 不破坏后续代码时使用。
- 对简单一层或两层 loop，offset computation 通常便宜，post-update 往往不收益。
- 对三层及以上 nested loop、多维 indexing 或重复 scalar offset expression，可能有收益。
- 将它作为单独 round hypothesis；不要和 loop restructuring 或 unroll change 混在同一轮。

风险：

- pointer state 成为 correctness 的一部分；必要时为每个 loop/segment reset 或使用独立 pointer
- 简单 loop 可能因 post-update overhead 或不利 scheduling 而回退
- 错误 stride 或 pointer lifetime 可能静默破坏 UB/GM addressing

## Instruction Reordering

当拓扑允许移动 independent instruction 来降低 long-op density、store-port tail 或 live range 时使用。

规则：

- 保持 producer/consumer dependency
- 保持必要 memory ordering
- 单轮只改变一个 local order dimension
- correctness failure 表示 dependency proof 错误或不完整

## Processing Order / Live-Range Tuning

当 fused VF 处理多个 independent segment，并保持多个 intermediate live 时使用。

候选方向：

- 先处理最近创建 live value 的 segment
- 先处理 live range 最长的 value
- 只有 evidence 表明 order matter 时，才测试 bounded permutation

当 permutation 持平或回退时停止；没有瓶颈理由时不要穷举搜索。
