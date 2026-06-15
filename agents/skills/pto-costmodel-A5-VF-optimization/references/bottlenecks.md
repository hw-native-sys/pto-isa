# 瓶颈分类

修改代码之前，先分类当前瓶颈。如果瓶颈未知，先收集更多证据，不要直接改源码。

## 常见类别

### 数据搬运

证据：

- 重复 GM/UB copy
- 简单 arithmetic 前后反复 UB load/store
- VF 内 compute density 低

候选方向：

- 合批 copy
- 复用 UB-resident data
- 减少冗余 store/load
- 当 UB/register 压力允许时融合 stage

### VF Dispatch / Head-Tail 开销

证据：

- 大量短 VF
- 总窗口主要由 VF 间 launch spacing 或 drain 主导
- 单 VF `instr_num` 较小

候选方向：

- VF fusion
- 将独立 segment 分组
- 减少不必要的 VF split

监控：

- `instr_num` 增长
- first-start/last-end 总时间
- per-VF execute 和 drain 行为

### Long Instruction / SFU 压力

证据：

- 大量 `vexp`、`vdiv`、`vln` 或目标相关 long/SFU op
- unroll 后 long-op clustering 明显

候选方向：

- invariant hoisting
- 用一次 setup 加廉价 op 替换重复 long op
- 如果拓扑允许，在 long op 之间穿插独立 short op

### Reduction / EXU0-Only 压力

证据：

- 大量 `vcadd`、`vcmax`、`vcmin`、`vdup` 或 ISA 标记的 EXU0-only op
- 串行 reduction chain

候选方向：

- vector accumulation 后做一次 final reduction
- tree reduction
- 减少最终 scalar/broadcast reduction 次数

### 依赖链

证据：

- loop-carried dependency 反复依赖同一个 vector register
- consumer wait 主导 issue window

候选方向：

- 多个独立 accumulator
- tree merge
- loop split 暴露独立工作

### Register 压力

证据：

- 更大 fusion/unroll 在指令数相近或更少时反而变慢
- first VF execute 或 drain 显著增长
- barrier 或 long stage 跨越了过多 live intermediate

候选方向：

- 降低 fusion/unroll factor
- 改变处理顺序，尽早释放 live value
- 如果 sync/movement 成本可接受，split stage

### UB 压力

证据：

- temporary buffer 较大
- UB 区域重叠导致 correctness failure
- fusion 需要新增 intermediate storage

候选方向：

- lifetime 结束后复用旧 UB 区域
- 降低 fusion factor
- 如果存储代价高于计算代价，重算 cheap value

### Barrier / Sync

证据：

- store-to-load phase 被 barrier 隔开
- 大量 `mem_bar`、`wait_flag`、`pipe_barrier`

候选方向：

- 证明 barrier 是否可移动或合并
- 多个独立 store 先完成，再使用一个 barrier
- 没有 ordering proof 和 correctness rerun 时，绝不删除 ordering

### Loop Structure / Hardware Loop

证据：

- loop 不是 hardware-loop-friendly 形式
- loop variable 或 bound 不够简单/静态，无法生成 hardware loop
- loop 内 branch/control flow 阻止 hardware loop
- 多个重复 loop 带来可避免的 loop head overhead
- 嵌套 loop 形态导致重复 scalar index/offset 计算

候选方向：

- 在语义允许时，把 loop 规范化为简单 canonical form
- 从 hot VF loop 中移除可避免 branch
- 根据哪种方式能减少 loop head overhead 和 scalar indexing，将重复 loop 改成 nested loop，或把 nested loop flatten 成单层 loop
- split 或 merge loop 暴露 hardware-loop-friendly 结构
- 当 scalar offset 计算复杂时，考虑 `POST_UPDATE` addressing

监控：

- address progression correctness
- scalar instruction overhead 和 VF start spacing
- 改写后是否仍映射到高效 hardware-loop 行为
- 额外 loop overhead 或不利 scheduling 导致的 regression

### Queue / Inflight / Issue 限制

证据：

- 更大 VF 或 unroll 增加 `instr_num` 并使总时间变差
- first VF execute time 激增
- 指令数减少但总时间变差，原因可能是 loop overhead 或 scheduling 变化

候选方向：

- 降低 VF size/unroll
- 搜索中等 fusion factor
- 保持 hardware-loop-friendly 结构
