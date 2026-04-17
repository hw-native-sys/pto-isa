# PTO 微指令参考

本节记录 PTO 微指令表面，主要对应 A5（Ascend 950）profile。这里的指令直接暴露向量流水线层面的状态：DMA 配置、向量寄存器、mask、同步，以及 `__VEC_SCOPE__` 边界。

> **说明**：这一层与 Tile ISA 不同。Tile 指令（`pto.t*`）围绕 tile、layout 和 valid region 建模；微指令则围绕向量寄存器（`vreg`）、mask 和标量状态建模。

## 指令分组

| 分组 | 说明 | 操作 |
|-------|------|------|
| [BlockDim 与运行时查询](./block-dim-query_zh.md) | 查询 block / subblock 编号与数量 | `pto.get_block_idx`、`pto.get_subblock_idx`、`pto.get_block_num`、`pto.get_subblock_num` |
| [指针操作](./pointer-operations_zh.md) | 构造类型化指针并做指针算术 | `pto.castptr`、`pto.addptr`、`pto.load_scalar`、`pto.store_scalar` |
| [向量执行作用域](./vecscope_zh.md) | 向量函数启动与作用域边界 | `pto.vecscope`、`pto.strict_vecscope` |
| [对齐状态类型](./align-type_zh.md) | 非对齐 load/store 的对齐状态管理 | `pto.init_align`、`pto.vldas`、`pto.vldus`、`pto.vstus` |

## 覆盖范围

PTO 微指令源程序并不只包含 `pto` 方言操作。在实践中，它还会配合共享的 MLIR 方言一起使用：

- **`arith`**：完整的标量 `arith` 表面，包括标量常量、算术、比较、select、cast、shift。见 [共享标量算术](../../shared-arith_zh.md)。
- **`scf`**：结构化控制流，包括 `scf.for`、`scf.if`、`scf.while`。见 [共享 SCF](../../shared-scf_zh.md)。

这些共享方言操作同样属于 PTO 微指令可接受的源表面，不应被看成 PTO 外部的“额外语言”。

## 机制

这一节不是单条 opcode 的说明页，而是整个微指令层的入口页。这里最核心的契约是：PTO 微指令把向量流水线可见的状态全部显式化。指针是带地址空间和元素类型的、mask 是一等 SSA 值、对齐状态通过 `!pto.align` 线程化传递、向量执行边界由 `pto.vecscope` / `pto.strict_vecscope` 明确划定，而不是依赖后端隐含状态。

## 输入

这张入口页本身没有自己的指令操作数。应把上面的分组看成继续深入阅读微指令表面的入口。

## 预期输出

本页定义的是微指令文档地图，以及理解微指令表面所需的架构概念。本页本身不会产生新的 SSA 值，也不会改变执行状态。

## 约束

- PTO 微指令表面是 profile 相关的；当前文档记录的是面向 A5 的可见表面。
- 微指令代码仍然会和标量 `arith` / `scf` 共同出现。
- 不应把微指令表面与 Tile ISA 混为一谈：两者的操作数模型、调度模型和状态载体不同。

## 与 PTO Tile ISA 的关系

| 维度 | PTO Tile ISA（`pto.t*`） | PTO 微指令 ISA（`pto.v*`、`pto.*`） |
|------|--------------------------|-------------------------------------|
| 抽象层级 | 以 tile 为中心，带 layout / valid region | 以向量寄存器、mask、标量状态为中心 |
| 操作数模型 | `!pto.tile<shape x type x layout>` | `!pto.vreg<NxT>`、`!pto.mask<G>` |
| 数据移动 | GM ↔ Tile（带 layout 变换） | 向量 tile buffer ↔ vreg、GM ↔ 向量 tile buffer（DMA） |
| 调度模型 | tile 级调度与融合 | 向量流水线调度、DAE |

## 关键架构概念

### VLane

在 A5 上，一个向量寄存器由 **8 个 VLane** 组成，每个 VLane 为 32 字节。VLane 是 group reduction 一类操作的原子归约单元。

```text
vreg（总宽度 256B）:
┌─────────┬─────────┬─────────┬─────┬─────────┬─────────┐
│ VLane 0 │ VLane 1 │ VLane 2 │ ... │ VLane 6 │ VLane 7 │
│   32B   │   32B   │   32B   │     │   32B   │   32B   │
└─────────┴─────────┴─────────┴─────┴─────────┴─────────┘
```

不同数据类型下，每个 VLane 对应的元素个数如下：

| 数据类型 | 每个 VLane 的元素数 | 每个 vreg 的总元素数 |
|----------|---------------------|----------------------|
| i8 / si8 / ui8 | 32 | 256 |
| i16 / si16 / ui16 / f16 / bf16 | 16 | 128 |
| i32 / si32 / ui32 / f32 | 8 | 64 |
| i64 / si64 / ui64 | 4 | 32 |

### Mask 类型

`mask<G>`：也就是 `!pto.mask<G>`，表示带粒度信息的谓词寄存器视图。`G` 取值为 `b8`、`b16`、`b32`，表示 VPTO 操作和 verifier 使用的字节粒度解释。

| Mask 类型 | 每个元素槽的字节数 | 常见元素族 | 导出的逻辑 lane 数 |
|-----------|--------------------|------------|--------------------|
| `!pto.mask<b32>` | 4 | `f32` / `i32` | 64 |
| `!pto.mask<b16>` | 2 | `f16` / `bf16` / `i16` | 128 |
| `!pto.mask<b8>` | 1 | 8-bit 元素族 | 256 |

### 内存层级

```text
┌─────────────────────────────────────────────┐
│               全局内存（GM）                │
│             片外 HBM / DDR                  │
└─────────────────────┬───────────────────────┘
                      │ DMA（MTE2 / MTE3）
┌─────────────────────▼───────────────────────┐
│   向量 tile buffer（硬件实现为 UB，256KB）   │
└─────────────────────┬───────────────────────┘
                      │ 向量加载 / 存储（PIPE_V）
┌─────────────────────▼───────────────────────┐
│           向量寄存器文件（VRF）              │
│      vreg（每个 256B）+ mask（256-bit）      │
└─────────────────────────────────────────────┘
```

### 谓词行为（Zero-Merge）

硬件原生 predication 采用 **ZEROING** 语义：inactive lane 产出 0。

```c
dst[i] = mask[i] ? op(src0[i], src1[i]) : 0
```

## 相关页面

- [Vector ISA 参考](../../../vector/README_zh.md)
- [标量与控制参考](../../README_zh.md)
- [流水线同步](../../pipeline-sync_zh.md)
- [DMA 拷贝](../../dma-copy_zh.md)
