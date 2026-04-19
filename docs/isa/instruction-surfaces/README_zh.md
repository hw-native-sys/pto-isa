# 指令集总览

PTO ISA 被组织成四类指令集，每类代表一种不同的机制、不同的操作数域和不同的执行路径。在阅读单条指令页之前，先理解这一层的分工非常重要。

## 总览

| 指令集 | 前缀 | 执行路径 | 主要职责 | 典型操作数 |
|--------|------|----------|----------|------------|
| [Tile 指令集](./tile-instructions_zh.md) | `pto.t*` | 通过 tile buffer 参与本地执行 | 面向 tile 的计算、数据搬运、布局变换、同步 | `!pto.tile<...>`、`!pto.tile_buf<...>`、`!pto.partition_tensor_view<...>` |
| [向量指令集](./vector-instructions_zh.md) | `pto.v*` | 向量流水线（PIPE_V） | lane 级向量微指令、mask、对齐状态、向量寄存器搬运 | `!pto.vreg<NxT>`、`!pto.mask`、`!pto.ptr<T, ub>` |
| [标量与控制指令集](./scalar-and-control-instructions_zh.md) | `pto.*` | 标量单元 / DMA / 同步外壳 | 配置、控制流、DMA、同步、谓词 | 标量寄存器、pipe/event 标识、buffer 标识、GM/UB 指针 |
| [其他指令集](./other-instructions_zh.md) | `pto.*` | 通信 / 运行时 / 跨 NPU | 集体通信、运行时支撑、别名与序列类辅助操作 | `!pto.group<N>`、tile 序列、分配句柄等 |

## 为什么要分成四类指令集

PTO 不是把所有 opcode 塞进一个扁平列表里，而是按架构可见状态来分层。原因很直接：tile、vector、scalar/control、communication 各自暴露的是不同类型的状态，如果把它们混成一层，会让 ISA 契约变得模糊。

| 指令集 | 核心抽象 | 主要职责 |
|--------|----------|----------|
| Tile（`pto.t*`） | tile：带 shape、layout、role、valid region 的架构可见对象 | GM ↔ tile 搬运、逐元素/归约/布局/matmul 运算、同步 |
| 向量（`pto.v*`） | vreg、谓词、向量可见 UB | 向量寄存器操作、lane 级 mask、UB ↔ vreg 搬运 |
| 标量与控制（`pto.*`） | 标量寄存器、pipe/event id、buffer id | 同步边、DMA 配置、谓词构造、控制流 |
| 其他（`pto.*`） | 集体组、tile 序列、分配句柄 | 集体通信、tile 序列操作、内存管理 |

## 指令级数据流关系

四类指令集共同组成 PTO 的执行层次：

```text
GM（片外全局内存）
        │
        ├── Tile 指令：TLOAD / TSTORE
        └── Vector 路径：copy_gm_to_ubuf / copy_ubuf_to_gm
        ▼
向量 tile buffer（硬件实现为 UB）
        │
        ├── Tile 指令：直接读写 tile buffer
        └── Vector 指令：vlds / vsts
        ▼
┌─────────────────┐              ┌─────────────────────────────┐
│  Tile Buffers   │              │  Vector Registers           │
│  (Vec/Mat/Acc/  │              │  !pto.vreg<NxT>            │
│   Left/Right)    │              │                             │
└────────┬─────────┘              └──────────────┬────────────┘
         │                                       │
         │  Tile 指令：pto.t*                  │  向量指令：pto.v*
         │  (TMATMUL 通过 Mat/Left/Right/Acc)  │  (vadd, vmul, vcmp, ...)
         │                                       │
         │  ◄── Matrix Multiply Unit            │  ◄── Vector Pipeline
         └─────────────────────────────────────┘
                       │
                       ▼
              [tile buffer → GM]
```

## 指令数量摘要

| 指令集 | 分组数 | 操作数量 | 说明 |
|--------|--------|----------|------|
| Tile | 8 | 约 120 | matmul、逐元素、归约、布局变换、数据搬运 |
| Vector | 9 | 约 99 | 完整向量微指令、加载存储、SFU |
| Scalar / Control | 6 | 约 60 | 同步、DMA、谓词、控制 |
| Other / Communication | 2 | 约 24 | 通信与支撑操作 |

## 规范语言

指令集页描述的是"这一组操作共同遵守什么契约"，不是逐条重复 opcode 说明。文中使用 **MUST / SHOULD / MAY** 时，应只用于 verifier、测试或 review 能够检查的规则；解释性内容应尽量用自然语言。

## 相关页面

- [指令族](../instruction-families/README_zh.md) — Tile / Vector / 标量 / 通信指令族
- [Tile 参考入口](../tile/README_zh.md) — Tile 指令逐条参考
- [Vector 参考入口](../vector/README_zh.md) — 向量指令逐条参考
- [标量与控制参考入口](../scalar/README_zh.md) — 标量与控制指令逐条参考
- [其他与通信参考入口](../other/README_zh.md) — 通信与支撑操作逐条参考
- [指令描述格式](../reference/format-of-instruction-descriptions_zh.md) — per-op 页面格式标准
