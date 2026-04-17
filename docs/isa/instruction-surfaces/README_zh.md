# 指令集总览

PTO ISA 被组织成四类指令集。每一类指令集都对应不同的机制、不同的操作数域，以及不同的执行路径。在阅读单条指令页之前，先理解这一层的分工非常重要。

## 总览

| 指令集 | 前缀 | 执行路径 | 主要职责 | 典型操作数 |
|--------|------|----------|----------|------------|
| [Tile 指令集](./tile-instructions_zh.md) | `pto.t*` | 通过 tile buffer 参与本地执行 | 面向 tile 的计算、数据搬运、布局变换、同步 | `!pto.tile<...>`、`!pto.tile_buf<...>`、`!pto.partition_tensor_view<...>` |
| [向量指令集](./vector-instructions_zh.md) | `pto.v*` | 向量流水线（PIPE_V） | lane 级向量微指令、mask、对齐状态、向量寄存器搬运 | `!pto.vreg<NxT>`、`!pto.mask`、`!pto.ptr<T, ub>` |
| [标量与控制指令集](./scalar-and-control-instructions_zh.md) | `pto.*` | 标量单元 / DMA / 同步外壳 | 配置、控制流、DMA、同步、谓词 | 标量寄存器、pipe/event 标识、buffer 标识、GM/UB 指针 |
| [其他指令集](./other-instructions_zh.md) | `pto.*` | 通信 / 运行时 / 跨 NPU | 集体通信、运行时支撑、别名与序列类辅助操作 | `!pto.group<N>`、tile 序列、分配句柄等 |

## 为什么要分成四类指令集

PTO 不是把所有 opcode 塞进一个扁平列表里，而是按架构可见状态来分层。原因很直接：tile、vector、scalar/control、communication 各自暴露的是不同类型的状态，如果把它们混成一层，会让 ISA 契约变得模糊。

### Tile 指令集（`pto.t*`）

Tile 指令围绕 **tile** 建模。tile 是带 shape、layout、role、valid region 元数据的架构可见对象。它们的主要职责是：

- 在 GM 和本地 tile buffer 之间搬运数据
- 在 tile 这一层执行逐元素运算、归约、扩展、布局变换和 matmul
- 建立 tile 级同步边

```text
输入：tile、标量修饰符、GlobalTensor 视图
输出：tile payload、valid region 变化、同步边
关注点：shape / layout / role / valid region / 目标 profile 收缩
```

### 向量指令集（`pto.v*`）

向量指令直接暴露向量流水线。它们处理的是向量寄存器、谓词寄存器和向量 tile buffer（硬件实现为 UB），而不是 tile 级 valid region。

```text
输入：vreg、标量、谓词、UB 指针、对齐状态
输出：vreg、谓词、UB 写回
关注点：lane、mask、对齐状态、distribution mode、目标 profile 限制
```

### 标量与控制指令集（`pto.*`）

标量与控制指令不直接产生 tile 或向量 payload。它们负责建立执行外壳：

- 同步与 producer-consumer 顺序
- DMA 配置与启动
- 谓词构造与谓词搬运
- 标量控制流与控制状态

```text
输入：标量值、pipe/event id、buffer id、DMA 参数
输出：控制状态、事件边、谓词、DMA 配置
关注点：顺序、配置、控制、可见状态
```

### 其他指令集（`pto.*`）

这一类用于放不能自然归入 tile / vector / scalar-control 的内容，例如：

- 通信与运行时
- 非 ISA 但仍与手册主线相关的支撑操作
- tile 序列、分配句柄等辅助结构

## 指令级数据流关系

四类指令集共同组成 PTO 的执行层次：

```text
┌─────────────────────────────────────────────────────────────┐
│  GM（片外全局内存）                                         │
└──────────┬──────────────────────────────────────┬───────────┘
           │                                      │
           │  Tile 指令：TLOAD / TSTORE                │
           │  Vector 路径：copy_gm_to_ubuf / copy_ubuf_to_gm
           ▼                                      ▼
┌─────────────────────────────────────────────────────────────┐
│  本地 tile buffer                                           │
│  其中 Vec tile buffer 的硬件实现就是 UB                      │
└──────┬──────────────────────────────────────────┬──────────┘
       │                                      │
       │  Tile 指令：直接读写 tile buffer           │
       │  Vector 指令：vlds / vsts                 │
       ▼                                      ▼
┌─────────────────┐              ┌─────────────────────────────┐
│  Tile Buffers   │              │  Vector Registers           │
│  !pto.tile_buf  │              │  !pto.vreg<NxT>            │
│  (Vec/Mat/Acc/  │              │                             │
│   Left/Right)   │              │                             │
└────────┬─────────┘              └──────────────┬────────────┘
         │                                     │
         │  Tile 指令：pto.t*                         │  向量指令：pto.v*
         │  (TMATMUL 通过 Mat / Left / Right / Acc) │  (vadd, vmul, vcmp, ...)
         │                                     │
         │  ◄── Matrix Multiply Unit           │  ◄── Vector Pipeline
         └─────────────────────────────────────┘
                       │
                       ▼
                [本地 tile buffer → GM]
```

## 指令数量摘要

| 指令集 | 分组数 | 操作数量 | 说明 |
|--------|--------|----------|------|
| Tile | 8 | 约 120 | matmul、逐元素、归约、布局变换、数据搬运 |
| Vector | 9 | 约 99 | 完整向量微指令、加载存储、SFU |
| Scalar / Control | 6 | 约 60 | 同步、DMA、谓词、控制 |
| Other / Communication | 2 | 约 24 | 通信与支撑操作 |

## 规范语言

指令集页描述的是“这一组操作共同遵守什么契约”，不是逐条重复 opcode 说明。文中使用 **MUST / SHOULD / MAY** 时，应只用于 verifier、测试或 review 能够检查的规则；解释性内容应尽量用自然语言而不是模板句。

## 相关页面

- [指令族](../instruction-families/README_zh.md)
- [指令描述格式](../reference/format-of-instruction-descriptions_zh.md)
- [Tile 参考入口](../tile/README_zh.md)
- [Vector 参考入口](../vector/README_zh.md)
- [标量与控制参考入口](../scalar/README_zh.md)
- [其他与通信参考入口](../other/README_zh.md)
