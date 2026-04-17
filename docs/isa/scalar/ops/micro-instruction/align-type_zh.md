# PTO 微指令：对齐状态类型（`!pto.align`）

本页说明 `!pto.align` 类型及其相关的对齐状态操作。这些内容属于 PTO 微指令表面，当前主要对应 A5（Ascend 950）profile。

## 概览

`!pto.align` 建模的是 A5 向量对齐缓冲区的状态载体。它不是 payload 数据，而是在线性非对齐 load/store 序列中传递的对齐状态。

## 机制

`!pto.align` 把原本隐藏在硬件里的对齐状态显式搬到了 SSA 里。像 `pto.vldas` 或 `pto.init_align` 这样的操作负责产生初始状态；后续每一条非对齐 load/store 都消耗一个对齐状态并产出下一个状态。只有当这条状态链在线性序列里被正确传递时，这个流才是良构的。

## 输入

本页记录的是一种架构类型以及使用它的操作族。具体输入是下面各操作条目里出现的指针、偏移、向量值和对齐状态值。

## 预期输出

本页定义了 `!pto.align` 的契约，以及围绕它建立的流式约束。相关操作会产生新的对齐状态、消耗旧状态，或者同时处理 payload 和对齐状态。

## `!pto.align` 类型

`!pto.align` 是非对齐 load/store 家族使用的 SSA 对齐状态载体。PTO 微指令 IR 把它显式化，而不是依赖后端隐含状态。

### 关键性质

- `!pto.align` **不是** payload 类型；它只携带对齐状态。
- 它必须在线性的非对齐内存序列中被显式传递。
- 某些 store 序列在结尾仍然可能需要 flush 形式来提交尾部字节。
- 所有有状态的非对齐形式都以 SSA 结果的方式暴露状态更新。

## 对齐状态相关操作

### `pto.init_align`

**语法**：`%align = pto.init_align : -> !pto.align`

**语义**：初始化一个新的对齐状态。

```c
align = init_align();
```

### `pto.vldas`：为非对齐 load 预热对齐状态

**语法**：`%align = pto.vldas %ub : !pto.ptr<T, ub> -> !pto.align`

**语义**：为后续的非对齐 load 预热对齐状态。源地址周围的对齐块会作为后续 load 流的种子状态。

```mlir
%align = pto.vldas %ub : !pto.ptr<f32, ub> -> !pto.align
```

### `pto.vldus`：带对齐状态更新的非对齐 load

**语法**：`%vec, %align_out = pto.vldus %ub, %align : !pto.ptr<T, ub>, !pto.align -> !pto.vreg<NxT>, !pto.align`

**语义**：使用给定的对齐状态执行一次非对齐 load，并同时返回加载得到的向量与更新后的对齐状态。

```mlir
%vec, %align_out = pto.vldus %ub, %align : !pto.ptr<f32, ub>, !pto.align -> !pto.vreg<64xf32>, !pto.align
```

### `pto.vstus`：带对齐状态更新的非对齐 store

**语法**：`%align_out = pto.vstus %align, %offset, %vec, %ub : !pto.align, i32, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align`

**语义**：使用给定的对齐状态执行一次非对齐 store，并返回更新后的对齐状态。

```mlir
%store_align = pto.init_align : -> !pto.align
%next_align = pto.vstus %store_align, %offset, %vec, %ub
    : !pto.align, i32, !pto.vreg<64xf32>, !pto.ptr<f32, ub> -> !pto.align
```

## 完整的对齐状态流模式

下面的例子展示了一个完整的非对齐 load/store 流：

```mlir
// ─── Load 流 ───
%align0 = pto.vldas %ub_in : !pto.ptr<f32, ub> -> !pto.align
%v0, %align1 = pto.vldus %ub_in, %align0 : !pto.ptr<f32, ub>, !pto.align -> !pto.vreg<64xf32>, !pto.align
%v1, %align2 = pto.vldus %ub_in, %align1 : !pto.ptr<f32, ub>, !pto.align -> !pto.vreg<64xf32>, !pto.align

// ─── Compute ───
%result0 = pto.vabs %v0, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%result1 = pto.vabs %v1, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

// ─── Store 流 ───
%store_align0 = pto.init_align : -> !pto.align
%align_out1 = pto.vstus %store_align0, %c32, %result0, %ub_out : !pto.align, i32, !pto.vreg<64xf32>, !pto.ptr<f32, ub> -> !pto.align
%align_out2 = pto.vstus %align_out1, %c32, %result1, %ub_out : !pto.align, i32, !pto.vreg<64xf32>, !pto.ptr<f32, ub> -> !pto.align
```

## 约束

- `pto.vldas` 必须是非对齐 load 流的起始操作。
- `pto.vldus` 必须接在同一条对齐状态链上的 `pto.vldas` 之后。
- `pto.vstus` 必须以 `pto.init_align` 启动新的 store 对齐流。
- 对齐状态必须在线性流中传递，不能随意分叉。
- 对于 `pto.vstus`，`%offset` 控制每次 store 在流中的步进。

## 为什么要显式化对齐状态

把 `!pto.align` 暴露为 SSA 值有三个直接收益：

1. **正确性可验证**：编译器可以检查对齐状态是否被正确串接。
2. **调度可分析**：谁消费状态、谁产生状态，一目了然。
3. **IR 变换可推理**：中间变换不必依赖“硬件里还藏着一个状态机”这种隐含前提。

## 相关页面

- [向量加载存储](../../../vector/vector-load-store_zh.md)
- [向量执行作用域](./vecscope_zh.md)
