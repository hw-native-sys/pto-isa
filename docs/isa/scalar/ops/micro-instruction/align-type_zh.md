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

### `pto.init_align`：初始化 store 端对齐状态载体

**语法**：`%result = pto.init_align : !pto.align`

**语义**：初始化 store 端对齐状态载体。

**输出**：`%result` 是一个全新的、零初始化的对齐状态载体，用于 **store 端** 的非对齐流，例如 `pto.vstus`、`pto.vstur`、`pto.vstar`、`pto.vstas` 和 `pto.pstu`。

**约束**：此操作只用于 store 端初始化。非对齐 load 流仍然必须从 `pto.vldas` 开始，而不是 `pto.init_align`。

```c
align = init_align();
```

### `pto.vldas`：为非对齐 load 预热对齐状态

**语法**：`%result = pto.vldas %source : !pto.ptr<T, ub> -> !pto.align`

**语义**：为后续的非对齐 load 预热对齐缓冲区。

**输入**：`%source` 是 UB 地址，其所在的对齐块用于作为 load 对齐状态的种子。

**输出**：`%result` 是初始化后的 load 端对齐状态。

**约束**：

- 此操作是同一对齐状态链上 `pto.vldus` 流的必备起始操作。
- 源地址本身不需要 32 字节对齐；硬件会将其截断到对齐块边界以执行预热 load。

**延迟**：**9** 周期。

```mlir
%align = pto.vldas %ub : !pto.ptr<f32, ub> -> !pto.align
```

### `pto.vldus`：带对齐状态更新的非对齐 load

**语法**：`%result, %align_out = pto.vldus %source, %align : !pto.ptr<T, ub>, !pto.align -> !pto.vreg<NxT>, !pto.align`

**语义**：使用预热好的对齐状态执行一次非对齐 load。

**输入**：`%source` 是当前 UB 地址；`%align` 是由 `pto.vldas` 或前一条 `pto.vldus` 产出的 load 端对齐状态。

**输出**：`%result` 是装配出的向量值；`%align_out` 是更新后的对齐状态。

**约束**：

- 同一向量循环中，第一条依赖的 `pto.vldus` 之前必须出现匹配的 `pto.vldas`。
- A5 no-post 接口在内部保留了一个结构体形式的返回值以便下沉，但其中的 `base` 字段不是用户可见的有意义状态；VPTO 在表面上隐藏该值，仅暴露更新后的对齐状态载体。
- 重新使用原始的 `%source` 会启动一个新的显式访问点；如果调用方希望再次进行 no-post 访问，应显式计算下一个源指针，并配上必需的对齐状态初始化。

**延迟**：**9** 周期。

```mlir
%vec, %align_out = pto.vldus %ub, %align : !pto.ptr<f32, ub>, !pto.align -> !pto.vreg<64xf32>, !pto.align
```

### `pto.vstus`：带标量偏移的 no-post 非对齐 store

**语法**：`%align_out = pto.vstus %align_in, %offset, %value, %base : !pto.align, i32, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align`

**语义**：带标量偏移的 no-post 非对齐 store。

**输入**：`%align_in` 是输入的 store 端对齐状态；`%offset` 是标量步长；`%value` 是被存储的向量；`%base` 是 UB 基址。

**输出**：`%align_out` 是更新后的缓冲尾部状态。

**约束**：

- 这是非对齐 store 家族里带标量偏移的有状态形式。同一个流的首个 `%align_in` 应来自 `pto.init_align`。
- 此指令 **不** 表示「从 `%base + %offset` 开始存储一整个向量」。相反，`%offset` 描述当前这步在流中前进多远，而 `%align_out` 携带尚未提交的尾部残量。
- no-post 表面不暴露已更新的基址指针。后续 flush 操作（`pto.vstas` / `pto.vstar`）必须显式使用与本条 `pto.vstus` 对应的目的地/偏移对，指明同一个逻辑 flush 点。

**延迟**：**9** 周期。

```mlir
%store_align = pto.init_align : !pto.align
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
%store_align0 = pto.init_align : !pto.align
%align_out1 = pto.vstus %store_align0, %c32, %result0, %ub_out : !pto.align, i32, !pto.vreg<64xf32>, !pto.ptr<f32, ub> -> !pto.align
%align_out2 = pto.vstus %align_out1, %c32, %result1, %ub_out : !pto.align, i32, !pto.vreg<64xf32>, !pto.ptr<f32, ub> -> !pto.align
```

## 约束

!!! warning "约束"
    - `pto.vldas` 必须是非对齐 load 流的起始操作。
    - `pto.vldus` 必须接在同一条对齐状态链上的 `pto.vldas` 之后。
    - store 端的非对齐流（`pto.vstus` 及相关的 `pto.vstur`、`pto.vstar`、`pto.vstas`、`pto.pstu`）必须由 `pto.init_align` 初始化。`pto.init_align` **只用于 store 端**，不能用来预热 load 流。
    - 对齐状态必须在线性流中传递，不能随意分叉。
    - 对于 `pto.vstus`，`%offset` 控制每一步在流中前进多远，而不是相对 `%base` 的绝对存储位移。后续 flush 操作（`pto.vstas` / `pto.vstar`）必须复用配套的目的地/偏移对。

## 为什么要显式化对齐状态

把 `!pto.align` 暴露为 SSA 值有三个直接收益：

1. **正确性可验证**：编译器可以检查对齐状态是否被正确串接。
2. **调度可分析**：谁消费状态、谁产生状态，一目了然。
3. **IR 变换可推理**：中间变换不必依赖“硬件里还藏着一个状态机”这种隐含前提。

## 相关页面

- [向量加载存储](../../../vector/vector-load-store_zh.md)
- [向量执行作用域](./vecscope_zh.md)
