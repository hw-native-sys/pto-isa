# pto.vci

`pto.vci` 属于[转换操作](../../conversion-ops_zh.md)指令集。

## 概述

根据一个标量种子生成索引向量。

## 机制

`pto.vci` 实际上是索引生成操作，而不是数值转换操作。它从标量种子 `%index` 出发，按 lane 递增或递减 1，生成一整向量的索引值。典型用途是为 gather/scatter 或 argsort 准备索引向量。

## 语法

### PTO 汇编形式

```asm
vci %index, %mask {order = "ORDER"} : !pto.vreg<Nxi32> -> !pto.vreg<Nxi32>
```

### AS Level 1（SSA）

```mlir
%indices = pto.vci %index {order = "ASC"} : i32 -> !pto.vreg<64xi32>
```

### AS Level 2（DPS）

```mlir
pto.vci ins(%index : i32) outs(%indices : !pto.vreg<64xi32>) {order = "ASC"}
```

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%index` | 标量 `i32` | 索引生成的种子或起始值 |
| `%mask` | `!pto.mask<G>` | 某些形式下可选的谓词掩码；inactive lane 可能写零或保留既有值 |

### 属性

| 属性 | 取值 | 说明 |
|------|------|------|
| `order` | `"ASC"` / `"DESC"` | 排序方向；`ASC` 生成递增索引，`DESC` 生成递减索引 |

## 预期输出

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%result` | `!pto.vreg<Nxi32>` | 生成出来的索引向量 |

### C 语义

```c
// ASC: base, base+1, base+2, ...
// DESC: base, base-1, base-2, ...
```

`%index` 是起始值。对于第 `i` 个 lane，结果是 `base + i`（ASC）或 `base - i`（DESC）。

## 副作用

这条指令除了产生 SSA 结果，没有其他架构副作用。它不会隐式占用 buffer、发送事件，也不会建立内存栅栏。

## 约束

- `%result` 使用整数元素类型，常见形式是 `i32`。
- 标量 `%index` 的类型应与结果元素类型匹配。
- 使用排序型索引生成时，`order` 属性是必须的。
- 标准形式中，lane 数 `N` 由结果类型推导。

## 异常与非法情形

- verifier 会拒绝非法的操作数形状、不支持的元素类型以及不合法的 `order` 属性。

## 目标 Profile 限制

- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。
- 当前文档化的 A5 profile 下，`pto.vci` 在 sampled `veccore0` trace 中不表现为标准 `RV_*` 向量算术指令，而更接近谓词 / 物化层实现。

## 性能

### 执行模型

`pto.vci` 在 `pto.vecscope` 内执行，但它更像索引物化，而不是主向量 ALU 计算。其成本通常由 mask 建立和逐 lane 简单算术主导。

### A2/A3 吞吐

`vci` 在 A2/A3 cost model 中不对应直接的 CCE 向量指令，通常会以标量索引生成循环形式实现：

| 指标 | 数值 | 说明 |
|------|------|------|
| 启动 | 约 10 周期 | mask 建立 + 循环开销 |
| 每元素成本 | O(1) | 简单逐 lane 算术 |
| 总复杂度 | O(N) | 每个输出 lane 一次操作 |

## 示例

### 生成递增索引

```mlir
%indices = pto.vci %c0 {order = "ASC"} : i32 -> !pto.vreg<64xi32>
// 结果：[0, 1, 2, 3, ..., 63]
```

### 生成递减索引

```mlir
%indices = pto.vci %c63 {order = "DESC"} : i32 -> !pto.vreg<64xi32>
// 结果：[63, 62, 61, 60, ..., 0]
```

### 与 gather 配合

```mlir
%idx = pto.vci %c0 {order = "ASC"} : i32 -> !pto.vreg<64xi32>
%data = pto.vgather2 %ub_table[%c0], %idx {dist = "DIST"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
```

### 与排序配合

```mlir
%indices = pto.vci %c0 {order = "ASC"} : i32 -> !pto.vreg<64xi32>
pto.vsort32 %sorted_indices, %indices, %config : !pto.ptr<i32, ub>, !pto.ptr<i32, ub>, i64
```

## 详细说明

`pto.vci` 的两个核心用途是：

1. 为 `vgather2` / `vscatter` 生成索引向量
2. 在 argsort 或间接排序前生成稳定索引键

对于同一个 `%index` 种子，它生成的索引是稳定可复现的，因此适合作为排序键或间接访问键。

## 相关页面

- 指令集总览：[转换操作](../../conversion-ops_zh.md)
- 上一条指令：[pto.vci](./vci_zh.md)
- 下一条指令：[pto.vcvt](./vcvt_zh.md)
- 相关索引生成：[pto.vsort32](../sfu-and-dsa-ops/vsort32_zh.md)
- 相关访问：[pto.vgather2](../vector-load-store/vgather2_zh.md)、[pto.vscatter](../vector-load-store/vscatter_zh.md)
