# pto.vbitsort

`pto.vbitsort` 属于[SFU 与 DSA 操作](../../sfu-and-dsa-ops_zh.md)指令集。

## 概述

按分数降序排序 32 个 proposal，并把排序后的记录写入目标缓冲区。典型用途是硬件加速的 top-K / NMS 预处理。

## 机制

`pto.vbitsort` 是一个 UB-to-UB 排序加速器。它从 `%src` 读取 32 个 score，从 `%indices` 读取对应原始索引，按分数**降序**排序后，把固定格式的记录写入 `%dest`。

### 输出记录格式

每条输出记录 8 字节：

| 字段 | 字节数 | 说明 |
|------|--------|------|
| 高 4 字节 | `[31:0]` | 原始索引 |
| 低 4 字节 | `[31:0]` | 分数值 |

对 `f16` 分数形式，分数字段只占低 2 字节，其余 2 字节保留。

### 排序规则

- 按 score **降序**
- 相同 score 时保持原始输入先后顺序，即稳定排序
- `%repeat_times` 控制连续处理多少组 32 元素块

## 语法

### PTO 汇编形式

```asm
vbitsort %dest, %src, %indices, %repeat_times : !pto.ptr<T, ub>, !pto.ptr<T, ub>, !pto.ptr<T, ub>, index
```

### AS Level 1（SSA）

```mlir
pto.vbitsort %dest, %src, %indices, %repeat_times
    : !pto.ptr<f32, ub>, !pto.ptr<f32, ub>, !pto.ptr<i32, ub>, index
```

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%dest` | `!pto.ptr<T, ub>` | 排序结果记录写回的 UB 目标缓冲区 |
| `%src` | `!pto.ptr<T, ub>` | 待排序分数所在的 UB 缓冲区 |
| `%indices` | `!pto.ptr<i32, ub>` | 原始索引缓冲区 |
| `%repeat_times` | `index` | 连续处理多少组 32 元素块 |

## 预期输出

- 这条指令没有 SSA 结果；它会直接写 UB
- 对 `f32`：记录格式是 `[index: u32][score: f32]`
- 对 `f16`：记录格式是 `[index: u32][score: f16][reserved: u16]`

## 副作用

这条指令会直接读写 UB 内存。

## 约束

- 排序方向固定为降序。
- 相同 score 的 tie 是稳定的。
- `%dest`、`%src`、`%indices` 都必须是 UB 指针。
- 对齐应满足 A5 `VBS32` 指令要求，否则结果可能落到软件回退路径。
- 每次基本调用处理 32 组 score/index 对，`%repeat_times` 把它扩展为 `32 × repeat_times`。

## 异常与非法情形

- 任一指针不是 UB 指针都属于非法。
- `%repeat_times` 若为 0 或负值则非法。

## 目标 Profile 限制

- 这是 A5 特有的排序加速能力（`VBS32`）。
- CPU 模拟器可能提供保留可见 PTO 语义的软件回退。
- 不适用于 A2/A3 profile。

## 性能

### A5 时延

当前手册没有给出 `VBS32` 的固定周期级条目，但它是专用排序硬件单元。

### A2/A3 吞吐

仓内 costmodel 为 `vbitsort` 记录了独立桶：

| 指标 | 数值 | 常量 |
|------|------|------|
| 启动时延 | 14 | `A2A3_STARTUP_BINARY` |
| 完成时延 | 14 | `A2A3_COMPL_DUP_VCOPY` |
| 每次 repeat 吞吐 | 4 | `A2A3_RPT_4` |
| 流水间隔 | 18 | `A2A3_INTERVAL` |

示例：排序 128 个 f32 元素（`repeatTimes = 4`）时，英文页给出的近似模型是：

```text
total ≈ 14 + 14 + 4×4 + 3×18 = 98 cycles
```

## 示例

### 基本 top-K / NMS 预处理

```mlir
pto.vbitsort %sorted_records, %score_buf, %idx_buf, %c1
    : !pto.ptr<f32, ub>, !pto.ptr<f32, ub>, !pto.ptr<i32, ub>, index
```

### 批量处理多个 32 元素块

```mlir
pto.vbitsort %dest0, %src0, %idx0, %c1 : !pto.ptr<f32, ub>, !pto.ptr<f32, ub>, !pto.ptr<i32, ub>, index
pto.vbitsort %dest1, %src1, %idx1, %c1 : !pto.ptr<f32, ub>, !pto.ptr<f32, ub>, !pto.ptr<i32, ub>, index
```

## 详细说明

`pto.vbitsort` 适合：

1. NMS：按置信度对 proposal 排序
2. Top-K：从候选集合里取最高分若干项
3. 固定批量排序加速：每次正好 32 元素的排序任务

它和 `vmrgsort` 的关系是：

- `vbitsort` 负责局部排序
- `vmrgsort` 负责归并已排序的段

两者可以串成“分块排序 → 多路归并”的流水。

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vbitsort`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vbitsort` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[SFU 与 DSA 操作](../../sfu-and-dsa-ops_zh.md)
- 上一条指令：[pto.vmrgsort](./vmrgsort_zh.md)
