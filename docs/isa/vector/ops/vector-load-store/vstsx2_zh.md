# pto.vstsx2

`pto.vstsx2` 属于[向量加载与存储](../../vector-load-store_zh.md)指令集。

## 概述

双路交错存储，常用于 SoA → AoS 转换。

## 机制

`pto.vstsx2` 属于 PTO 的向量内存 / 数据搬运指令。它把两路源向量按选定交错布局写回 UB。这里的关键语义不是“写两次”，而是“两个源向量构成有顺序的交错对”。

## 语法

### PTO 汇编形式

```text
vstsx2 %low, %high, %dest[%offset], "DIST", %mask
```

### AS Level 1（SSA）

```mlir
pto.vstsx2 %low, %high, %dest[%offset], "DIST", %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.ptr<T, ub>, index, !pto.mask<G>
```

## 输入

- `%low`、`%high`：两路源向量
- `%dest`：UB 基址
- `%offset`：位移
- `DIST`：交错布局
- `%mask`：限定参与元素的谓词

## 预期输出

- 这条指令没有 SSA 结果；它会把交错流写入 UB

## 副作用

这条指令会写 UB 可见内存。某些有状态的非对齐流式形式还可能推进对齐状态，但尾部 flush 仍可能需要单独指令完成。

## 约束

!!! warning "约束"
    - 这条指令只对交错类分布合法。
    - 两个源向量构成一个有序对，交错语义必须保留，不能在 lowering 里交换。

## 异常与非法情形

!!! danger "异常与非法情形"
    - 使用超出 UB 可见空间的地址，或违反所选分布模式的地址 / 对齐契约，都是非法的。
    - 约束部分列出的额外非法情形，同样属于 `pto.vstsx2` 的契约。

## 目标 Profile 限制

??? info "目标 Profile 限制"
    - A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 示例

```c
// INTLV_B32
for (int i = 0; i < 64; i++) {
    UB[base + 8*i]     = low[i];
    UB[base + 8*i + 4] = high[i];
}
```

## 详细说明

### 支持的分布模式

`INTLV_B8`、`INTLV_B16`、`INTLV_B32`

## 性能

### 时延与吞吐披露

PTO-Gym v0.6 SPEC 为 A5 profile 上 `pto.vstsx2` 的 `INTLV` 分布族公布了统一的 12 周期时延。

| 指标 | 值 | 来源 |
|------|------|------|
| A5 时延（`INTLV`，所有元素宽度） | **12** 周期 | PTO-Gym v0.6 SPEC §III 向量加载/存储 |
| 稳态吞吐 | 未公开 | 当前公开 VPTO 时序材料 |

CPU 模拟和 A2/A3 类目标按 target-defined 处理；如需精确成本，请在具体 backend 实测，不要直接套用 A5 数值。

## 相关页面

- 指令集总览：[向量加载与存储](../../vector-load-store_zh.md)
- 上一条指令：[pto.vsts](./vsts_zh.md)
- 下一条指令：[pto.vsst](./vsst_zh.md)
