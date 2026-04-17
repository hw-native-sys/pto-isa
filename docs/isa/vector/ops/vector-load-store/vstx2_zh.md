# pto.vstx2

`pto.vstx2` 属于[向量加载与存储](../../vector-load-store_zh.md)指令集。

## 概述

双路交错存储，常用于 SoA → AoS 转换。

## 机制

`pto.vstx2` 属于 PTO 的向量内存 / 数据搬运指令。它把两路源向量按选定交错布局写回 UB。这里的关键语义不是“写两次”，而是“两个源向量构成有顺序的交错对”。

## 语法

### PTO 汇编形式

```text
vstx2 %low, %high, %dest[%offset], "DIST", %mask
```

### AS Level 1（SSA）

```mlir
pto.vstx2 %low, %high, %dest[%offset], "DIST", %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.ptr<T, ub>, index, !pto.mask
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

- 这条指令只对交错类分布合法。
- 两个源向量构成一个有序对，交错语义必须保留，不能在 lowering 里交换。

## 异常与非法情形

- 使用超出 UB 可见空间的地址，或违反所选分布模式的地址 / 对齐契约，都是非法的。
- 约束部分列出的额外非法情形，同样属于 `pto.vstx2` 的契约。

## 目标 Profile 限制

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

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vstx2`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vstx2` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[向量加载与存储](../../vector-load-store_zh.md)
- 上一条指令：[pto.vsts](./vsts_zh.md)
- 下一条指令：[pto.vsst](./vsst_zh.md)
