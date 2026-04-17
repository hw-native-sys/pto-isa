# pto.vldx2

`pto.vldx2` 属于[向量加载与存储](../../vector-load-store_zh.md)指令集。

## 概述

双路加载并带去交错语义，常用于 AoS → SoA 转换。

## 机制

`pto.vldx2` 属于 PTO 的向量内存 / 数据搬运指令。它从 UB 中读取交错布局的数据，并一次返回两路结果向量。关键点不只是“读两次”，而是这两路结果构成一个有顺序的语义对。

## 语法

### PTO 汇编形式

```text
vldx2 %low, %high, %source[%offset], "DIST"
```

### AS Level 1（SSA）

```mlir
%low, %high = pto.vldx2 %source[%offset], "DIST" : !pto.ptr<T, ub>, index -> !pto.vreg<NxT>, !pto.vreg<NxT>
```

## 输入

- `%source`：UB 基址
- `%offset`：位移
- `DIST`：双路装载 / 去交错布局

## 预期输出

- `%low`、`%high`：两路目标向量

## 副作用

这条指令会读取 UB 可见存储并返回 SSA 结果。它不会单独分配 buffer、发送事件，也不会建立栅栏。

## 约束

- 这条指令只对交错 / 去交错类分布合法。
- 两个输出构成一个有序结果对，顺序必须被保留，不能在 lowering 里交换。

## 异常与非法情形

- 使用超出 UB 可见空间的地址，或违反所选分布模式的地址 / 对齐契约，都是非法的。
- 约束部分列出的额外非法情形，同样属于 `pto.vldx2` 的契约。

## 目标 Profile 限制

- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 示例

```c
// DINTLV_B32
for (int i = 0; i < 64; i++) {
    low[i]  = UB[base + 8*i];
    high[i] = UB[base + 8*i + 4];
}
```

```mlir
%x, %y = pto.vldx2 %ub[%offset], "DINTLV_B32" : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
```

## 详细说明

### 支持的分布模式

`DINTLV_B8`、`DINTLV_B16`、`DINTLV_B32`、`BDINTLV`

## 相关页面

- 指令集总览：[向量加载与存储](../../vector-load-store_zh.md)
- 上一条指令：[pto.vldus](./vldus_zh.md)
- 下一条指令：[pto.vsld](./vsld_zh.md)
