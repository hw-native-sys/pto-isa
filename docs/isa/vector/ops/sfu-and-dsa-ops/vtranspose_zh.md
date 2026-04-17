# pto.vtranspose

`pto.vtranspose` 属于[SFU 与 DSA 操作](../../sfu-and-dsa-ops_zh.md)指令集。

## 概述

在 UB 内完成转置的 UB-to-UB 操作，不经过 `vreg -> vreg` 路径。

## 机制

`pto.vtranspose` 把 UB 中的数据按 `%config` 控制字描述的布局规则做转置，并把结果写回另一个 UB 区域。虽然它位于 `pto.v*` 命名空间中，但它不是普通向量寄存器运算，而是带专用布局语义的 UB helper。

## 语法

### PTO 汇编形式

```text
vtranspose %dest, %src, %config
```

### AS Level 1（SSA）

```mlir
pto.vtranspose %dest, %src, %config : !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64
```

## 输入

- `%dest`、`%src`：UB 指针
- `%config`：ISA 控制 / 配置字

## 预期输出

- 这条指令没有 SSA 结果；它会把转置结果写回 `%dest`

## 副作用

这条指令会直接读写 UB 内存。

## 约束

- 它不是 `vreg -> vreg` 指令。
- 正确性依赖 `%config` 和 UB 布局契约。

## 异常与非法情形

- verifier 会拒绝非法的操作数形状、不支持的元素类型以及不合法的属性组合。
- 约束部分列出的额外非法情形，同样属于 `pto.vtranspose` 的契约。

## 目标 Profile 限制

- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 性能

### A5 时延

当前手册未单列 `vtranspose` 的周期条目。

### A2/A3 吞吐

仓内当前没有给出 `vtranspose` 的单独 cost bucket，应视为 profile 相关 UB helper。

## 示例

```mlir
pto.vtranspose %dest, %src, %config : !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64
```

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vtranspose`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vtranspose` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[SFU 与 DSA 操作](../../sfu-and-dsa-ops_zh.md)
- 上一条指令：[pto.vmula](./vmula_zh.md)
- 下一条指令：[pto.vsort32](./vsort32_zh.md)
