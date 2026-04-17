# pto.vsort32

`pto.vsort32` 属于[SFU 与 DSA 操作](../../sfu-and-dsa-ops_zh.md)指令集。

## 概述

对 UB 中的 32 个元素排序。

## 机制

`pto.vsort32` 是一个 UB-to-UB 排序 helper。它读取 `%src` 指向的 UB 数据，按 `%config` 所定义的比较方式和方向完成排序，再把结果写回 `%dest`。它不是纯向量寄存器算术，而是服务于排序流水的专用辅助操作。

## 语法

### PTO 汇编形式

```text
vsort32 %dest, %src, %config
```

### AS Level 1（SSA）

```mlir
pto.vsort32 %dest, %src, %config : !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64
```

## 输入

- `%dest`、`%src`：UB 指针
- `%config`：ISA 配置字

## 预期输出

- 这条指令没有 SSA 结果；它会把排序结果写回 `%dest`

## 副作用

这条指令会直接读写 UB 内存。

## 约束

- 它是 UB-to-UB 加速器 helper，而不是纯 `vreg -> vreg` 操作。

## 异常与非法情形

- verifier 会拒绝非法的操作数形状、不支持的元素类型以及不合法的属性组合。
- 约束部分列出的额外非法情形，同样属于 `pto.vsort32` 的契约。

## 目标 Profile 限制

- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 性能

### A5 时延

当前手册未单列 `vsort32` 的周期级条目。

### A2/A3 吞吐

仓内现有 costmodel 没有直接给出 `vsort32` 的独立 `SetParam`，排序相关可参照 `vbitsort` 与 `vmrgsort4` 的分块 / 归并路径。

## 示例

```mlir
pto.vsort32 %dest, %src, %config : !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64
```

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vsort32`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vsort32` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[SFU 与 DSA 操作](../../sfu-and-dsa-ops_zh.md)
- 上一条指令：[pto.vtranspose](./vtranspose_zh.md)
- 下一条指令：[pto.vmrgsort](./vmrgsort_zh.md)
