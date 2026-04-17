# pto.vsld

`pto.vsld` 属于[向量加载与存储](../../vector-load-store_zh.md)指令集。

## 概述

带固定 stride 模式的加载。

## 机制

`pto.vsld` 属于 PTO 的向量内存 / 数据搬运指令。它从 UB 中按固定 stride token 选定的子元素模式装载数据。这里真正决定访问模式的不是普通的字节位移，而是 `STRIDE` 所代表的硬件步长模式。

## 语法

### PTO 汇编形式

```text
vsld %result, %source[%offset], "STRIDE"
```

### AS Level 1（SSA）

```mlir
%result = pto.vsld %source[%offset], "STRIDE" : !pto.ptr<T, ub> -> !pto.vreg<NxT>
```

## 输入

- `%source`：UB 基址
- `%offset`：与所选固定 stride 模式共同解释的位移

## 预期输出

- `%result`：装载得到的向量

## 副作用

这条指令会读取 UB 可见存储并返回 SSA 结果。

## 约束

- 这是一个兼容性保留指令集。
- 真正决定“从源块里读哪些子元素”的是 stride token，而不是简单的 lane 编号或普通偏移。

## 异常与非法情形

- 使用超出 UB 可见空间的地址，或违反所选形式的地址 / 对齐契约，都是非法的。
- 约束部分列出的额外非法情形，同样属于 `pto.vsld` 的契约。

## 目标 Profile 限制

- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 示例

```mlir
%result = pto.vsld %source[%offset], "STRIDE" : !pto.ptr<T, ub> -> !pto.vreg<NxT>
```

## 详细说明

### 常见 stride 模式

`STRIDE_S3_B16`、`STRIDE_S4_B64`、`STRIDE_S8_B32`、`STRIDE_S2_B64`

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vsld`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vsld` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[向量加载与存储](../../vector-load-store_zh.md)
- 上一条指令：[pto.vldx2](./vldx2_zh.md)
- 下一条指令：[pto.vsldb](./vsldb_zh.md)
