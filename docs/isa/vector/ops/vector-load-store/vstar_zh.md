# pto.vstar

`pto.vstar` 属于[向量加载与存储](../../vector-load-store_zh.md)指令集。

## 概述

提交剩余的对齐状态。

## 机制

`pto.vstar` 属于 PTO 的向量内存 / 数据搬运指令。它会把仍然保存在对齐 / 缓冲状态里的剩余字节一次性冲刷到内存，从而结束一段非对齐存储序列。

## 语法

### PTO 汇编形式

```text
vstar %value, %dest
```

### AS Level 1（SSA）

```mlir
pto.vstar %value, %dest : !pto.align, !pto.ptr<T, ub>
```

## 输入

- `%value`：仍待提交的对齐 / 缓冲状态
- `%dest`：UB 目标基址

## 预期输出

- 没有 SSA 结果；它的效果是在内存侧完成剩余字节的 flush

## 副作用

这条指令会写 UB 可见内存，并终止当前非对齐存储序列。

## 约束

- 它必须与前面兼容的状态产生型存储序列配对，否则待提交尾部状态没有定义。

## 异常与非法情形

- 使用超出 UB 可见空间的地址，或违反所选形式的地址 / 对齐契约，都是非法的。
- 约束部分列出的额外非法情形，同样属于 `pto.vstar` 的契约。

## 目标 Profile 限制

- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 示例

```mlir
pto.vstar %value, %dest : !pto.align, !pto.ptr<T, ub>
```

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vstar`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vstar` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[向量加载与存储](../../vector-load-store_zh.md)
- 上一条指令：[pto.vstas](./vstas_zh.md)
- 下一条指令：[pto.vstu](./vstu_zh.md)
