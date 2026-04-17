# pto.vsta

`pto.vsta` 属于[向量加载与存储](../../vector-load-store_zh.md)指令集。

## 概述

把对齐状态 flush 到内存。

## 机制

`pto.vsta` 属于 PTO 的向量内存 / 数据搬运指令。它消费一个待提交的 store-alignment 状态，把其中缓冲的尾部字节写回 UB，并结束这一段对齐状态在当前位置上的提交。

## 语法

### PTO 汇编形式

```text
vsta %value, %dest[%offset]
```

### AS Level 1（SSA）

```mlir
pto.vsta %value, %dest[%offset] : !pto.align, !pto.ptr<T, ub>, index
```

## 输入

- `%value`：待提交的存储对齐状态
- `%dest`：UB 基址
- `%offset`：flush 位移

## 预期输出

- 这条指令没有 SSA 结果；它会把缓冲尾部字节写回 UB

## 副作用

这条指令会写 UB 可见内存，并消费对应的对齐状态。

## 约束

- flush 地址必须与前面那条非对齐存储流所期望的后更新地址一致。
- flush 完成后，对应的存储对齐状态被消耗掉。

## 异常与非法情形

- 使用超出 UB 可见空间的地址，或违反所选形式的地址 / 对齐契约，都是非法的。
- 约束部分列出的额外非法情形，同样属于 `pto.vsta` 的契约。

## 目标 Profile 限制

- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 示例

```mlir
pto.vsta %value, %dest[%offset] : !pto.align, !pto.ptr<T, ub>, index
```

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vsta`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vsta` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[向量加载与存储](../../vector-load-store_zh.md)
- 上一条指令：[pto.vscatter](./vscatter_zh.md)
- 下一条指令：[pto.vstas](./vstas_zh.md)
