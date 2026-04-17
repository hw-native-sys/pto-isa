# pto.vsst

`pto.vsst` 属于[向量加载与存储](../../vector-load-store_zh.md)指令集。

## 概述

带固定 stride 模式的存储。

## 机制

`pto.vsst` 属于 PTO 的向量内存 / 数据搬运指令。它把向量寄存器按固定 stride token 所描述的模式写回 UB。这里真正决定写回布局的是 stride token，而不是简单的 lane 序号。

## 语法

### PTO 汇编形式

```text
vsst %value, %dest[%offset], "STRIDE"
```

### AS Level 1（SSA）

```mlir
pto.vsst %value, %dest[%offset], "STRIDE" : !pto.vreg<NxT>, !pto.ptr<T, ub>
```

## 输入

- `%value`：源向量
- `%dest`：UB 基址
- `%offset` / `STRIDE`：共同决定固定 stride 存储布局

## 预期输出

- 这条指令没有 SSA 结果；它会把数据写回 UB

## 副作用

这条指令会写 UB 可见内存。对某些有状态的非对齐流式形式，还可能推进对齐状态，但尾部 flush 仍可能需要额外指令完成。

## 约束

- 这是一个兼容性保留指令集。
- 真正决定“写哪些子元素”的是 stride token，而不是单纯的向量 lane 编号。

## 异常与非法情形

- 使用超出 UB 可见空间的地址，或违反所选形式的地址 / 对齐契约，都是非法的。
- 约束部分列出的额外非法情形，同样属于 `pto.vsst` 的契约。

## 目标 Profile 限制

- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 示例

```mlir
pto.vsst %value, %dest[%offset], "STRIDE" : !pto.vreg<NxT>, !pto.ptr<T, ub>
```

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vsst`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vsst` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[向量加载与存储](../../vector-load-store_zh.md)
- 上一条指令：[pto.vstx2](./vstx2_zh.md)
- 下一条指令：[pto.vsstb](./vsstb_zh.md)
