# pto.vstu

`pto.vstu` 属于[向量加载与存储](../../vector-load-store_zh.md)指令集。

## 概述

带对齐状态与 offset 状态更新的非对齐存储。

## 机制

`pto.vstu` 属于 PTO 的向量内存 / 数据搬运指令。它显式消费一个输入对齐状态和一个逻辑 offset 状态，写入当前向量后，再返回更新后的对齐状态和 offset 状态。这样非对齐存储流的推进顺序在 SSA 里是完全可见的。

## 语法

### PTO 汇编形式

```text
vstu %align_out, %offset_out, %align_in, %offset_in, %value, %base, "MODE"
```

### AS Level 1（SSA）

```mlir
%align_out, %offset_out = pto.vstu %align_in, %offset_in, %value, %base, "MODE" : !pto.align, index, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align, index
```

## 输入

- `%align_in`：输入存储对齐状态
- `%offset_in`：当前逻辑字节 / 元素位移
- `%value`：要写入的向量
- `%base`：UB 基址

## 预期输出

- `%align_out`：更新后的对齐 / 尾部状态
- `%offset_out`：按所选后更新规则得到的下一个 offset

## 副作用

这条指令会写 UB 可见内存，并推进一段有状态的非对齐存储流。

## 约束

- 对齐状态必须按程序顺序串接。
- 最终仍然需要 `pto.vstar` / `pto.vstas` 这类终止 flush 形式，才能把缓冲尾部字节完全提交。

## 异常与非法情形

- 使用超出 UB 可见空间的地址，或违反所选形式的地址 / 对齐契约，都是非法的。
- 约束部分列出的额外非法情形，同样属于 `pto.vstu` 的契约。

## 目标 Profile 限制

- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 示例

```mlir
%align_out, %offset_out = pto.vstu %align_in, %offset_in, %value, %base, "MODE"
    : !pto.align, index, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align, index
```

## 详细说明

### 模式令牌

`POST_UPDATE`、`NO_POST_UPDATE`

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vstu`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vstu` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[向量加载与存储](../../vector-load-store_zh.md)
- 上一条指令：[pto.vstar](./vstar_zh.md)
- 下一条指令：[pto.vstus](./vstus_zh.md)
