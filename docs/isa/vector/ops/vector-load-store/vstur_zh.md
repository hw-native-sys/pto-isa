# pto.vstur

`pto.vstur` 属于[向量加载与存储](../../vector-load-store_zh.md)指令集。

## 概述

带残余状态更新的非对齐存储。

## 机制

`pto.vstur` 属于 PTO 的向量内存 / 数据搬运指令。它会写入当前向量，同时只返回更新后的残余对齐状态，不再显式返回 offset 或 base。它适合那些外围控制流已经固定、只需要继续推进尾部状态的场景。

## 语法

### PTO 汇编形式

```text
vstur %align_out, %align_in, %value, %base, "MODE"
```

### AS Level 1（SSA）

```mlir
%align_out = pto.vstur %align_in, %value, %base, "MODE" : !pto.align, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align
```

## 输入

- `%align_in`：输入存储对齐状态
- `%value`：要写入的向量
- `%base`：UB 基址

## 预期输出

- `%align_out`：当前部分存储之后的更新残余状态

## 副作用

这条指令会写 UB 可见内存，并推进一段有状态的非对齐存储流。

## 约束

- 它只暴露推进后的状态，并不保证所有缓冲尾部字节已经提交。
- 除非外围序列明确已经闭合，否则仍然需要兼容的最终 flush 形式。

## 异常与非法情形

- 使用超出 UB 可见空间的地址，或违反所选形式的地址 / 对齐契约，都是非法的。
- 约束部分列出的额外非法情形，同样属于 `pto.vstur` 的契约。

## 目标 Profile 限制

- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 示例

```mlir
%align_out = pto.vstur %align_in, %value, %base, "MODE"
    : !pto.align, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align
```

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vstur`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vstur` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[向量加载与存储](../../vector-load-store_zh.md)
- 上一条指令：[pto.vstus](./vstus_zh.md)
