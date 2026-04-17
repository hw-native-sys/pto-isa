# pto.vstas

`pto.vstas` 属于[向量加载与存储](../../vector-load-store_zh.md)指令集。

## 概述

带标量寄存器偏移形式的对齐状态 flush。

## 机制

`pto.vstas` 属于 PTO 的向量内存 / 数据搬运指令。它和 `pto.vsta` 一样会把缓冲尾部字节提交到 UB，但显式把标量寄存器风格的偏移形式暴露在指令接口里。

## 语法

### PTO 汇编形式

```text
vstas %value, %dest, %offset
```

### AS Level 1（SSA）

```mlir
pto.vstas %value, %dest, %offset : !pto.align, !pto.ptr<T, ub>, i32
```

## 输入

- `%value`：待提交的存储对齐状态
- `%dest`：UB 基址
- `%offset`：标量寄存器样式的位移

## 预期输出

- 这条指令没有 SSA 结果；它会把缓冲尾部字节写回 UB

## 副作用

这条指令会写 UB 可见内存，并消费相应的对齐状态。

## 约束

- 它沿用与 `pto.vsta` 相同的 buffered-tail 语义，只是把标量偏移形式显式化。

## 异常与非法情形

- 使用超出 UB 可见空间的地址，或违反所选形式的地址 / 对齐契约，都是非法的。
- 约束部分列出的额外非法情形，同样属于 `pto.vstas` 的契约。

## 目标 Profile 限制

- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 示例

```mlir
pto.vstas %value, %dest, %offset : !pto.align, !pto.ptr<T, ub>, i32
```

## 性能

### 时延与吞吐披露

`pto.vstas` 当前可公开对齐的 VPTO 时序来源是 `~/visa.txt` 与 `PTOAS` `feature_vpto_backend` 分支上的 `docs/vpto-spec.md`。
这些来源把带尾部缓冲的 flush 语义写得很清楚，但**没有**公布 `pto.vstas` 的数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt` §8.6 `VSTAS`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt` §8.6 `VSTAS`、`PTOAS/docs/vpto-spec.md` |

如果代码调度依赖尾刷写步骤的成本，必须在具体 backend 上实测，不能把公开 ISA 文本当作已经给出固定周期常数。

## 相关页面

- 指令集总览：[向量加载与存储](../../vector-load-store_zh.md)
- 上一条指令：[pto.vsta](./vsta_zh.md)
- 下一条指令：[pto.vstar](./vstar_zh.md)
