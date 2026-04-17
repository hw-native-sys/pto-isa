# pto.vstus

`pto.vstus` 属于[向量加载与存储](../../vector-load-store_zh.md)指令集。

## 概述

带标量 offset 与状态更新的非对齐存储。

## 机制

`pto.vstus` 属于 PTO 的向量内存 / 数据搬运指令。它和 `pto.vstu` 一样会推进有状态的非对齐存储流，只是这里显式使用标量 offset，而不是 index 风格的位移状态。

## 语法

### PTO 汇编形式

```text
vstus %align_out, %base_out, %align_in, %offset, %value, %base, "MODE"
```

### AS Level 1（SSA）

```mlir
%align_out, %base_out = pto.vstus %align_in, %offset, %value, %base, "MODE" : !pto.align, i32, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align, !pto.ptr<T, ub>
```

## 输入

- `%align_in`：输入存储对齐状态
- `%offset`：标量位移
- `%value`：要写入的向量
- `%base`：UB 基址

## 预期输出

- `%align_out`：更新后的 buffered-tail 状态
- `%base_out`：当 lowering 采用 post-update 形式时得到的下一个基址

## 副作用

这条指令会写 UB 可见内存，并推进一段有状态的非对齐存储流。

## 约束

- 它是非对齐存储流的标量 offset 形式。
- 标量 offset 的位宽和 update mode 必须与所选形式匹配。
- 之后仍需要 flush 指令把缓冲尾部字节真正提交。

## 异常与非法情形

- 使用超出 UB 可见空间的地址，或违反所选形式的地址 / 对齐契约，都是非法的。
- 约束部分列出的额外非法情形，同样属于 `pto.vstus` 的契约。

## 目标 Profile 限制

- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 示例

```mlir
%align_out, %base_out = pto.vstus %align_in, %offset, %value, %base, "MODE"
    : !pto.align, i32, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align, !pto.ptr<T, ub>
```

## 性能

### 时延与吞吐披露

`pto.vstus` 当前可公开对齐的 VPTO 时序来源是 `~/visa.txt` 与 `PTOAS` `feature_vpto_backend` 分支上的 `docs/vpto-spec.md`。
这些来源详细定义了有状态非对齐 store 的缓冲语义，但**没有**公布 `pto.vstus` 的数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt` §8.9 `VSTUS`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt` §8.9 `VSTUS`、`PTOAS/docs/vpto-spec.md` |

由于 `pto.vstus` 参与的是带状态的缓冲写流，只要公开 ISA 来源还没有给出数字时序，就必须把它视为 backend-specific timing。

## 相关页面

- 指令集总览：[向量加载与存储](../../vector-load-store_zh.md)
- 上一条指令：[pto.vstu](./vstu_zh.md)
- 下一条指令：[pto.vstur](./vstur_zh.md)
