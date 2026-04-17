# pto.vsubc

`pto.vsubc` 属于[二元向量操作](../../binary-vector-ops_zh.md)指令集。

## 概述

`pto.vsubc` 是逐 lane 的整数减法，同时产生两个结果：

- 一个逐 lane 差值向量
- 一个逐 lane 的 borrow 谓词掩码

## 机制

这条指令对两个整数向量做逐 lane 相减，并额外返回每个 lane 的 borrow / underflow 标志。对每个满足谓词的 lane `i`：

$$ \mathrm{dst}_i = \mathrm{lhs}_i - \mathrm{rhs}_i $$

$$ \mathrm{borrow}_i = (\mathrm{lhs}_i < \mathrm{rhs}_i) $$

在当前 A5 指令语义里，应把它视作无符号 32 位 borrow-chain 减法。`%borrow` 可以继续接到下一条 `vsubc` 上，实现多字长整数减法链。非活跃 lane 上，结果寄存器和 borrow 掩码都保持原值。

## 语法

### PTO 汇编形式

```text
vsubc %dst, %borrow, %lhs, %rhs, %mask : !pto.vreg<NxT>
```

### AS Level 1（SSA）

```mlir
%result, %borrow = pto.vsubc %lhs, %rhs, %mask : (!pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask) -> !pto.vreg<NxT>, !pto.mask
```

### AS Level 2（DPS）

```mlir
pto.vsubc ins(%lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask)
          outs(%result, %borrow : !pto.vreg<NxT>, !pto.mask)
```

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%lhs` | `!pto.vreg<NxT>` | 被减数 |
| `%rhs` | `!pto.vreg<NxT>` | 减数 |
| `%mask` | `!pto.mask` | 谓词掩码；掩码位为 1 的 lane 参与减法 |

两个源寄存器必须具有相同的元素类型和相同的向量宽度 `N`。掩码宽度必须与 `N` 一致。

## 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%result` | `!pto.vreg<NxT>` | 活跃 lane 上得到逐 lane 差值；非活跃 lane 保持原值 |
| `%borrow` | `!pto.mask` | 逐 lane 的借位谓词；无符号下溢时，对应 lane 为 1 |

## 副作用

这条指令除了产生目标向量寄存器和 borrow 谓词外，没有其他架构副作用。它不会隐式占用 buffer、发送事件，也不会建立内存栅栏。

## 约束

- 只支持整数元素类型。这是一条 borrow-chain 整数减法指令。
- 在 A5 上，应把它视作无符号 32 位 borrow-chain 运算，除非 verifier 另行收窄或放宽。
- `%lhs`、`%rhs` 和 `%result` 的元素类型必须一致。
- 所有寄存器的向量宽度必须一致。
- `%mask` 的宽度必须与 `N` 一致。
- 非活跃 lane 上，结果与 borrow 都保持原值。

## 异常与非法情形

- verifier 会拒绝非整数元素类型、类型不匹配、宽度不匹配或掩码宽度不匹配。
- [二元向量操作](../../binary-vector-ops_zh.md)页面列出的额外非法情形，同样适用于 `pto.vsubc`。

## 目标 Profile 限制

A5 是当前文档里最具体的向量 profile。CPU 模拟器和 A2/A3 类目标会在保留可见 PTO 契约的前提下模拟这条指令。

## 性能

### A5 时延

| 元素类型 | 时延（周期） | A5 RV |
|----------|--------------|-------|
| `i32` | 7 | `RV_VSUBC` |

## 示例

### C 语义

```c
for (int i = 0; i < N; i++) {
    dst[i] = src0[i] - src1[i];
    borrow[i] = (src0[i] < src1[i]);
}
```

### MLIR 用法

```mlir
%result, %borrow = pto.vsubc %a, %b, %active
    : (!pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask) -> !pto.vreg<64xi32>, !pto.mask

%diff0, %borrow0 = pto.vsubc %a0, %b0, %active : ...
%diff1, %borrow1 = pto.vsubc %a1, %b1, %borrow0 : ...
```

### 典型场景：多字长整数减法

```mlir
%diff_low, %borrow = pto.vsubc %a_low, %b_low, %active : ...
%diff_high, %borrow2 = pto.vsubc %a_high, %b_high, %borrow : ...
```

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vsubc`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vsubc` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[二元向量操作](../../binary-vector-ops_zh.md)
- 上一条指令：[pto.vaddc](./vaddc_zh.md)
