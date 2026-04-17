# pto.vaddc

`pto.vaddc` 属于[二元向量操作](../../binary-vector-ops_zh.md)指令集。

## 概述

`pto.vaddc` 是逐 lane 的整数加法，同时产生两个结果：

- 一个截断后的结果向量
- 一个按 lane 给出的 carry / overflow 谓词掩码

## 机制

这条指令对两个整数向量做逐 lane 加法，并额外返回每个 lane 的 carry-out。对每个满足谓词的 lane `i`：

$$ \mathrm{dst}_i = \mathrm{lhs}_i + \mathrm{rhs}_i $$

$$ \mathrm{carry}_i = \text{lane } i \text{ 的无符号进位输出} $$

在当前 A5 指令语义里，应把它看成无符号整数 carry-chain 加法。`%carry` 可以继续作为下一条 `vaddc` 的掩码输入，用来实现多字长整数加法链。非活跃 lane 上，结果寄存器和 carry 掩码都保持原值。

## 语法

### PTO 汇编形式

```text
vaddc %dst, %carry, %lhs, %rhs, %mask : !pto.vreg<NxT>
```

### AS Level 1（SSA）

```mlir
%result, %carry = pto.vaddc %lhs, %rhs, %mask : (!pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask) -> !pto.vreg<NxT>, !pto.mask
```

### AS Level 2（DPS）

```mlir
pto.vaddc ins(%lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask)
          outs(%result, %carry : !pto.vreg<NxT>, !pto.mask)
```

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%lhs` | `!pto.vreg<NxT>` | 第一个加数 |
| `%rhs` | `!pto.vreg<NxT>` | 第二个加数 |
| `%mask` | `!pto.mask` | 谓词掩码；掩码位为 1 的 lane 参与加法 |

两个源寄存器必须具有相同的元素类型和相同的向量宽度 `N`。掩码宽度必须与 `N` 一致。

## 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%result` | `!pto.vreg<NxT>` | 活跃 lane 上得到截断后的和；非活跃 lane 保持原值 |
| `%carry` | `!pto.mask` | 逐 lane 的进位 / 溢出谓词；无符号加法发生溢出时，对应 lane 为 1 |

## 副作用

这条指令除了产生目标向量寄存器和 carry 谓词外，没有其他架构副作用。它不会隐式占用 buffer、发送事件，也不会建立内存栅栏。

## 约束

- 只支持整数元素类型。这是一条 carry-chain 整数加法指令。
- 在 A5 上，应把它视作无符号整数运算。
- `%lhs`、`%rhs` 和 `%result` 的元素类型必须一致。
- 所有寄存器的向量宽度必须一致。
- `%mask` 的宽度必须与 `N` 一致。
- 非活跃 lane 上，结果与 carry 都保持原值。

## 异常与非法情形

- verifier 会拒绝非整数元素类型、类型不匹配、宽度不匹配或掩码宽度不匹配。
- [二元向量操作](../../binary-vector-ops_zh.md)页面列出的额外非法情形，同样适用于 `pto.vaddc`。

## 目标 Profile 限制

A5 是当前文档里最具体的向量 profile。CPU 模拟器和 A2/A3 类目标会在保留可见 PTO 契约的前提下模拟这条指令。

## 性能

### A5 时延

| 元素类型 | 时延（周期） | A5 RV |
|----------|--------------|-------|
| `i32` | 7 | `RV_VADDC` |

## 示例

### C 语义

```c
for (int i = 0; i < N; i++) {
    uint64_t r = (uint64_t)src0[i] + src1[i];
    dst[i] = (T)r;
    carry[i] = (r >> bitwidth);
}
```

### MLIR 用法

```mlir
%result, %carry = pto.vaddc %a, %b, %active
    : (!pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask) -> !pto.vreg<64xi32>, !pto.mask

%sum0, %carry0 = pto.vaddc %a0, %b0, %active : ...
%sum1, %carry1 = pto.vaddc %a1, %b1, %carry0 : ...
```

### 典型场景：多字长整数加法

```mlir
%sum_low, %carry = pto.vaddc %a_low, %b_low, %active : ...
%sum_high, %carry2 = pto.vaddc %a_high, %b_high, %carry : ...
```

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vaddc`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vaddc` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[二元向量操作](../../binary-vector-ops_zh.md)
- 上一条指令：[pto.vshr](./vshr_zh.md)
- 下一条指令：[pto.vsubc](./vsubc_zh.md)
