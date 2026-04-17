# pto.vsubcs

`pto.vsubcs` 属于[向量-标量操作](../../vec-scalar-ops_zh.md)指令集。

## 概述

`pto.vsubcs` 是带显式 borrow-in / borrow-out 掩码的逐 lane 减法。

## 机制

对每个活跃 lane `i`，执行：

```text
diff = lhs[i] - rhs[i] - borrow_in[i]
result[i] = low_bits(diff)
borrow[i] = borrow_out(diff)
```

在 PTO 可见表面里，这条 borrow 链也是逐 lane 独立的：每个 lane 消费一个输入借位 bit，并产出一个输出借位 bit。

## 语法

### PTO 汇编形式

```text
vsubcs %dst, %borrow_out, %lhs, %rhs, %borrow_in, %mask : !pto.vreg<NxT>, !pto.mask
```

### AS Level 1（SSA）

```mlir
%result, %borrow = pto.vsubcs %lhs, %rhs, %borrow_in, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask, !pto.mask -> !pto.vreg<NxT>, !pto.mask
```

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%lhs` | `!pto.vreg<NxT>` | 被减数向量 |
| `%rhs` | `!pto.vreg<NxT>` | 减数向量 |
| `%borrow_in` | `!pto.mask` | 每个 lane 的输入借位 bit |
| `%mask` | `!pto.mask` | 谓词掩码；只有掩码位为 1 的 lane 参与运算 |

## 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%result` | `!pto.vreg<NxT>` | 活跃 lane 上得到算术结果 |
| `%borrow` | `!pto.mask` | 每个活跃 lane 产生的 borrow-out bit |

## 副作用

这条指令除了产生目标值和 borrow 掩码，没有其他架构副作用。

## 约束

- borrow 链形式只对整数元素类型定义。
- `%lhs`、`%rhs` 与 `%result` 必须具有相同的向量宽度 `N` 和相同的元素类型 `T`。
- `%borrow_in`、`%borrow` 与 `%mask` 的宽度都必须为 `N`。

## 异常与非法情形

- verifier 会拒绝非法的操作数形状、不支持的元素类型以及不合法的属性组合。
- 约束部分列出的额外非法情形，同样属于 `pto.vsubcs` 的契约。

## 目标 Profile 限制

- 除非某个目标 profile 明确扩大合法域，否则应把这条形式视为无符号整数 borrow-chain。
- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 示例

```c
for (int i = 0; i < N; i++) {
    if (!mask[i]) continue;
    uint64_t rhs_total = (uint64_t)rhs[i] + borrow_in[i];
    result[i] = lhs[i] - rhs_total;
    borrow[i] = lhs[i] < rhs_total;
}
```

```mlir
%result, %borrow = pto.vsubcs %lhs, %rhs, %borrow_in, %mask
    : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask, !pto.mask -> !pto.vreg<64xi32>, !pto.mask
```

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vsubcs`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vsubcs` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[向量-标量操作](../../vec-scalar-ops_zh.md)
- 上一条指令：[pto.vaddcs](./vaddcs_zh.md)
