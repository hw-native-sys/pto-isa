# pto.vshr

`pto.vshr` 属于[二元向量操作](../../binary-vector-ops_zh.md)指令集。

## 概述

`pto.vshr` 在每个活跃 lane 上做右移：

```text
dst[i] = lhs[i] >> rhs[i]
```

对有符号类型，这是算术右移；对无符号类型，这是逻辑右移。

## 机制

这条指令用 `%rhs` 每个 lane 上的计数去右移 `%lhs` 对应 lane 的值。对每个满足谓词的 lane `i`：

$$ \mathrm{dst}_i = \mathrm{lhs}_i \gg \mathrm{rhs}_i $$

- 有符号整数类型（`i8`–`i64`）：做算术右移，符号位复制。
- 无符号整数类型（`u8`–`u64`）：做逻辑右移，高位补零。
- `rhs[i]` 按无符号位移量解释。
- 移出的比特被丢弃。
- 非活跃 lane 保持目标原值。

## 语法

### PTO 汇编形式

```text
vshr %dst, %lhs, %rhs, %mask : !pto.vreg<NxT>
```

### AS Level 1（SSA）

```mlir
%result = pto.vshr %lhs, %rhs, %mask : (!pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask) -> !pto.vreg<NxT>
```

### AS Level 2（DPS）

```mlir
pto.vshr ins(%lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask)
          outs(%result : !pto.vreg<NxT>)
```

支持的元素类型是全部整数类型：`i8`–`i64`、`u8`–`u64`。

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%lhs` | `!pto.vreg<NxT>` | 被右移的值 |
| `%rhs` | `!pto.vreg<NxT>` | 每个 lane 的无符号位移计数 |
| `%mask` | `!pto.mask` | 谓词掩码；掩码位为 1 的 lane 参与位移 |

## 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%result` | `!pto.vreg<NxT>` | 活跃 lane 上得到逐 lane 右移结果；非活跃 lane 保持原值 |

## 副作用

这条指令除了产生目标向量寄存器外，没有其他架构副作用。它不会隐式占用 buffer、发送事件，也不会建立内存栅栏。

## 约束

- 只支持整数元素类型。
- 元素类型的 signedness 决定算术右移还是逻辑右移。
- `%lhs`、`%rhs` 和 `%result` 的元素类型必须一致。
- 三个寄存器的向量宽度必须一致。
- `%mask` 的宽度必须与 `N` 一致。
- 位移量最好保持在 `[0, bitwidth(T) - 1]` 范围内；超范围行为由目标平台定义。
- 非活跃 lane 保持目标原值。

## 异常与非法情形

- verifier 会拒绝非整数元素类型、类型不匹配、宽度不匹配或掩码宽度不匹配。
- [二元向量操作](../../binary-vector-ops_zh.md)页面列出的额外非法情形，同样适用于 `pto.vshr`。

## 目标 Profile 限制

| 元素类型 | CPU Simulator | A2/A3 | A5 |
|-----------|:-------------:|:-----:|:--:|
| 整数类型 | 模拟 | 模拟 | 支持 |

A5 是当前文档里最具体的向量 profile。

## 性能

### A5 时延

| 元素类型 | 时延（周期） | A5 RV |
|----------|--------------|-------|
| `i32` | 7 | `RV_VSHR` |

### A2/A3 吞吐

| 指标 | 数值 | 常量 |
|------|------|------|
| 启动时延 | 14 | `A2A3_STARTUP_BINARY` |
| 完成时延 | 17 | `A2A3_COMPL_INT_BINOP` |
| 每次 repeat 吞吐 | 2 | `A2A3_RPT_2` |
| 流水间隔 | 18 | `A2A3_INTERVAL` |

## 示例

### C 语义

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] >> src1[i];
```

### MLIR 用法

```mlir
%count = pto.vbroadcast %c2 : i32 -> !pto.vreg<64xi32>
%shifted = pto.vshr %data, %count, %active
    : (!pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask) -> !pto.vreg<64xi32>

%shifted2 = pto.vshr %data, %counts, %active
    : (!pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask) -> !pto.vreg<64xi32>
```

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vshr`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vshr` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[二元向量操作](../../binary-vector-ops_zh.md)
- 上一条指令：[pto.vshl](./vshl_zh.md)
- 下一条指令：[pto.vaddc](./vaddc_zh.md)
