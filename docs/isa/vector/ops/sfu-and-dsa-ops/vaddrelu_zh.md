# pto.vaddrelu

`pto.vaddrelu` 属于[SFU 与 DSA 操作](../../sfu-and-dsa-ops_zh.md)指令集。

## 概述

把逐 lane 加法和 ReLU 融合成一条指令。

## 机制

这条指令执行：

```text
dst[i] = max(0, lhs[i] + rhs[i])
```

它把“加法 + ReLU”压成单条硬件操作，减少中间寄存器结果落地，也避免把融合语义退化成普通的 `vadd` 后接 `vrelu`。

## 语法

### PTO 汇编形式

```text
vaddrelu %dst, %lhs, %rhs, %mask : !pto.vreg<NxT>
```

### AS Level 1（SSA）

```mlir
%result = pto.vaddrelu %lhs, %rhs, %mask : (!pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask) -> !pto.vreg<NxT>
```

A5 当前文档化支持的类型：`f16`、`f32`。

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%lhs` | `!pto.vreg<NxT>` | 左操作数向量 |
| `%rhs` | `!pto.vreg<NxT>` | 右操作数向量 |
| `%mask` | `!pto.mask` | 谓词掩码；掩码位为 1 的 lane 为活跃 lane |

## 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%result` | `!pto.vreg<NxT>` | 融合 add-then-ReLU 结果 |

## 副作用

这条指令除了产生目标向量寄存器，没有其他架构副作用。

## 约束

- `%lhs`、`%rhs` 与 `%result` 必须有相同的元素类型和相同的向量宽度 `N`。
- `%mask` 的宽度必须与 `N` 一致。
- 当前文档只记录浮点形式。

## 异常与非法情形

- verifier 会拒绝非法的操作数形状、不支持的元素类型以及不合法的属性组合。
- 约束部分列出的额外非法情形，同样属于 `pto.vaddrelu` 的契约。

## 目标 Profile 限制

- A5 当前文档化支持：`f16`、`f32`。
- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 性能

### A5 时延

当前手册没有给出 `vaddrelu` 的独立周期条目。

### A2/A3 吞吐

英文 leaf 页当前沿用 fused/SFU 桶：

| 指标 | 数值 | 常量 |
|------|------|------|
| 启动时延 | 14 | `A2A3_STARTUP_BINARY` |
| 完成时延 | 26 | `A2A3_COMPL_FP32_EXP` |
| 每次 repeat 吞吐 | 2 | `A2A3_RPT_2` |
| 流水间隔 | 18 | `A2A3_INTERVAL` |

## 示例

```c
for (int i = 0; i < N; i++)
    dst[i] = max(src0[i] + src1[i], 0);
```

```mlir
%result = pto.vaddrelu %lhs, %rhs, %mask
    : (!pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask) -> !pto.vreg<64xf32>
```

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vaddrelu`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vaddrelu` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[SFU 与 DSA 操作](../../sfu-and-dsa-ops_zh.md)
- 上一条指令：[pto.vexpdiff](./vexpdiff_zh.md)
- 下一条指令：[pto.vsubrelu](./vsubrelu_zh.md)
