# pto.vdiv

`pto.vdiv` 属于[二元向量操作](../../binary-vector-ops_zh.md)指令集。

## 概述

`%result` 是 `%lhs` 与 `%rhs` 的逐 lane 商值。

## 机制

`pto.vdiv` 是 `pto.v*` 计算指令，在活跃 lane 上执行向量除法。`%lhs` 是被除数，`%rhs` 是除数。

对每个活跃 lane：

$$ \mathrm{dst}_i = \mathrm{lhs}_i / \mathrm{rhs}_i $$

这条指令当前只记录浮点形式，不包含整数除法。

## 语法

### PTO 汇编形式

```text
vdiv %result, %lhs, %rhs, %mask
```

### AS Level 1（SSA）

```mlir
%result = pto.vdiv %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

A5 当前文档化的形式只有 `f16` 与 `f32`。

## 输入

- `%lhs`：被除数
- `%rhs`：除数
- `%mask`：选择活跃 lane 的谓词掩码

## 预期输出

- `%result`：逐 lane 商值

## 副作用

这条指令除了产生 SSA 结果外，没有其他架构副作用。除非具体形式另有说明，它不会隐式占用 buffer、发送事件，也不会建立内存栅栏。

## 约束

- 只支持浮点元素类型。
- 活跃 lane 上如果除数是 `+0` 或 `-0`，其行为遵循目标平台的异常语义。

## 异常与非法情形

- verifier 会拒绝非法的向量形状、不支持的元素类型，以及不合法的属性组合。
- 除零、域外输入等数值异常，仍由所选 backend profile 决定，除非本页进一步收窄。
- 约束部分列出的额外非法情形同样属于 `pto.vdiv` 的契约。

## 目标 Profile 限制

- A5 当前文档化支持：仅 `f16`、`f32`，不支持整数除法。
- A5 是当前手册中最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。
- 任何依赖具体指令类型清单、分布模式或融合路径的代码，都应视为 profile 相关。

## 性能

### A5 时延

| 元素类型 | 时延（周期） | A5 RV |
|----------|--------------|-------|
| `f32` | 17 | `RV_VDIV` |
| `f16` | 22 | `RV_VDIV` |

### A2/A3 吞吐

| 指标 | f32 | f16 |
|------|-----|-----|
| 启动时延 | 14 | `A2A3_STARTUP_BINARY` |
| 完成时延 | 20 | `A2A3_COMPL_FP_MUL` |
| 每次 repeat 吞吐 | 2 | 4 |
| 流水间隔 | 18 | 18 |

**性能说明：** 除法显著慢于乘法。A5 上 `vdiv` 需要 17–22 周期，而 `vmul` 只要 8 周期。若精度允许，更建议先求倒数，再用乘法实现。

## 示例

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] / src1[i];
```

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vdiv`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vdiv` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[二元向量操作](../../binary-vector-ops_zh.md)
- 上一条指令：[pto.vmul](./vmul_zh.md)
- 下一条指令：[pto.vmax](./vmax_zh.md)
