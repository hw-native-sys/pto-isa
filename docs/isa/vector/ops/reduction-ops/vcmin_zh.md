# pto.vcmin

`pto.vcmin` 属于[归约操作](../../reduction-ops_zh.md)指令集。

## 概述

对整个向量做最小值归约，并把最小值及其 argmin 信息打包到低位结果 lane。

## 机制

`pto.vcmin` 会扫描所有活跃 lane，找出最小值。输出向量的低位 lane 承载最小值和对应 lane 索引，其余 lane 清零。最小值和索引的精确打包方式由所选目标 profile 决定。

## 语法

### PTO 汇编形式

```text
vcmin %dst, %src, %mask : !pto.vreg<NxT>
```

### AS Level 1（SSA）

```mlir
%result = pto.vcmin %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%input` | `!pto.vreg<NxT>` | 待归约的源向量 |
| `%mask` | `!pto.mask` | 谓词掩码；inactive lane 不参与归约 |

## 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%result` | `!pto.vreg<NxT>` | 低位结果 lane 保存最小值及其索引，其他 lane 清零 |

## 副作用

这条指令除了产生目标值，没有其他架构副作用。

## 约束

- 值 / 索引的精确打包形式必须遵循所选目标 profile。
- 如果所有谓词位都为 0，结果遵循该指令族的零填充规则。
- 掩码宽度必须与 `N` 一致。

## 异常与非法情形

- verifier 会拒绝非法的操作数形状、不支持的元素类型以及不合法的属性组合。
- 约束部分列出的额外非法情形，同样属于 `pto.vcmin` 的契约。

## 目标 Profile 限制

- A5 当前文档化支持：`i16-i32`、`f16`、`f32`。
- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 示例

```c
T mn = INF;
int idx = 0;
for (int i = 0; i < N; i++)
    if (mask[i] && src[i] < mn) { mn = src[i]; idx = i; }
result_value = mn;
result_index = idx;
```

```mlir
%result = pto.vcmin %input, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vcmin`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vcmin` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[归约操作](../../reduction-ops_zh.md)
- 上一条指令：[pto.vcmax](./vcmax_zh.md)
- 下一条指令：[pto.vcgadd](./vcgadd_zh.md)
