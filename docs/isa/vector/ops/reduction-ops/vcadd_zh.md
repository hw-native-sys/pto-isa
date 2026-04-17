# pto.vcadd

`pto.vcadd` 属于[归约操作](../../reduction-ops_zh.md)指令集。

## 概述

对整个向量做完整归约求和，把单个标量结果写到 lane 0，其余 lane 清零。

## 机制

`pto.vcadd` 会把源向量中所有活跃 lane 通过树形归约压成一个总和，并把结果写到输出向量的 lane 0；其余 lane 都按零填充。inactive lane 按 0 处理。如果所有谓词位都为 0，结果也是 0。

对所有活跃 lane：

$$ \mathrm{dst}_{0} = \sum_{i=0}^{N-1} \mathrm{src}_{i} $$

## 语法

### PTO 汇编形式

```text
vcadd %dst, %src, %mask : !pto.vreg<NxT>
```

### AS Level 1（SSA）

```mlir
%result = pto.vcadd %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

A5 当前文档化支持的类型：`i16-i64`、`f16`、`f32`。

## 输入

| 操作数 | 角色 | 说明 |
|--------|------|------|
| `%input` | 源向量 | 保存待归约的值 |
| `%mask` | 谓词掩码 | 选择哪些 lane 参与归约；inactive lane 贡献 0 |

## 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%result` | `!pto.vreg<NxT>` | lane 0 保存总和，其余 lane 全部清零 |

## 副作用

这条指令除了产生 SSA 结果，没有其他架构副作用。

## 约束

- 某些窄整数形式可能在内部使用更宽的累加器，但最终结果仍按声明结果类型返回。
- 如果所有谓词位都为 0，lane 0 为 0，其余 lane 也为 0。
- mask 粒度是逐 lane 的，不支持更细粒度的子 lane 掩码。

## 异常与非法情形

- verifier 会拒绝非法的操作数形状、不支持的元素类型以及不合法的属性组合。
- 约束部分列出的额外非法情形，同样属于 `pto.vcadd` 的契约。

## 目标 Profile 限制

- A5 当前文档化支持：`i16-i64`、`f16`、`f32`。
- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 性能

### A5 时延与吞吐

归约操作的时延和吞吐强依赖目标 profile。与逐元素算术相比，完整向量归约通常更贵，因为硬件需要执行树形规约过程。

## 示例

### C 语义

```c
T sum = 0;
for (int i = 0; i < N; i++)
    sum += src[i];
dst[0] = sum;
for (int i = 1; i < N; i++)
    dst[i] = 0;
```

### MLIR

```mlir
%result = pto.vcadd %input, %mask : !pto.vreg<128xf32>, !pto.mask -> !pto.vreg<128xf32>
```

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vcadd`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vcadd` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[归约操作](../../reduction-ops_zh.md)
- 下一条指令：[pto.vcmax](./vcmax_zh.md)
- 相关归约：[pto.vcgadd](./vcgadd_zh.md)
