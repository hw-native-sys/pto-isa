# pto.vcgadd

`pto.vcgadd` 属于[归约操作](../../reduction-ops_zh.md)指令集。

## 概述

按 VLane 分组做求和归约。每个 32 字节 VLane 组产生一个结果，写到该组的低位 lane，其余 lane 清零。

## 机制

`pto.vcgadd` 不是对整个向量求一个总和，而是把每个硬件 32 字节 VLane 组独立归约。每组内部做树形规约，得到的组内总和写回该组的低位 lane，其余 lane 置零。inactive lane 按 0 处理。

对 f32 而言，每个 VLane 组包含 8 个元素，因此：

$$ \mathrm{dst}_{g \times 8} = \sum_{i=0}^{7} \mathrm{src}_{g \times 8 + i} $$

## 语法

### PTO 汇编形式

```text
vcgadd %dst, %src, %mask : !pto.vreg<NxT>
```

### AS Level 1（SSA）

```mlir
%result = pto.vcgadd %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

A5 当前文档化支持的类型：`i16-i32`、`f16`、`f32`。

## 输入

| 操作数 | 角色 | 说明 |
|--------|------|------|
| `%input` | 源向量 | 在每个 VLane 组内参与归约的值 |
| `%mask` | 谓词掩码 | 选择哪些 lane 参与归约；inactive lane 贡献 0 |

## 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%result` | `!pto.vreg<NxT>` | 每个 VLane 组的低位 lane 保存该组总和，其余 lane 清零 |

## 副作用

这条指令除了产生 SSA 结果，没有其他架构副作用。

## 约束

- 归约粒度固定是硬件 32 字节 VLane，不是任意软件切片。
- 如果某个组里所有谓词位都为 0，则该组的结果为 0。
- A5 当前支持 `i16-i32`、`f16`、`f32`。

## 异常与非法情形

- verifier 会拒绝非法的操作数形状、不支持的元素类型以及不合法的属性组合。
- 约束部分列出的额外非法情形，同样属于 `pto.vcgadd` 的契约。

## 目标 Profile 限制

- A5 当前文档化支持：`i16-i32`、`f16`、`f32`。
- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 性能

### A5 时延与吞吐

按组归约通常比完整向量归约更容易并行，因为不同 VLane 组是彼此独立的。

## 示例

### C 语义

```c
int K = N / 8;
for (int g = 0; g < 8; g++) {
    T sum = 0;
    for (int i = 0; i < K; i++)
        sum += src[g*K + i];
    dst[g*K] = sum;
    for (int i = 1; i < K; i++)
        dst[g*K + i] = 0;
}
```

### MLIR

```mlir
%result = pto.vcgadd %input, %mask : !pto.vreg<128xf32>, !pto.mask -> !pto.vreg<128xf32>
```

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vcgadd`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vcgadd` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[归约操作](../../reduction-ops_zh.md)
- 上一条指令：[pto.vcmin](./vcmin_zh.md)
- 下一条指令：[pto.vcgmax](./vcgmax_zh.md)
- 相关归约：[pto.vcadd](./vcadd_zh.md)
