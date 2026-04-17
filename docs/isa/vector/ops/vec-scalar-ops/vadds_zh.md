# pto.vadds

`pto.vadds` 属于[向量-标量操作](../../vec-scalar-ops_zh.md)指令集。

## 概述

`pto.vadds` 对一个向量寄存器和一个广播标量做逐 lane 加法。

## 机制

对每个活跃 lane `i`，执行 `dst[i] = src[i] + scalar`。标量操作数会被逻辑广播到每个活跃 lane。非活跃 lane 不参与计算。

## 语法

### PTO 汇编形式

```text
vadds %dst, %src, %scalar, %mask : !pto.vreg<NxT>, T
```

### AS Level 1（SSA）

```mlir
%result = pto.vadds %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>
```

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%input` | `!pto.vreg<NxT>` | 源向量寄存器 |
| `%scalar` | `T` | 广播到每个活跃 lane 的标量操作数 |
| `%mask` | `!pto.mask` | 谓词掩码；只有掩码位为 1 的 lane 参与运算 |

## 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%result` | `!pto.vreg<NxT>` | 活跃 lane 上得到逐 lane 和 |

## 副作用

这条指令除了产生目标值，没有其他架构副作用。

## 约束

- `%input` 与 `%result` 必须具有相同的向量宽度 `N` 和相同的元素类型 `T`。
- 掩码宽度必须与 `N` 一致。
- 只有活跃 lane 参与加法。

## 异常与非法情形

- verifier 会拒绝非法的操作数形状、不支持的元素类型以及不合法的属性组合。
- 约束部分列出的额外非法情形，同样属于 `pto.vadds` 的契约。

## 目标 Profile 限制

- 常见数值元素类型都可能支持；精确覆盖范围由目标 profile 决定。
- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 示例

```c
for (int i = 0; i < N; i++)
    if (mask[i])
        dst[i] = src[i] + scalar;
```

```mlir
%result = pto.vadds %values, %bias, %mask : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.vreg<64xf32>
```

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vadds`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vadds` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[向量-标量操作](../../vec-scalar-ops_zh.md)
- 下一条指令：[pto.vsubs](./vsubs_zh.md)
