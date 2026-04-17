# pto.vmov

`pto.vmov` 属于[一元向量操作](../../unary-vector-ops_zh.md)指令集。

## 概述

向量寄存器复制。

## 机制

`pto.vmov` 执行逐 lane 的寄存器复制：`dst[i] = src[i]`。带谓词的形式只复制活跃 lane，非活跃 lane 保持目标原值；不带谓词的形式则复制全部 lane。

## 语法

### PTO 汇编形式

```text
vmov %result, %input, %mask
```

### AS Level 1（SSA）

```mlir
%result = pto.vmov %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%input` | `!pto.vreg<NxT>` | 源向量寄存器；在每个活跃 lane 上读取 |
| `%mask` | `!pto.mask` | 谓词掩码；掩码位为 1 的 lane 为活跃 lane |

## 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%result` | `!pto.vreg<NxT>` | 活跃 lane 上得到逐 lane 拷贝；非活跃 lane 保持原值 |

## 副作用

这条指令除了产生 SSA 结果，没有其他架构副作用。

## 约束

- 带谓词的 `pto.vmov` 表现为 masked copy。
- 不带谓词的形式表现为完整寄存器复制。

## 异常与非法情形

- verifier 会拒绝非法的操作数形状、不支持的元素类型以及不合法的属性组合。
- 约束部分列出的额外非法情形，同样属于 `pto.vmov` 的契约。

## 目标 Profile 限制

- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。
- 任何依赖具体类型清单、分布模式或融合路径的代码，都应视为 profile 相关。

## 示例

### C 语义

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i];
```

### MLIR 用法

```mlir
%copy = pto.vmov %src, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vmov`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vmov` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[一元向量操作](../../unary-vector-ops_zh.md)
- 上一条指令：[pto.vcls](./vcls_zh.md)
