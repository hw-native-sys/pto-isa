# pto.vexpdiff

`pto.vexpdiff` 属于[SFU 与 DSA 操作](../../sfu-and-dsa-ops_zh.md)指令集。

## 概述

计算 `exp(x - max)` 的融合形式，用于数值稳定的 softmax。

## 机制

`pto.vexpdiff` 把“先减去最大值，再做指数”这两个步骤压成一条指令：

```text
dst[i] = exp(input[i] - max[i])
```

它的价值不只是省一条指令，而是把 softmax 中最关键的数值稳定路径固定成硬件可以直接识别和优化的形式。

## 语法

### PTO 汇编形式

```text
vexpdiff %result, %input, %max
```

### AS Level 1（SSA）

```mlir
%result = pto.vexpdiff %input, %max : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>
```

A5 当前文档化支持的类型：`f16`、`f32`。

## 输入

- `%input`：源向量
- `%max`：广播后的减数项，通常是按行或按块求出的最大值

## 预期输出

- `%result`：融合后的 `exp(input - max)` 结果向量

## 副作用

这条指令除了产生 SSA 结果，没有其他架构副作用。

## 约束

- 只支持浮点元素类型。

## 异常与非法情形

- verifier 会拒绝非法的操作数形状、不支持的元素类型以及不合法的属性组合。
- 数值异常仍由目标 profile 决定。
- 约束部分列出的额外非法情形，同样属于 `pto.vexpdiff` 的契约。

## 目标 Profile 限制

- A5 当前文档化支持：`f16`、`f32`。
- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 性能

### A5 时延

当前手册未给出 `vexpdiff` 的独立周期表，但它正是为了替代 `vsub + vexp` 的两段式路径而存在，通常应视为 softmax 路径的优先形式。

### A2/A3 吞吐

仓内英文 leaf 页未单列独立 cost bucket；当前手册只给出这类 fused/SFU 操作的共用经验桶：

| 指标 | 数值 | 常量 |
|------|------|------|
| 启动时延 | 14 | `A2A3_STARTUP_BINARY` |
| 完成时延 | 26 | `A2A3_COMPL_FP32_EXP` |
| 每次 repeat 吞吐 | 2 | `A2A3_RPT_2` |
| 流水间隔 | 18 | `A2A3_INTERVAL` |

## 示例

```c
for (int i = 0; i < N; i++)
    dst[i] = expf(src[i] - max[i]);
```

## 相关页面

- 指令集总览：[SFU 与 DSA 操作](../../sfu-and-dsa-ops_zh.md)
- 上一条指令：[pto.vprelu](./vprelu_zh.md)
- 下一条指令：[pto.vaddrelu](./vaddrelu_zh.md)
