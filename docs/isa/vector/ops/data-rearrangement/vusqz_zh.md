# pto.vusqz

`pto.vusqz` 属于[数据重排](../../data-rearrangement_zh.md)指令集。

## 概述

把前部压紧的流重新展开到 mask 指定的位置。

## 机制

`pto.vusqz` 可以视为 `vsqz` 的反向放置形式。它消费一个“前部已经压紧”的隐式源流，然后按 `%mask` 的活跃位置把这些元素重新放回结果向量，对未选中的位置补零：

```c
int j = 0;
for (int i = 0; i < N; i++)
    if (mask[i]) dst[i] = src_front[j++];
    else dst[i] = 0;
```

这里的难点是：前部压紧流本身在当前指令形式里是隐式的，因此 lowering 不能擅自改变其放置顺序。

## 语法

### PTO 汇编形式

```text
vusqz %dst, %mask
```

### AS Level 1（SSA）

```mlir
%result = pto.vusqz %mask : !pto.mask -> !pto.vreg<NxT>
```

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%mask` | `!pto.mask` | 指定哪些 lane 应接收前部压紧流元素的谓词 |

## 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%result` | `!pto.vreg<NxT>` | 选中位置被填入、未选中位置补零的展开结果 |

## 副作用

这条指令除了产生目标值，没有其他架构副作用。

## 约束

- 前部压紧流是隐式源流。
- 活跃 / 非活跃位置的放置次序必须精确保留。
- 未选中的位置补零。

## 异常与非法情形

- verifier 会拒绝非法的操作数形状、不支持的元素类型以及不合法的属性组合。
- 约束部分列出的额外非法情形，同样属于 `pto.vusqz` 的契约。

## 目标 Profile 限制

- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 性能

当前仓内没有为 `vusqz` 单列周期表。若代码对具体延迟敏感，应把它视为目标 profile 相关的寄存器重排路径。

## 示例

```c
int j = 0;
for (int i = 0; i < N; i++)
    if (mask[i]) dst[i] = src_front[j++];
    else dst[i] = 0;
```

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vusqz`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vusqz` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[数据重排](../../data-rearrangement_zh.md)
- 上一条指令：[pto.vsqz](./vsqz_zh.md)
- 下一条指令：[pto.vperm](./vperm_zh.md)
