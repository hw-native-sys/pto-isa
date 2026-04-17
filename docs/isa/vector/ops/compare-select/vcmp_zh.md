# pto.vcmp

`pto.vcmp` 属于[比较与选择](../../compare-select_zh.md)指令集。

## 概述

`pto.vcmp` 是逐元素的向量对向量比较，输出一个谓词掩码。

## 机制

对每个 seed 掩码位为真的 lane `i`，`pto.vcmp` 会把 `CMP_MODE` 应用到 `src0[i]` 与 `src1[i]` 上，并把比较结果写进输出谓词。被 `%seed` 禁用的 lane 在结果掩码中写 0。

```text
result[i] = seed[i] ? cmp(src0[i], src1[i], CMP_MODE) : 0
```

## 语法

### PTO 汇编形式

```text
vcmp %dst, %src0, %src1, %seed, "CMP_MODE" : !pto.mask
```

### AS Level 1（SSA）

```mlir
%result = pto.vcmp %src0, %src1, %seed, "CMP_MODE" : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.mask
```

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%src0` | `!pto.vreg<NxT>` | 左操作数向量 |
| `%src1` | `!pto.vreg<NxT>` | 右操作数向量 |
| `%seed` | `!pto.mask` | 限定哪些 lane 真正参与比较的输入谓词 |
| `CMP_MODE` | 枚举 | 比较模式，如 `eq`、`ne`、`lt`、`le`、`gt`、`ge` |

## 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%result` | `!pto.mask` | 每个活跃 bit 记录比较结果的谓词掩码 |

## 副作用

这条指令除了产生目标谓词外，没有其他架构副作用。它不会隐式占用 buffer、发送事件，也不会建立内存栅栏。

## 约束

- `%src0` 与 `%src1` 必须有相同的向量宽度 `N` 和相同的元素类型 `T`。
- seed 掩码宽度必须与 `N` 相同。
- 浮点和整数比较遵循所选目标 profile 的类型特定比较规则。
- 支持的比较模式是 `eq`、`ne`、`lt`、`le`、`gt`、`ge`。

## 异常与非法情形

- verifier 会拒绝非法的操作数形状、不支持的元素类型以及不合法的属性组合。
- 约束部分列出的额外非法情形，同样属于 `pto.vcmp` 的契约。

## 目标 Profile 限制

- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。
- 如果代码依赖隐式谓词来源或某种目标专用编码变体，这类依赖都应视为 profile 相关。

## 示例

```c
for (int i = 0; i < N; i++)
    result[i] = seed[i] ? cmp(src0[i], src1[i], CMP_MODE) : 0;
```

```mlir
%lt_mask = pto.vcmp %a, %b, %all_active, "lt"
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.mask
```

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vcmp`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vcmp` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[比较与选择](../../compare-select_zh.md)
- 下一条指令：[pto.vcmps](./vcmps_zh.md)
