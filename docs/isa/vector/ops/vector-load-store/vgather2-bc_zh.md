# pto.vgather2_bc

`pto.vgather2_bc` 属于[向量加载与存储](../../vector-load-store_zh.md)指令集。

## 概述

带广播行为、并受谓词控制的 gather 加载。

## 机制

`pto.vgather2_bc` 属于 PTO 的向量内存 / 数据搬运指令。它从 `%source` 和 `%offsets` 里做 gather，但额外受 `%mask` 控制：只有掩码打开的 lane 参与地址收集与数据加载。

## 语法

### PTO 汇编形式

```text
vgather2_bc %result, %source, %offsets, %mask
```

### AS Level 1（SSA）

```mlir
%result = pto.vgather2_bc %source, %offsets, %mask : !pto.ptr<T, ub>, !pto.vreg<NxI>, !pto.mask -> !pto.vreg<NxT>
```

## 输入

- `%source`：UB 基址
- `%offsets`：gather 索引
- `%mask`：限定哪些 lane 参与 gather 的谓词

## 预期输出

- `%result`：gather 得到的向量

## 副作用

这条指令会读取 UB 可见存储并返回 SSA 结果。

## 约束

- 被 mask 关闭的 lane 不参与地址合并，也不会触发地址越界异常。
- 被 mask 关闭的 lane 在结果里写零。

## 异常与非法情形

- 使用超出 UB 可见空间的地址，或违反所选形式的地址 / 对齐契约，都是非法的。
- 约束部分列出的额外非法情形，同样属于 `pto.vgather2_bc` 的契约。

## 目标 Profile 限制

- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 示例

```mlir
%result = pto.vgather2_bc %source, %offsets, %mask : !pto.ptr<T, ub>, !pto.vreg<NxI>, !pto.mask -> !pto.vreg<NxT>
```

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vgather2_bc`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vgather2_bc` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[向量加载与存储](../../vector-load-store_zh.md)
- 上一条指令：[pto.vgatherb](./vgatherb_zh.md)
- 下一条指令：[pto.vsts](./vsts_zh.md)
