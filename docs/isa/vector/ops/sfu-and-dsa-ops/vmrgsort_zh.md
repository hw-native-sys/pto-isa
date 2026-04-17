# pto.vmrgsort

`pto.vmrgsort` 属于[SFU 与 DSA 操作](../../sfu-and-dsa-ops_zh.md)指令集。

## 概述

把 4 路已经排好序的输入段做归并排序。

## 机制

`pto.vmrgsort` 执行的是 4 路 merge。它从 UB 里读取 4 个已经排好序的输入段，按 `%config` 指定的顺序归并成一条有序输出流，再写回 `%dest`。

关键性质：

- 4 路输入必须事先已经按同一顺序排好
- 归并过程稳定，相等元素保持原始相对顺序
- `%config` 控制升序 / 降序及比较模式

## 语法

### PTO 汇编形式

```asm
vmrgsort4 %dest, %src0, %src1, %src2, %src3, %count, %config : !pto.ptr<T, ub>, !pto.ptr<T, ub>, !pto.ptr<T, ub>, !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64, i64
```

### AS Level 1（SSA）

```mlir
pto.vmrgsort4 %dest, %src0, %src1, %src2, %src3, %count, %config
    : !pto.ptr<f32, ub>, !pto.ptr<f32, ub>, !pto.ptr<f32, ub>, !pto.ptr<f32, ub>, !pto.ptr<f32, ub>, i64, i64
```

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%dest` | `!pto.ptr<T, ub>` | 归并结果写回的 UB 目标指针 |
| `%src0..%src3` | `!pto.ptr<T, ub>` | 四路预排序输入段 |
| `%count` | `i64` | 每一路输入段中的有效元素个数 |
| `%config` | `i64` | 控制升 / 降序与比较模式的配置字 |

## 预期输出

- 这条指令没有 SSA 结果；归并后的有序输出写回 `%dest`

## 副作用

这条指令会直接读写 UB 内存。

## 约束

- 四路输入都必须已经按 `%config` 指定的顺序排好。
- 四路输入必须使用相同的排序方向和比较模式。
- 所有输入与输出必须使用相同的元素类型 `T`。
- 所有指针都必须位于 UB 地址空间。

## 异常与非法情形

- 任一输入不是 UB 指针都属于非法。
- 有效地址若不满足对齐要求则非法。
- 约束部分列出的额外非法情形，同样属于 `pto.vmrgsort` 的契约。

## 目标 Profile 限制

- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。
- `%config` 的位域布局与支持的元素类型属于 profile 相关语义。

## 性能

### A5 时延

当前手册未给出 `vmrgsort4` 的独立周期级条目。

### A2/A3 吞吐

仓内 costmodel 为 `vmrgsort4` 记录了独立桶：

| 指标 | 数值 | 常量 |
|------|------|------|
| 启动时延 | 14 | `A2A3_STARTUP_BINARY` |
| 完成时延 | `A2A3_COMPL_DUP_VCOPY` | costmodel 常量 |
| 每次 repeat 吞吐 | 6 | `A2A3_RPT_6` |
| 流水间隔 | 18 | `A2A3_INTERVAL` |

测试注释还给出了一个更直观的经验公式：

- 多源（2/3/4 路）合并：`vmrgsort4(1) = startup(14) + 1*2 ≈ 16`
- 单源场景：`vmrgsort4(R) = 14 + R*2`

这说明 `vmrgsort4` 在代价模型里是明显的“归并专用路径”，不应按普通逐元素向量算术估算。

## 示例

```mlir
pto.vmrgsort4 %dest,
               %sorted_a, %sorted_b, %sorted_c, %sorted_d,
               %count, %config
    : !pto.ptr<f32, ub>, !pto.ptr<f32, ub>, !pto.ptr<f32, ub>,
      !pto.ptr<f32, ub>, !pto.ptr<f32, ub>, i64, i64
```

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vmrgsort`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vmrgsort` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[SFU 与 DSA 操作](../../sfu-and-dsa-ops_zh.md)
- 上一条指令：[pto.vsort32](./vsort32_zh.md)
- 下一条指令：[pto.vbitsort](./vbitsort_zh.md)
