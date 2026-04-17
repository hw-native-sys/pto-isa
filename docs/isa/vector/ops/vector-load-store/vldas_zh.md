# pto.vldas

`pto.vldas` 属于[向量加载与存储](../../vector-load-store_zh.md)指令集。

## 概述

为后续非对齐加载预热对齐缓冲状态。

## 机制

`pto.vldas` 会围绕 `%source` 所在的对齐块初始化加载对齐状态。它本身不是返回向量数据，而是返回一个 `!pto.align` 状态值，供后续 `pto.vldus` 串流使用。

## 语法

### PTO 汇编形式

```text
vldas %result, %source
```

### AS Level 1（SSA）

```mlir
%result = pto.vldas %source : !pto.ptr<T, ub> -> !pto.align
```

## 输入

- `%source`：用于初始化加载对齐状态的 UB 地址

## 预期输出

- `%result`：初始化好的 load-alignment 状态

## 副作用

这条指令会读取 UB 可见存储并返回 SSA 结果。它不会单独分配 buffer、发送事件，也不会建立栅栏。

## 约束

- `pto.vldas` 是同一条 `pto.vldus` 流的必需前导指令。
- `%source` 自身不要求 32B 对齐；硬件会先把它截到对齐块边界，再完成预热加载。

## 异常与非法情形

- 使用超出 UB 可见空间的地址，或违反所选形式的地址契约，都是非法的。
- 约束部分列出的额外非法情形，同样属于 `pto.vldas` 的契约。

## 目标 Profile 限制

- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 示例

```mlir
%align = pto.vldas %source : !pto.ptr<T, ub> -> !pto.align
```

## 性能

### 时延与吞吐披露

`pto.vldas` 当前可公开对齐的 VPTO 时序来源是 `~/visa.txt` 与 `PTOAS` `feature_vpto_backend` 分支上的 `docs/vpto-spec.md`。
这些来源**没有**给出 `pto.vldas` 本身的独立数字时延，但 `visa.txt` 对它所引导的非对齐加载流给出了明确吞吐约束。

| 指标 | 取值 | 来源依据 |
|------|------|----------|
| A5 初始化指令时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 后续非对齐加载吞吐 | 同一流中的后续非对齐加载指令达到 one-CPI | `visa.txt` §7.5 `VLDAS` |

因此，`pto.vldas` 的公开时序语义应理解为**流初始化指令**：已披露的是后续非对齐加载流的吞吐，而不是该 setup 指令自身的独立周期常数。

## 相关页面

- 指令集总览：[向量加载与存储](../../vector-load-store_zh.md)
- 上一条指令：[pto.vlds](./vlds_zh.md)
- 下一条指令：[pto.vldus](./vldus_zh.md)
