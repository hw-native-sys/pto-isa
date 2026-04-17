# pto.vldus

`pto.vldus` 属于[向量加载与存储](../../vector-load-store_zh.md)指令集。

## 概述

利用预热好的对齐状态执行非对齐向量加载。

## 机制

`pto.vldus` 属于 PTO 的向量内存 / 数据搬运指令。它显式消费一个 `!pto.align` 状态，并返回更新后的对齐状态和基址状态。这样一来，非对齐加载的状态推进是 SSA 可见的，而不是被隐藏在后端里。

## 语法

### PTO 汇编形式

```text
vldus %result, %align_out, %base_out, %source, %align
```

### AS Level 1（SSA）

```mlir
%result, %align_out, %base_out = pto.vldus %source, %align : !pto.ptr<T, ub>, !pto.align -> !pto.vreg<NxT>, !pto.align, !pto.ptr<T, ub>
```

## 输入

- `%source`：当前 UB 地址
- `%align`：来自 `pto.vldas` 或前一条 `pto.vldus` 的输入对齐状态

## 预期输出

- `%result`：拼装后的向量值
- `%align_out`：更新后的对齐状态
- `%base_out`：以 SSA 暴露出来的后更新基址状态

## 副作用

这条指令会读取 UB 可见存储并返回 SSA 结果。它不会单独分配 buffer、发送事件，也不会建立栅栏。

## 约束

- 同一条非对齐加载流的第一条 `pto.vldus` 前，必须先出现匹配的 `pto.vldas`。
- 对齐状态和基址都会沿着流前进，PTO 向量表示要求把这种推进显式暴露成 SSA 结果。

## 异常与非法情形

- 使用超出 UB 可见空间的地址，或违反所选形式的地址契约，都是非法的。
- 约束部分列出的额外非法情形，同样属于 `pto.vldus` 的契约。

## 目标 Profile 限制

- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 示例

```mlir
%align = pto.vldas %ub : !pto.ptr<f32, ub> -> !pto.align
%vec, %align2, %ub2 = pto.vldus %ub, %align
    : !pto.ptr<f32, ub>, !pto.align -> !pto.vreg<64xf32>, !pto.align, !pto.ptr<f32, ub>
```

## 详细说明

### 典型非对齐加载模式

```mlir
%align = pto.vldas %ub : !pto.ptr<f32, ub> -> !pto.align
%vec, %align2, %ub2 = pto.vldus %ub, %align
    : !pto.ptr<f32, ub>, !pto.align -> !pto.vreg<64xf32>, !pto.align, !pto.ptr<f32, ub>
```

## 性能

### 时延与吞吐披露

`pto.vldus` 当前可公开对齐的 VPTO 时序来源是 `~/visa.txt` 与 `PTOAS` `feature_vpto_backend` 分支上的 `docs/vpto-spec.md`。
这些来源**没有**给出 `pto.vldus` 的独立数字时延，但对由 `pto.vldas` 引导的非对齐加载流给出了吞吐约束，而 `pto.vldus` 就运行在该流中。

| 指标 | 取值 | 来源依据 |
|------|------|----------|
| A5 独立时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 与匹配 `pto.vldas` 配对后的流吞吐 | 每条后续非对齐加载指令达到 one-CPI | `visa.txt` §7.5 `VLDAS` / §7.7 `VLDUS` |

当文档或调度需要引用这一吞吐结论时，应把它理解为**已初始化非对齐加载流**的性质，而不是 `pto.vldus` 单独一条指令的独立时延保证。

## 相关页面

- 指令集总览：[向量加载与存储](../../vector-load-store_zh.md)
- 上一条指令：[pto.vldas](./vldas_zh.md)
- 下一条指令：[pto.vldx2](./vldx2_zh.md)
