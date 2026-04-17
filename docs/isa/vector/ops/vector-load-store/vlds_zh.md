# pto.vlds

`pto.vlds` 属于[向量加载与存储](../../vector-load-store_zh.md)指令集。

## 概述

带分布模式的向量加载。

## 机制

`pto.vlds` 属于 PTO 的向量内存 / 数据搬运指令。它把向量 tile buffer 中的数据按选定分布模式装成一个向量寄存器，同时把 UB 地址、分布模式以及映射规则保持为架构可见语义，而不是完全隐藏到后端 lowering 里。

## 语法

### PTO 汇编形式

```text
vlds %result, %source[%offset] {dist = "DIST"}
```

### AS Level 1（SSA）

```mlir
%result = pto.vlds %source[%offset] {dist = "DIST"} : !pto.ptr<T, ub> -> !pto.vreg<NxT>
```

## 输入

- `%source`：UB 基址
- `%offset`：加载位移
- `DIST`：分布模式，决定数据怎样映射到目标 lane

## 预期输出

- `%result`：装载得到的向量寄存器值

## 副作用

这条指令会读取 UB 可见存储并返回 SSA 结果。它不会单独分配 buffer、发送事件，也不会建立栅栏。

## 约束

- 有效地址必须满足所选分布模式的对齐规则。
- `NORM` 读取一个完整向量 footprint。
- 广播、上采样、下采样、拆包、拆通道和去交错等模式会改变“内存字节如何映射到目标 lane”，但不会改变“数据源来自 UB”这一点。

## 异常与非法情形

- 使用超出 UB 可见空间的地址，或违反所选分布模式的对齐 / 分布契约，都是非法的。
- 被屏蔽的 lane 或 inactive block，除非指令正文明确说明，否则不能把一个本来非法的地址变成合法。
- 约束部分列出的额外非法情形，同样属于 `pto.vlds` 的契约。

## 目标 Profile 限制

- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。
- 如果代码依赖某个分布模式、类型列表或特殊行为，应把这种依赖视为 profile 相关。

## 示例

```mlir
%v = pto.vlds %ub[%offset] {dist = "NORM"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
```

```mlir
%v = pto.vlds %ub[%c0] {dist = "BRC_B32"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
```

## 详细说明

### 常见分布模式

| 模式 | 说明 | C 语义 |
|------|------|--------|
| `NORM` | 连续 256B 装载 | `dst[i] = UB[base + i * sizeof(T)]` |
| `BRC_B8/B16/B32` | 单元素广播 | `dst[i] = UB[base]` |
| `US_B8/B16` | 上采样 | `dst[2*i] = dst[2*i+1] = UB[base + i]` |
| `DS_B8/B16` | 下采样 | `dst[i] = UB[base + 2*i]` |
| `UNPK_B8/B16/B32` | 拆包并零扩展 | `dst_i32[i] = (uint32_t)UB_i16[base + 2*i]` |
| `SPLT4CHN_B8` | 四通道拆分 | 每 4 字节抽 1 字节 |
| `SPLT2CHN_B8/B16` | 双通道拆分 | 每 2 个元素抽 1 个 |
| `DINTLV_B32` | 32 位去交错 | 取偶数位元素 |
| `BLK` | 分块装载 | 块状访问模式 |

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vlds`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vlds` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[向量加载与存储](../../vector-load-store_zh.md)
- 下一条指令：[pto.vldas](./vldas_zh.md)
