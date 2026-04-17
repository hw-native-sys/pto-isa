# pto.vsts

`pto.vsts` 属于[向量加载与存储](../../vector-load-store_zh.md)指令集。

## 概述

带分布模式的向量存储。

## 机制

`pto.vsts` 属于 PTO 的向量内存 / 数据搬运指令。它把一个向量寄存器按指定分布模式写回 UB，同时把 UB 地址、分布模式以及 lane 到内存字节的映射关系保持为架构可见语义。

## 语法

### PTO 汇编形式

```text
vsts %value, %dest[%offset], %mask {dist = "DIST"}
```

### AS Level 1（SSA）

```mlir
pto.vsts %value, %dest[%offset], %mask {dist = "DIST"} : !pto.vreg<NxT>, !pto.ptr<T, ub>, !pto.mask
```

## 输入

- `%value`：源向量寄存器
- `%dest`：UB 基址
- `%offset`：位移
- `%mask`：选择活跃 lane 或子元素的谓词
- `DIST`：存储分布模式

## 预期输出

- 这条指令没有 SSA 结果；它会把数据写入 UB 内存

## 副作用

这条指令会写 UB 可见内存，某些有状态的非对齐流式形式还会推进对齐状态；尾部数据是否真正完成写出，可能还需要后续 flush 指令。

## 约束

- 有效目标地址必须满足所选存储模式的对齐规则。
- 打包 / 收窄模式可能只保留源向量中的部分比特。
- merge-channel 模式会把源向量解释成若干通道平面，再交织写回。

## 异常与非法情形

- 使用超出 UB 可见空间的地址，或违反所选分布模式的地址 / 对齐契约，都是非法的。
- 约束部分列出的额外非法情形，同样属于 `pto.vsts` 的契约。

## 目标 Profile 限制

- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 示例

```mlir
pto.vsts %v, %ub[%offset], %mask {dist = "NORM_B32"} : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask
```

## 详细说明

### 常见分布模式

| 模式 | 说明 | C 语义 |
|------|------|--------|
| `NORM_B8/B16/B32` | 连续存储 | `UB[base + i] = src[i]` |
| `PK_B16/B32` | 打包 / 收窄存储 | `UB_i16[base + 2*i] = truncate_16(src_i32[i])` |
| `MRG4CHN_B8` | 四通道合并 | 把四个 plane 交织为 RGBA |
| `MRG2CHN_B8/B16` | 双通道合并 | 把两个 plane 交织写回 |

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vsts`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vsts` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[向量加载与存储](../../vector-load-store_zh.md)
- 上一条指令：[pto.vgather2_bc](./vgather2-bc_zh.md)
- 下一条指令：[pto.vstx2](./vstx2_zh.md)
