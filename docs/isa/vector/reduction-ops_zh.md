# 向量指令集：归约操作

`pto.v*` 的归约操作定义在这里。lane 分组方式、结果落位方式以及 inactive lane 的处理规则都属于可见契约，不能被后端当成“默认常识”随意改写。

> **类别：** 向量归约操作
> **流水线：** `PIPE_V`

这组操作把一个向量归约成标量结果，或者按硬件 VLane 分组归约成多个局部结果。

---

## 通用操作数模型

- `%input`：源向量寄存器
- `%mask`：谓词操作数 `Pg`，inactive lane 不参与归约
- `%result`：目标向量寄存器

归约结果统一写入目标向量的低位区域，其余位置按零填充规则处理。

---

## 执行模型：`vecscope`

归约操作通常在 `pto.vecscope { ... }` 中执行。跨 lane 的完整归约（如 `vcadd` / `vcmax` / `vcmin`）在一条指令内部完成树形规约；按 VLane 分组的归约（如 `vcgadd` / `vcgmax` / `vcgmin`）则只在每个 32 字节 VLane 内部做局部归约。

一个常见例子是按行求和，用于 softmax 分母计算：

```mlir
pto.vecscope {
  %active = pto.pset_b32 "PAT_ALL" : !pto.mask
  scf.for %row = %c0 to %row_count step %c1 {
    %vec = pto.vlds %ub_q[%row] : !pto.ptr -> !pto.vreg<64xf32>
    %row_sum_raw = pto.vcadd %vec, %active
        : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
    pto.vsts %row_sum_raw, %ub_sum[%row], %one_mask {dist = "1PT"} : ...
  }
}
```

可以把它理解成：

- `vcadd`：在整个向量上做树形求和，最后总和放到 lane 0
- `vcmax` / `vcmin`：在整个向量上做树形极值归约，同时保留位置
- `vcg*`：只在每个 VLane 里归约，每个 VLane 产出一个结果

如果组归约的结果还要继续参与下一段向量计算，通常需要配合 `pto.barrier #pto.pipe` 或同等同步手段，避免后续操作过早消费。

---

## A5 时延与吞吐

> 下表记录的是周期级模拟器上的 popped→retire 周期数。

### A5（Ascend 950 PR / DT）摘要

| PTO 操作 | A5 RV（CA） | f32 | f16 | i32 | i16 |
|----------|-------------|-----|-----|-----|-----|
| `pto.vcadd` | `RV_VCADD` | 19 | 21 | 19 | 17 |
| `pto.vcmax` / `pto.vcmin` | `RV_VCMIN` | 19 | 21 | 19 | 17 |
| `pto.vcpadd` | `RV_VCPADD` | 19 | 21 | — | — |
| `pto.vcgadd` | `RV_VCGADD` | 19 | 21 | 19 | 17 |
| `pto.vcgmax` / `pto.vcgmin` | `RV_VCGMAX` | 19 | 21 | 19 | 17 |

### A2A3（Ascend 910B / 910C）摘要

| 指标 | 常量 | 周期值 | 适用范围 |
|------|------|--------|----------|
| 启动时延 | `A2A3_STARTUP_REDUCE` | 13 | 所有归约 |
| 完成时延：FP 组归约（f16） | `A2A3_COMPL_FP_CGOP` | 21 | `vcgadd` / `vcgmax` / `vcgmin`（f16） |
| 完成时延：FP 归约（f32） | `A2A3_COMPL_FP_BINOP` | 19 | `vcadd` / `vcmax` / `vcmin`（f32） |
| 完成时延：INT 归约（i16） | `A2A3_COMPL_INT_BINOP` | 17 | 全部 INT16 归约 |
| 完成时延：INT / FP32 | `A2A3_COMPL_FP_BINOP` | 19 | INT32 / FP32 归约 |
| 每次 repeat 吞吐 | `A2A3_RPT_1` | 1 | INT16 组归约 |
| 每次 repeat 吞吐 | `A2A3_RPT_2` | 2 | INT32 / FP32 / FP16 归约 |
| 流水间隔 | `A2A3_INTERVAL` | 18 | 所有向量操作 |

周期模型可写成：

```text
total_cycles = startup + completion + repeats × per_repeat + (repeats - 1) × interval
```

---

## 完整向量归约

### `pto.vcadd`

- **语法：** `%result = pto.vcadd %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 类型：** i16-i64、f16、f32
- **语义：** 对整个向量求和，结果写到 lane 0，其他位置清零。

```c
T sum = 0;
for (int i = 0; i < N; i++)
    sum += src[i];
dst[0] = sum;
for (int i = 1; i < N; i++)
    dst[i] = 0;
```

约束：

- `%mask` 只允许活跃 lane 参与求和
- 如果所有谓词位都为 0，结果为 0
- 某些窄整数形式可能在内部做扩宽累加

### `pto.vcmax`

- **语法：** `%result = pto.vcmax %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 类型：** i16-i32、f16、f32
- **语义：** 在整个向量里找最大值，同时产出 argmax 信息。

```c
T mx = -INF; int idx = 0;
for (int i = 0; i < N; i++)
    if (src[i] > mx) { mx = src[i]; idx = i; }
dst_val[0] = mx;
dst_idx[0] = idx;
```

这里的值与索引如何打包进目标向量，取决于具体形式，但“值和位置信息一起保留”这一点是架构可见语义。

### `pto.vcmin`

- **语法：** `%result = pto.vcmin %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 类型：** i16-i32、f16、f32
- **语义：** 在整个向量里找最小值，同时产出 argmin 信息。

和 `vcmax` 一样，值与位置的精确打包方式取决于具体形式，但后端必须稳定保留这种打包约定。

---

## 按 VLane 分组的归约

向量寄存器按 32 字节 VLane 分成 8 组。组归约只在每个 VLane 内部做局部规约，不跨组传播。

```text
以 f32 64 元素向量为例：
VLane 0: [0..7]    VLane 1: [8..15]   VLane 2: [16..23]  VLane 3: [24..31]
VLane 4: [32..39]  VLane 5: [40..47]  VLane 6: [48..55]  VLane 7: [56..63]
```

### `pto.vcgadd`

- **语法：** `%result = pto.vcgadd %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 类型：** i16-i32、f16、f32
- **语义：** 在每个 VLane 内求和，每组产生一个结果。

```c
int K = N / 8;
for (int g = 0; g < 8; g++) {
    T sum = 0;
    for (int i = 0; i < K; i++)
        sum += src[g*K + i];
    dst[g*K] = sum;
    for (int i = 1; i < K; i++)
        dst[g*K + i] = 0;
}
```

以 f32 为例，8 个结果分别出现在索引 `0, 8, 16, 24, 32, 40, 48, 56`。

### `pto.vcgmax`

- **语法：** `%result = pto.vcgmax %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 在每个 VLane 内求局部最大值。

### `pto.vcgmin`

- **语法：** `%result = pto.vcgmin %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 在每个 VLane 内求局部最小值。

这三条指令最容易被误解的地方是：它们不是“把寄存器拆成任意八段”的软件概念，而是严格按硬件 32 字节 VLane 分组。

---

## 前缀归约

### `pto.vcpadd`

- **语法：** `%result = pto.vcpadd %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 类型：** f16、f32
- **语义：** 做 inclusive prefix sum。

```c
dst[0] = src[0];
for (int i = 1; i < N; i++)
    dst[i] = dst[i - 1] + src[i];
```

示例：

```text
输入： [1, 2, 3, 4, 5, ...]
输出： [1, 3, 6, 10, 15, ...]
```

当前文档只给出浮点形式。

---

## 典型用法

```mlir
// Softmax：先找最大值，做数值稳定化
%max_vec = pto.vcmax %logits, %mask
    : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// max 位于 lane 0，可再配合广播用到全向量
%max_broadcast = pto.vlds %ub_tmp[%c0] {dist = "BRC_B32"}
    : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

// 用 vcgadd 做按组求和
%row_sums = pto.vcgadd %tile, %mask
    : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// 对整向量求和
%total = pto.vcadd %values, %mask
    : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// 做前缀和
%cdf = pto.vcpadd %pdf, %mask
    : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

---

## 相关页面

- [向量指令面](../instruction-surfaces/vector-instructions_zh.md)
- [向量加载与存储](./vector-load-store_zh.md)
- [SFU 与 DSA 操作](./sfu-and-dsa-ops_zh.md)
