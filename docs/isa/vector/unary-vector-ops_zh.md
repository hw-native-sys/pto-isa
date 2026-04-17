# 向量指令集：一元向量操作

这里定义的是带单个向量输入的 `pto.v*` 计算指令。除非某条指令明确说明了例外，否则向量形状、active-lane 语义和目标 profile 约束，都以这一页和对应 leaf 页为准。

> **类别：** 单输入向量运算
> **流水线：** `PIPE_V`

这组操作对每个活跃 lane 独立执行一元变换，常见于绝对值、取负、指数、对数、平方根、倒数以及少量按位与移动类操作。

---

## 通用操作数模型

- `%input`：源向量寄存器
- `%mask`：谓词操作数
- `%result`：目标向量寄存器

inactive lane 的处理随具体形式而定：有的形式会零填充，有的形式会保持原目标值不变。这里不把所有一元指令硬压成同一种掩码策略。

---

## 执行模型：`vecscope`

一元向量操作一般出现在 `pto.vecscope { ... }` 区域中，区域内指令都发往 `PIPE_V`。

```mlir
pto.vecscope {
  %remaining_init = arith.constant 1024 : i32
  %_:1 = scf.for %offset = %c0 to %total step %c64
      iter_args(%remaining = %remaining_init) -> (i32) {
    %mask, %next_remaining = pto.plt_b32 %remaining : i32 -> !pto.mask, i32
    %vec = pto.vlds %ub_in[%offset] : !pto.ptr -> !pto.vreg<64xf32>
    %out = pto.vabs %vec, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
    pto.vsts %out, %ub_out[%offset], %mask : !pto.vreg<64xf32>, !pto.ptr, !pto.mask
    scf.yield %next_remaining : i32
  }
}
```

这里常见的配套谓词生成方式有两种：

- `pto.pset_b32 "PAT_ALL"`：整向量全活跃
- `pto.plt_b32 %remaining`：按剩余元素数生成尾掩码

---

## A5 时延与吞吐

> 下表记录的是周期级模拟器上的 popped→retire 周期数。

| PTO 操作 | A5 RV（CA） | f32 | f16 | bf16 | i32 | i16 | i8 |
|----------|-------------|-----|-----|------|-----|-----|----|
| `pto.vabs` | `RV_VABS_FP` | 5 | 5 | — | 5 | 5 | 5 |
| `pto.vneg` | `RV_VMULS` | 8 | 8 | — | 8 | 8 | 8 |
| `pto.vexp` | `RV_VEXP` | 16 | 21 | — | — | — | — |
| `pto.vln` | `RV_VLN` | 18 | 23 | — | — | — | — |
| `pto.vsqrt` | `RV_VSQRT` | 17 | 22 | — | — | — | — |
| `pto.vrelu` | `RV_VRELU` | 5 | 5 | — | — | — | — |
| `pto.vrec` | `RV_VDIV` 路径综合 | 见注 | 见注 | — | — | — | — |
| `pto.vrsqrt` | `RV_VRSQRT` 路径 | 见注 | 见注 | — | — | — | — |
| `pto.vnot` | `RV_VNOT` | — | — | — | 5 | 5 | 5 |
| `pto.vbcnt` | — | — | — | — | 逐 lane | 逐 lane | 逐 lane |
| `pto.vcls` | — | — | — | — | 逐 lane | 逐 lane | 逐 lane |
| `pto.vmov` | `RV_VLD` 代理 | 9 | 9 | — | 9 | 9 | 9 |

`vrec` 由除法路径综合，`vrsqrt` 与 `vsqrt` 共用同一类硬件路径，因此它们的成本不应按“普通一元 ALU”理解。

## A2A3 时延与吞吐

| 指标 | 常量 | 周期值 | 适用范围 |
|------|------|--------|----------|
| 启动时延（归约 / 特殊函数） | `A2A3_STARTUP_REDUCE` | 13 | `vexp`、`vsqrt`、`vln` |
| 启动时延（普通算术） | `A2A3_STARTUP_BINARY` | 14 | `vabs`、`vneg` 等 |
| 完成时延：FP 二元 | `A2A3_COMPL_FP_BINOP` | 19 | `vabs`、`vneg` 等 f32 形式 |
| 完成时延：INT 二元 | `A2A3_COMPL_INT_BINOP` | 17 | `vabs` 等整型形式 |
| 完成时延：FP32 exp | `A2A3_COMPL_FP32_EXP` | 26 | `vexp`（f32） |
| 完成时延：FP16 exp | `A2A3_COMPL_FP16_EXP` | 28 | `vexp`（f16） |
| 完成时延：FP32 sqrt | `A2A3_COMPL_FP32_SQRT` | 27 | `vsqrt`（f32） |
| 完成时延：FP16 sqrt | `A2A3_COMPL_FP16_SQRT` | 29 | `vsqrt`（f16） |
| 每次 repeat 吞吐 | `A2A3_RPT_1` | 1 | 普通一元操作 |
| 每次 repeat 吞吐 | `A2A3_RPT_2` | 2 | 部分浮点路径 |
| 每次 repeat 吞吐 | `A2A3_RPT_4` | 4 | f16 特殊函数 |
| 流水间隔 | `A2A3_INTERVAL` | 18 | 全部向量操作 |
| 流水间隔（拷贝） | `A2A3_INTERVAL_VCOPY` | 13 | `vmov` |

---

## 算术与符号处理

### `pto.vabs`

- **语法：** `%result = pto.vabs %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 对每个活跃 lane 取绝对值。

整数最小负数的溢出行为由目标平台定义。

### `pto.vneg`

- **语法：** `%result = pto.vneg %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 对每个活跃 lane 做算术取负。

在 A5 上它借用标量乘法类硬件路径，因此延迟不像 `vabs` 那么低。

---

## 超越函数

### `pto.vexp`

- **语法：** `%result = pto.vexp %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 逐 lane 计算 `exp(input[i])`。

只对浮点类型合法。f16 的成本高于 f32，因此 softmax 等场景通常更偏好融合形式 `vexpdiff`。

### `pto.vln`

- **语法：** `%result = pto.vln %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 逐 lane 计算自然对数。

对实数语义而言，活跃输入最好严格大于 0；非正输入的异常或 NaN 行为由目标平台决定。

### `pto.vsqrt`

- **语法：** `%result = pto.vsqrt %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 逐 lane 求平方根。

负输入的行为由目标平台定义。

### `pto.vrsqrt`

- **语法：** `%result = pto.vrsqrt %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 逐 lane 求倒数平方根。

它与 `vsqrt` 共享一类硬件路径，因此成本也接近。

### `pto.vrec`

- **语法：** `%result = pto.vrec %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 逐 lane 求倒数。

本质上走的是除法路径，因此不该把它当成廉价的一元 ALU 操作。

---

## 激活

### `pto.vrelu`

- **语法：** `%result = pto.vrelu %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 逐 lane 执行 `max(input, 0)`。

这是 A5 上延迟最低的一元浮点操作之一。需要带斜率的版本时，应使用 `vlrelu`。

---

## 位操作

### `pto.vnot`

- **语法：** `%result = pto.vnot %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 逐 lane 按位取反。

仅对整数类型合法。

### `pto.vbcnt`

- **语法：** `%result = pto.vbcnt %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 逐 lane 统计设置位个数。

统计范围是“单个元素的位宽”，不是整个寄存器的总位数。

### `pto.vcls`

- **语法：** `%result = pto.vcls %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 逐 lane 统计前导符号位数量。

它依赖元素的有符号解释，因此 signedness 是语义的一部分。

---

## 数据移动

### `pto.vmov`

- **语法：** `%result = pto.vmov %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 受谓词控制的寄存器复制。

无谓词形式是完整寄存器复制；有谓词时则更接近 masked copy。

---

## 典型用法

```mlir
%sub = pto.vsub %x, %max_broadcast, %mask
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
%exp = pto.vexp %sub, %mask
    : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

%sum_rcp = pto.vrec %sum, %mask
    : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

%activated = pto.vrelu %linear_out, %mask
    : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

---

## 相关页面

- [SFU 与 DSA 操作](./sfu-and-dsa-ops_zh.md)
- [二元向量操作](./binary-vector-ops_zh.md)
- [向量指令面](../instruction-surfaces/vector-instructions_zh.md)
