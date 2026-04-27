# 向量指令集：二元向量操作

这里定义的是带两个向量输入的 `pto.v*` 计算指令。逐 lane 运算的合法性、掩码约束以及某些平台特有的 carry 语义，都属于 PTO ISA 的架构契约，而不是外围文档里的“实现细节”。

> **类别：** 双输入向量运算
> **流水线：** `PIPE_V`

这组操作逐 lane 地消费两个向量寄存器，生成一个结果向量；少数 carry 变体还会同时返回一个谓词结果。

---

## 通用操作数模型

- `%lhs`、`%rhs`：两个源向量寄存器
- `%mask`：谓词操作数 `Pg`
- `%result`：目标向量寄存器

除非某条指令另有说明，否则 `%lhs`、`%rhs` 和 `%result` 必须具有相同的向量形状与元素类型。

---

## 执行模型：`vecscope`

二元向量运算通常出现在 `pto.vecscope { ... }` 中。这个区域建立了向量核心的执行上下文，区域内部所有向量指令都会发往 `PIPE_V`。

一个典型的生产者 / 消费者链条如下：

```mlir
pto.get_buf "PIPE_MTE2", %bufid, %c0 : i64, i64
pto.copy_gm_to_ubuf %gm_ptr, %ub_tile, ... : ...
pto.rls_buf "PIPE_MTE2", %bufid, %c0 : i64, i64

pto.get_buf "PIPE_V", %bufid, %c0 : i64, i64
pto.vecscope {
  scf.for %offset = %c0 to %N step %c64 iter_args(%remaining = %N_i32) -> (i32) {
    %mask, %next = pto.plt_b32 %remaining : i32 -> !pto.mask, i32
    %lhs = pto.vlds %ub_a[%offset] : !pto.ptr -> !pto.vreg<64xf32>
    %rhs = pto.vlds %ub_b[%offset] : !pto.ptr -> !pto.vreg<64xf32>
    %out = pto.vadd %lhs, %rhs, %mask
        : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
    pto.vsts %out, %ub_out[%offset], %mask : !pto.vreg<64xf32>, !pto.ptr, !pto.mask
    scf.yield %next : i32
  }
}
pto.rls_buf "PIPE_V", %bufid, %c0 : i64, i64
```

这里的关键不是 `vadd` 本身，而是 `get_buf` / `rls_buf` 把跨流水线的 RAW / WAR 依赖自动串起来了。向量运算页只负责说明“二元运算在 `PIPE_V` 里怎么工作”，不会把 DMA 语义混进来。

---

## A5 时延与吞吐

> 下表记录的是周期级模拟器上的 popped→retire 周期数。

| PTO 操作 | A5 RV（CA） | f32 | f16 | bf16 | i32 | i16 | i8 |
|----------|-------------|-----|-----|------|-----|-----|----|
| `pto.vadd` | `RV_VADD` | 7 | 7 | — | 7 | 7 | 7 |
| `pto.vsub` | `RV_VSUB` | 7 | 7 | — | 7 | 7 | 7 |
| `pto.vmul` | `RV_VMUL` | 8 | 8 | — | 8 | 8 | — |
| `pto.vdiv` | `RV_VDIV` | 17 | 22 | — | — | — | — |
| `pto.vmax` / `pto.vmin` | `RV_VMAX` | 7 | 7 | — | 7 | 7 | 7 |
| `pto.vand` / `pto.vor` / `pto.vxor` | `RV_VAND` | 7 | 7 | — | 7 | 7 | 7 |
| `pto.vshl` / `pto.vshr` | `RV_VSHL` | — | — | — | 7 | 7 | 7 |
| `pto.vaddc` | `RV_VADDC` | — | — | — | 7 | — | — |
| `pto.vsubc` | `RV_VSUBC` | — | — | — | 7 | — | — |

## A2A3 时延与吞吐

| 指标 | 常量 | 周期值 | 适用范围 |
|------|------|--------|----------|
| 启动时延（算术） | `A2A3_STARTUP_BINARY` | 14 | 全部算术类二元运算 |
| 完成时延：FP 二元运算 | `A2A3_COMPL_FP_BINOP` | 19 | `vadd` / `vsub`（f32） |
| 完成时延：INT 二元运算 | `A2A3_COMPL_INT_BINOP` | 17 | `vadd` / `vsub`（i16 / i32） |
| 完成时延：INT 乘法 | `A2A3_COMPL_INT_MUL` | 18 | `vmul` |
| 每次 repeat 吞吐 | `A2A3_RPT_1` | 1 | 简单位运算等 |
| 每次 repeat 吞吐 | `A2A3_RPT_2` | 2 | `vadd`、`vmul`、`vmax`、`vmin` |
| 每次 repeat 吞吐 | `A2A3_RPT_4` | 4 | 特殊函数类 |
| 流水间隔 | `A2A3_INTERVAL` | 18 | 全部向量运算 |

周期模型：

```text
total_cycles = startup + completion + repeats × per_repeat + (repeats - 1) × interval
```

---

## 算术操作

### `pto.vadd`

- **语法：** `%result = pto.vadd %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 逐 lane 相加。

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] + src1[i];
```

### `pto.vsub`

- **语法：** `%result = pto.vsub %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 逐 lane 相减。

`%lhs` 是被减数，`%rhs` 是减数。

### `pto.vmul`

- **语法：** `%result = pto.vmul %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 逐 lane 相乘。

当前 A5 文档没有把 `i8/u8` 形式纳入这条指令的常规范畴。整数溢出的精确行为由目标平台决定。

### `pto.vdiv`

- **语法：** `%result = pto.vdiv %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 逐 lane 相除。

这是最贵的常见二元运算之一。A5 上 f32 需要 17 周期，f16 需要 22 周期，显著高于乘法。如果精度允许，更推荐通过倒数与乘法来近似。

### `pto.vmax`

- **语法：** `%result = pto.vmax %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 逐 lane 取较大值。

### `pto.vmin`

- **语法：** `%result = pto.vmin %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 逐 lane 取较小值。

---

## 位运算

### `pto.vand`

- **语法：** `%result = pto.vand %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 逐 lane 按位与。

### `pto.vor`

- **语法：** `%result = pto.vor %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 逐 lane 按位或。

### `pto.vxor`

- **语法：** `%result = pto.vxor %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 逐 lane 按位异或。

这三条都只对整数元素类型合法。

---

## 位移操作

### `pto.vshl`

- **语法：** `%result = pto.vshl %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 逐 lane 左移。

这里右操作数 `%rhs` 不是一个统一的立即数，而是“每个 lane 自带一个位移量”的第二个向量寄存器。

### `pto.vshr`

- **语法：** `%result = pto.vshr %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 逐 lane 右移。

对有符号整数通常是算术右移，对无符号整数通常是逻辑右移；真正的行为由元素类型的 signedness 决定。

对这类指令，位移量最好保持在 `[0, bitwidth(T) - 1]` 范围内，超范围行为应视为目标定义行为。

---

## carry / borrow 链

### `pto.vaddc`

- **语法：** `%result, %carry = pto.vaddc %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- **语义：** 带 carry-out 的逐 lane 加法。

```c
for (int i = 0; i < N; i++) {
    uint64_t r = (uint64_t)src0[i] + src1[i];
    dst[i] = (T)r;
    carry[i] = (r >> bitwidth);
}
```

这条指令最重要的输出其实是 `%carry`，因为它让多精度加法可以沿着向量 lane 继续传播。

### `pto.vsubc`

- **语法：** `%result, %borrow = pto.vsubc %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- **语义：** 带 borrow-out 的逐 lane 减法。

```c
for (int i = 0; i < N; i++) {
    dst[i] = src0[i] - src1[i];
    borrow[i] = (src0[i] < src1[i]);
}
```

当前文档把这两条指令都视作无符号 carry-chain 指令。

---

## 典型用法

```mlir
%sum = pto.vadd %a, %b, %mask
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

%prod = pto.vmul %x, %y, %mask
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

%clamped_low = pto.vmax %input, %min_vec, %mask
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
%clamped = pto.vmin %clamped_low, %max_vec, %mask
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

%masked = pto.vand %data, %bitmask, %mask
    : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask -> !pto.vreg<64xi32>
```

---

## 相关页面

- [向量加载与存储](./vector-load-store_zh.md)
- [向量-标量操作](./vec-scalar-ops_zh.md)
- [向量指令面](README_zh.md)
