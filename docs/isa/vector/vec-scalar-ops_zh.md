# 向量指令集：向量-标量操作

这里定义的是“一个向量寄存器 + 一个标量操作数”的 `pto.v*` 指令。标量如何广播到各 lane、carry 链怎样表示、inactive lane 怎么处理，都是架构可见语义。

> **类别：** 向量-标量运算
> **流水线：** `PIPE_V`

这组指令把一个统一的标量施加到每个活跃 lane 上，因此常被用来做 bias、scale、阈值裁剪、统一位移和一些轻量激活。

---

## 通用操作数模型

- `%input`：源向量寄存器
- `%scalar`：SSA 形式的标量操作数
- `%mask`：谓词操作数
- `%result`：目标向量寄存器

对 32 位标量形式，标量源还必须满足后端对这类指令的合法标量来源约束。

---

## 算术操作

### `pto.vadds`

- **语法：** `%result = pto.vadds %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 逐 lane 执行 `src[i] + scalar`。

### `pto.vsubs`

- **语法：** `%result = pto.vsubs %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 逐 lane 执行 `src[i] - scalar`。

### `pto.vmuls`

- **语法：** `%result = pto.vmuls %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 逐 lane 执行 `src[i] * scalar`。

### `pto.vmaxs`

- **语法：** `%result = pto.vmaxs %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 逐 lane 与同一个标量取最大值。

### `pto.vmins`

- **语法：** `%result = pto.vmins %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 逐 lane 与同一个标量取最小值。

这些指令的关键语义不是“标量先被编译器显式广播，再套用普通二元运算”，而是广播本身就是这条指令的定义。

---

## 位运算

### `pto.vands`

- **语法：** `%result = pto.vands %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 逐 lane 与统一标量做按位与。

### `pto.vors`

- **语法：** `%result = pto.vors %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 逐 lane 与统一标量做按位或。

### `pto.vxors`

- **语法：** `%result = pto.vxors %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 逐 lane 与统一标量做按位异或。

这三条只对整数元素类型合法。

---

## 位移

### `pto.vshls`

- **语法：** `%result = pto.vshls %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 把统一位移量应用到每个活跃 lane 的左移。

### `pto.vshrs`

- **语法：** `%result = pto.vshrs %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 把统一位移量应用到每个活跃 lane 的右移。

这两条与 `vshl` / `vshr` 的差别在于：后者的位移量来自第二个向量寄存器，前者是单一标量统一广播。

位移量最好不要超出元素位宽。

---

## 标量斜率激活

### `pto.vlrelu`

- **语法：** `%result = pto.vlrelu %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`
- **语义：** Leaky ReLU，其中 `%scalar` 是统一斜率。

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] >= 0) ? src[i] : scalar * src[i];
```

当前文档只覆盖 f16 与 f32 形式。

---

## carry / borrow 链

### `pto.vaddcs`

- **语法：** `%result, %carry = pto.vaddcs %lhs, %rhs, %carry_in, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- **语义：** 带 carry-in 和 carry-out 的加法。

```c
for (int i = 0; i < N; i++) {
    uint64_t r = (uint64_t)src0[i] + src1[i] + carry_in[i];
    dst[i] = (T)r;
    carry_out[i] = (r >> bitwidth);
}
```

### `pto.vsubcs`

- **语法：** `%result, %borrow = pto.vsubcs %lhs, %rhs, %borrow_in, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- **语义：** 带 borrow-in 和 borrow-out 的减法。

```c
for (int i = 0; i < N; i++) {
    dst[i] = src0[i] - src1[i] - borrow_in[i];
    borrow_out[i] = (src0[i] < src1[i] + borrow_in[i]);
}
```

这两条指令真正重要的是把多精度整数链条显式地挂到向量谓词上。lowering 时如果把 carry / borrow 谓词丢掉，就已经破坏了架构语义。

---

## 典型用法

```mlir
%biased = pto.vadds %activation, %bias_scalar, %mask
    : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.vreg<64xf32>

%scaled = pto.vmuls %input, %scale, %mask
    : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.vreg<64xf32>

%clamped_low = pto.vmaxs %input, %c0, %mask
    : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.vreg<64xf32>
%clamped = pto.vmins %clamped_low, %c255, %mask
    : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.vreg<64xf32>

%shifted = pto.vshrs %data, %c4, %mask
    : !pto.vreg<64xi32>, i32, !pto.mask -> !pto.vreg<64xi32>
```

---

## 相关页面

- [二元向量操作](./binary-vector-ops_zh.md)
- [一元向量操作](./unary-vector-ops_zh.md)
- [向量指令面](README_zh.md)
