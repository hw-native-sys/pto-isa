# 向量指令集：谓词与物化

`pto.v*` 路径中的谓词寄存器与物化操作定义在这里。谓词装载 / 存储、掩码生成、谓词代数以及标量到向量的物化都属于体系结构可见行为，因为它们直接决定后续哪些 lane 参与运算，哪些 lane 保持不活跃。

> **类别：** 向量谓词寄存器、标量物化与掩码运算
> **流水线：** `PIPE_V`

谓词寄存器在 PTO ISA 中建模为 `!pto.mask`。当前文档采用 256 位谓词寄存器模型，既能保存逐 lane 条件，也能承载按 block 打包的活跃信息。

---

## 谓词装载

### `pto.plds`

- **语法：** `%result = pto.plds %source[%offset] {dist = "DIST"} : !pto.ptr<T, ub> -> !pto.mask`
- **语义：** 通过标量偏移，从向量 tile buffer 装载一个谓词寄存器。

支持的分布模式包括 `NORM`、`US`、`DS`。

```mlir
%mask = pto.plds %ub[%c0] {dist = "NORM"} : !pto.ptr<T, ub> -> !pto.mask
```

### `pto.pld`

- **语法：** `%result = pto.pld %source[%offset], "DIST" : !pto.ptr<T, ub>, index -> !pto.mask`
- **语义：** 通过 address-register 风格的偏移装载谓词。

### `pto.pldi`

- **语法：** `%result = pto.pldi %source, %offset, "DIST" : !pto.ptr<T, ub>, i32 -> !pto.mask`
- **语义：** 通过立即数偏移装载谓词。

---

## 谓词存储

### `pto.psts`

- **语法：** `pto.psts %value, %dest[%offset] : !pto.mask, !pto.ptr<T, ub>`
- **语义：** 用标量偏移把谓词寄存器写回向量 tile buffer。

```mlir
pto.psts %mask, %ub[%c0] : !pto.mask, !pto.ptr<T, ub>
```

### `pto.pst`

- **语法：** `pto.pst %value, %dest[%offset], "DIST" : !pto.mask, !pto.ptr<T, ub>, index`
- **语义：** 用 address-register 风格偏移写回谓词。

当前文档记录的分布模式包括 `NORM` 与 `PK`。

### `pto.psti`

- **语法：** `pto.psti %value, %dest, %offset, "DIST" : !pto.mask, !pto.ptr<T, ub>, i32`
- **语义：** 用立即数偏移写回谓词。

### `pto.pstu`

- **语法：** `%align_out, %base_out = pto.pstu %align_in, %value, %base : !pto.align, !pto.mask, !pto.ptr<T, ub> -> !pto.align, !pto.ptr<T, ub>`
- **语义：** 非对齐谓词存储形式，会更新对齐状态。

---

## 典型模式：谓词落地后再复用

很多向量内核会先通过比较生成谓词，再把这个谓词写回向量 tile buffer，供后续阶段复用：

```mlir
%mask = pto.vcmp %v0, %v1, %seed, "lt"
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.mask

pto.psts %mask, %ub_mask[%c0] : !pto.mask, !pto.ptr<T, ub>

// ... 之后在同一 kernel 或另一段向量循环中 ...

%saved_mask = pto.plds %ub_mask[%c0] {dist = "NORM"}
    : !pto.ptr<T, ub> -> !pto.mask

%result = pto.vsel %v_true, %v_false, %saved_mask
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

这说明 `!pto.mask` 不只是控制流影子，它也是可落地、可装载、可重新参与运算的向量数据类型。

---

## 标量物化

### `pto.vbr`

- **语法：** `%result = pto.vbr %value : T -> !pto.vreg<NxT>`
- **语义：** 把一个标量广播到目标向量寄存器的所有活跃 lane。

```c
for (int i = 0; i < N; i++)
    dst[i] = value;
```

约束：

- 当前文档记录的形式包括 `b8`、`b16`、`b32`
- 当使用 `b8` 时，只消费标量源的低 8 位

```mlir
%one = pto.vbr %c1_f32 : f32 -> !pto.vreg<64xf32>
```

### `pto.vdup`

- **语法：** `%result = pto.vdup %input {position = "POSITION"} : T|!pto.vreg<NxT> -> !pto.vreg<NxT>`
- **语义：** 把一个标量，或者一个已有向量中的指定元素，复制到所有 lane。

```c
for (int i = 0; i < N; i++)
    dst[i] = input_scalar_or_element;
```

这里的 `position` 由属性控制，用来说明“复制哪个元素”或“标量来自哪个位置”。PTO 的向量表示把这个选择器建模成属性，而不是额外的 SSA 操作数。

---

## 谓词生成

### `pto.pset_b8` / `pto.pset_b16` / `pto.pset_b32`

- **语法：** `%result = pto.pset_b32 "PAT_*" : !pto.mask`
- **语义：** 根据预定义模式生成谓词掩码。

| 模式 | 说明 |
|------|------|
| `PAT_ALL` | 所有 lane 有效 |
| `PAT_ALLF` | 所有 lane 无效 |
| `PAT_H` | 高半部分有效 |
| `PAT_Q` | 高四分之一有效 |
| `PAT_VL1` ... `PAT_VL128` | 前 N 个 lane 有效 |
| `PAT_M3`、`PAT_M4` | 周期性模模式 |

```mlir
%all_active = pto.pset_b32 "PAT_ALL" : !pto.mask
%first_16 = pto.pset_b32 "PAT_VL16" : !pto.mask
```

### `pto.pge_b8` / `pto.pge_b16` / `pto.pge_b32`

- **语法：** `%result = pto.pge_b32 "PAT_*" : !pto.mask`
- **语义：** 生成尾部掩码，常用于 remainder loop。

```c
for (int i = 0; i < TOTAL_LANES; i++)
    mask[i] = (i < len);
```

```mlir
%tail_mask = pto.pge_b32 "PAT_VL8" : !pto.mask
```

### `pto.plt_b8` / `pto.plt_b16` / `pto.plt_b32`

- **语法：** `%mask, %scalar_out = pto.plt_b32 %scalar : i32 -> !pto.mask, i32`
- **语义：** 一边生成谓词，一边返回更新后的标量状态。

这类形式常用于循环中持续消费“还剩多少元素”的计数器。

---

## 谓词逻辑运算

### `pto.pand`

- **语法：** `%result = pto.pand %src0, %src1, %mask : !pto.mask, !pto.mask, !pto.mask -> !pto.mask`
- **语义：** 谓词按位与。

```c
for (int i = 0; i < N; i++)
    if (mask[i]) dst[i] = src0[i] & src1[i];
```

### `pto.por`

- **语法：** `%result = pto.por %src0, %src1, %mask : !pto.mask, !pto.mask, !pto.mask -> !pto.mask`
- **语义：** 谓词按位或。

### `pto.pxor`

- **语法：** `%result = pto.pxor %src0, %src1, %mask : !pto.mask, !pto.mask, !pto.mask -> !pto.mask`
- **语义：** 谓词按位异或。

### `pto.pnot`

- **语法：** `%result = pto.pnot %input, %mask : !pto.mask, !pto.mask -> !pto.mask`
- **语义：** 在 `%mask` 允许的位上，对输入谓词做按位取反。

### `pto.psel`

- **语法：** `%result = pto.psel %src0, %src1, %sel : !pto.mask, !pto.mask, !pto.mask -> !pto.mask`
- **语义：** 谓词级多路选择。

```c
for (int i = 0; i < N; i++)
    dst[i] = sel[i] ? src0[i] : src1[i];
```

这类运算的重点不是算术，而是把“哪些 lane 继续参与后续运算”这件事变成可组合的数据流。

---

## 谓词搬运与重排

### `pto.ppack`

- **语法：** `%result = pto.ppack %input, "PART" : !pto.mask -> !pto.mask`
- **语义：** 对谓词做收窄打包。

支持的 `PART` 令牌包括 `LOWER` 与 `HIGHER`。

### `pto.punpack`

- **语法：** `%result = pto.punpack %input, "PART" : !pto.mask -> !pto.mask`
- **语义：** 对谓词做扩展拆包。

### `pto.pdintlv_b8`

- **语法：** `%low, %high = pto.pdintlv_b8 %src0, %src1 : !pto.mask, !pto.mask -> !pto.mask, !pto.mask`
- **语义：** 谓词去交织。

### `pto.pintlv_b16`

- **语法：** `%low, %high = pto.pintlv_b16 %src0, %src1 : !pto.mask, !pto.mask -> !pto.mask, !pto.mask`
- **语义：** 谓词交织。

这些形式常出现在按通道、按 block 重排数据时，与对应的向量 load/store 分布模式配合使用。

---

## 综合示例

```mlir
// 生成一个全活跃掩码
%all = pto.pset_b32 "PAT_ALL" : !pto.mask

// 生成尾掩码，例如最后 12 个元素里的有效部分
%tail = pto.pge_b32 "PAT_VL12" : !pto.mask

// 做比较，得到逐 lane 条件
%cmp_mask = pto.vcmp %a, %b, %all, "lt"
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.mask

// 把“比较通过”与“尾部范围内”组合起来
%combined = pto.pand %cmp_mask, %tail, %all
    : !pto.mask, !pto.mask, !pto.mask -> !pto.mask

// 用组合后的掩码驱动 predicated select
%result = pto.vsel %true_vals, %false_vals, %combined
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

---

## 相关页面

- [比较与选择](./compare-select_zh.md)
- [向量加载与存储](./vector-load-store_zh.md)
- [向量指令族](./vector-families_zh.md)
