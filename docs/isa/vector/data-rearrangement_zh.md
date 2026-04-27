# 向量指令集：数据重排

这里定义的是寄存器内、寄存器间的数据重排操作。它们会改动向量可见数据的排列方式，但不会退化成 tile 搬运或 DMA，因此仍属于 `pto.v*` 指令面。

> **类别：** 寄存器内数据移动与置换
> **流水线：** `PIPE_V`

这组操作覆盖交织 / 去交织、滑窗、压缩 / 展开、置换、pack / unpack 等场景。它们的共同点是：只处理寄存器内容，不直接访问向量 tile buffer。

---

## 通用操作数模型

- `%lhs` / `%rhs`：双输入重排的两个源寄存器
- `%src`：单输入重排的源寄存器
- `%result`：目标寄存器；少数指令会显式返回两个结果

因为这些指令不碰内存，所以页面的重点是“结果排列顺序”而不是“地址如何计算”。

---

## 交织与去交织

### `pto.vintlv`

- **语法：** `%low, %high = pto.vintlv %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- **语义：** 把两个源向量交织成两路输出。

```c
// low  = {src0[0], src1[0], src0[1], src1[1], ...}
// high = {src0[N/2], src1[N/2], src0[N/2+1], src1[N/2+1], ...}
```

两个输出构成一个成对结果，顺序必须保留。

### `pto.vdintlv`

- **语法：** `%low, %high = pto.vdintlv %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- **语义：** 把交织流拆回偶数 / 奇数位置。

```c
// low  = {src0[0], src0[2], src0[4], ...}
// high = {src0[1], src0[3], src0[5], ...}
```

这两条指令最容易被误写的点是“两个结果可交换”。它们不可交换，SSA 结果的顺序就是架构语义的一部分。

---

## slide / shift

### `pto.vslide`

- **语法：** `%result = pto.vslide %src0, %src1, %amt : !pto.vreg<NxT>, !pto.vreg<NxT>, i16 -> !pto.vreg<NxT>`
- **语义：** 先把 `%src1` 与 `%src0` 逻辑拼接，再从偏移 `%amt` 开始截出一个长度为 `N` 的窗口。

这类操作常用于滑窗、移位寄存器和跨向量边界的卷积准备。

### `pto.vshift`

- **语法：** `%result = pto.vshift %src, %amt : !pto.vreg<NxT>, i16 -> !pto.vreg<NxT>`
- **语义：** 单源 slide，空出来的位置按当前形式要求做零填充或其他规定填充。

这两条指令的关键语义不是“移多少位”，而是“逻辑窗口如何建立、从哪里截取”。lowering 不能偷偷改动源顺序。

---

## 压缩与展开

### `pto.vsqz`

- **语法：** `%result = pto.vsqz %src, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 把 mask 选中的活跃 lane 压紧到结果前部。

```c
int j = 0;
for (int i = 0; i < N; i++)
    if (mask[i]) dst[j++] = src[i];
while (j < N) dst[j++] = 0;
```

活跃元素之间的相对顺序必须与源 lane 顺序一致。

### `pto.vusqz`

- **语法：** `%result = pto.vusqz %mask : !pto.mask -> !pto.vreg<NxT>`
- **语义：** 把前部压紧流再按 `%mask` 的活跃位置展开回固定形状。

当前指令面把“前部压紧流”的来源隐含在形式之中，因此后端更不能随意改写其放置规则。

---

## 置换与选择

### `pto.vperm`

- **语法：** `%result = pto.vperm %src, %index : !pto.vreg<NxT>, !pto.vreg<NxI> -> !pto.vreg<NxT>`
- **语义：** 在寄存器内做表驱动置换。

```c
for (int i = 0; i < N; i++)
    dst[i] = src[index[i] % N];
```

它是寄存器内 permutation，不是从向量 tile buffer 读数据的 gather。

### `pto.vselr`

- **语法：** `%result = pto.vselr %src0, %src1 : !pto.vreg<NxT>, !pto.vreg<NxI> -> !pto.vreg<NxT>`
- **语义：** 带反向选择语义的寄存器选择。

这条指令也会在 compare/select 语境下被提到，但在这里强调的是它的重排属性。

---

## pack / unpack

### `pto.vpack`

- **语法：** `%result = pto.vpack %src0, %src1, %part : !pto.vreg<NxT_wide>, !pto.vreg<NxT_wide>, index -> !pto.vreg<2NxT_narrow>`
- **语义：** 把两个宽向量打包成一个窄向量。

```c
for (int i = 0; i < N; i++) {
    dst[i]     = truncate(src0[i]);
    dst[N + i] = truncate(src1[i]);
}
```

这是收窄操作，不是无损搬运。超出目标位宽的部分会按所选 pack 模式截断。

### `pto.vsunpack`

- **语法：** `%result = pto.vsunpack %src, %part : !pto.vreg<NxT_narrow>, index -> !pto.vreg<N/2xT_wide>`
- **语义：** 选择一半窄向量并做符号扩展。

### `pto.vzunpack`

- **语法：** `%result = pto.vzunpack %src, %part : !pto.vreg<NxT_narrow>, index -> !pto.vreg<N/2xT_wide>`
- **语义：** 选择一半窄向量并做零扩展。

`vsunpack` 与 `vzunpack` 的差别就在于扩展时是否保留符号位。

---

## 典型用法

```mlir
%even, %odd = pto.vdintlv %interleaved0, %interleaved1
    : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%pass_mask = pto.vcmps %values, %threshold, %all, "gt"
    : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.mask
%compacted = pto.vsqz %values, %pass_mask
    : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

%prev_window = pto.vslide %curr, %prev, %c1
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, i16 -> !pto.vreg<64xf32>
%window_sum = pto.vadd %curr, %prev_window, %all
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

%packed_i16 = pto.vpack %wide0_i32, %wide1_i32, %c0
    : !pto.vreg<64xi32>, !pto.vreg<64xi32>, index -> !pto.vreg<128xi16>
```

---

## V2 交织形式

### `pto.vintlvv2`

- **语法：** `%result = pto.vintlvv2 %lhs, %rhs, "PART" : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **语义：** V2 交织形式，但只把一半结果暴露为一个 SSA 值。

### `pto.vdintlvv2`

- **语法：** `%result = pto.vdintlvv2 %lhs, %rhs, "PART" : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **语义：** V2 去交织形式，同样只暴露一半结果。

这两条的限制是：结果并不是完整双路输出，而是只取其中一半，因此 `PART` 的选择必须被精确保留。

---

## 相关页面

- [向量加载与存储](./vector-load-store_zh.md)
- [谓词与物化](./predicate-and-materialization_zh.md)
- [向量指令面](README_zh.md)
