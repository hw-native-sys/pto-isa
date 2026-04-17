# pto.vcvt

`pto.vcvt` 属于[转换操作](../../conversion-ops_zh.md)指令集。

## 概述

`pto.vcvt` 在不同数值类型之间做向量转换。它覆盖浮点到整数、浮点到浮点、整数到浮点、整数到整数这四大类转换，并支持可选的舍入模式、饱和模式和 `part` 放置控制。

它的结果向量元素类型可以与源不同；如果是宽度变化型转换，目标 lane 数也会随之变化。

## 机制

`pto.vcvt` 在向量寄存器模型内部改变元素解释、元素位宽、舍入方式或饱和行为，不需要退出向量寄存器语义。统一的 `pto.vcvt` 表面覆盖四类转换：

- 浮点转整数
- 浮点转浮点
- 整数转浮点
- 整数转整数

### 谓词与 zero-merge 行为

inactive lane 不参与转换，并在目标 lane 写 0：

```c
for (int i = 0; i < min(N, M); i++) {
    if (mask[i])
        dst[i] = convert(src[i], T0, T1, rnd, sat);
    else
        dst[i] = 0;
}
```

## 语法

### PTO 汇编形式

```assembly
PTO.vcvt  vd, vs, vmask, rnd, sat, part
```

其中：

- `vd`：目标向量寄存器
- `vs`：源向量寄存器
- `vmask`：谓词寄存器
- `rnd`：舍入模式，可选
- `sat`：饱和模式，可选
- `part`：宽度变化转换时的 even/odd 放置控制，可选

### MLIR SSA 形式

```mlir
%result = pto.vcvt %input, %mask {rnd = "RND", sat = "SAT", part = "PART"}
    : !pto.vreg<NxT0>, !pto.mask<G> -> !pto.vreg<MxT1>
```

### AS Level 1（SSA）

```mlir
%result = pto.vcvt %input, %mask {rnd = "R", sat = "SAT"}
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xi32>
```

### AS Level 2（DPS）

```mlir
PTO.vcvt  v0, v1, vmask, R, SAT, NONE
```

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%input` | `!pto.vreg<NxT0>` | 源向量寄存器，`T0` 为源元素类型，`N` 为 lane 数 |
| `%mask` | `!pto.mask<G>` | 执行掩码；`G` 必须匹配源向量家族的 typed-mask 粒度 |

### typed-mask 粒度

- `b32`：f32 / i32 家族
- `b16`：f16 / bf16 / i16 家族
- `b8`：i8 家族

VPTO 没有 `!pto.mask<b64>` 形式。

## 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%result` | `!pto.vreg<MxT1>` | 目标向量寄存器，`T1` 为目标元素类型；宽度变化型转换时 `M` 可以与 `N` 不同。inactive lane 写 0 |

## 副作用

`pto.vcvt` 除了产生 SSA 结果，没有其他架构副作用。它不会隐式占用 buffer、发送事件，也不会建立内存栅栏。

## 约束

### 类型约束

- 只有本页[支持的类型转换](#支持的类型转换)列出的源 / 目标类型对是合法的。
- 除非手册明确把某种类型对定义为 identity / no-op，否则源和目标类型必须不同。
- 宽度变化型转换会自动调整 lane 数，并满足：

```text
M * bitwidth(T1) = N * bitwidth(T0) = 2048
```

### 掩码约束

- 执行掩码必须使用匹配源向量家族的 typed-mask 粒度。
- VPTO 没有 `!pto.mask<b64>`。

### 属性约束

- `rnd`：只能取[舍入模式](#舍入模式)表中列出的模式。省略时默认 `"R"`。
- `sat`：`"SAT"`` 表示溢出时饱和；`"NOSAT"``（默认）表示溢出时 wrap 或产生目标定义行为。
- `part`：只对宽度变化型转换合法。`"EVEN"` 表示写到每个 lane 组的偶数位置，`"ODD"` 表示写到奇数位置。

## 异常与非法情形

- verifier 会拒绝非法的操作数形状、不支持的类型对、掩码粒度不匹配以及不合法的属性组合。
- 转换溢出而 `sat = "NOSAT"` 时，结果由目标平台定义。

## 目标 Profile 限制

- **A5**：当前手册记录了最完整的 `pto.vcvt` 表面；详见[支持的类型转换](#支持的类型转换)与[性能](#性能)。
- **CPU simulation**：在行为层面保留转换语义；浮点部分尽量遵循 IEEE 754。
- **A2/A3**：只保证子集支持；具体可用的类型对和属性组合由目标 profile 决定。

## 舍入模式

| 模式 | 名称 | 说明 |
|------|------|------|
| `R` | 就近舍入，ties to even | 默认模式 |
| `A` | 远离 0 舍入 | 朝绝对值更大的可表示值舍入 |
| `F` | Floor | 向负无穷舍入 |
| `C` | Ceil | 向正无穷舍入 |
| `Z` | Truncate | 向 0 舍入 |
| `O` | Round to odd | 舍入到最近的奇数可表示值 |

## 饱和模式

| 模式 | 说明 |
|------|------|
| `SAT` | 溢出时饱和到目标类型可表示的最大 / 最小值 |
| `NOSAT` | 默认；溢出时 wrap 或产生目标定义行为 |

## Part 模式

用于宽度变化型转换，把结果写到更宽目标布局中的偶 / 奇半：

| 模式 | 说明 |
|------|------|
| `EVEN` | 写到每个 lane 组的偶数位置 |
| `ODD` | 写到每个 lane 组的奇数位置 |

## 支持的类型转换

### 浮点转整数

| 形式 | 舍入 | 饱和 | 说明 |
|------|------|------|------|
| `!pto.vreg<64xf32>` → `!pto.vreg<32xsi64>` | 支持 | 支持 | f32 → i64 |
| `!pto.vreg<64xf32>` → `!pto.vreg<64xsi32>` | 支持 | 支持 | f32 → i32 |
| `!pto.vreg<64xf32>` → `!pto.vreg<128xsi16>` | 支持 | 支持 | f32 → i16 |
| `!pto.vreg<128xf16>` → `!pto.vreg<64xsi32>` | 支持 | 可选 | f16 → i32 |
| `!pto.vreg<128xf16>` → `!pto.vreg<128xsi16>` | 支持 | 可选 | f16 → i16 |
| `!pto.vreg<128xf16>` → `!pto.vreg<256xsi8>` | 支持 | 支持 | f16 → i8 |
| `!pto.vreg<128xf16>` → `!pto.vreg<256xui8>` | 支持 | 支持 | f16 → ui8 |
| `!pto.vreg<128xbf16>` → `!pto.vreg<64xsi32>` | 支持 | 支持 | bf16 → i32 |

### 浮点转浮点

| 形式 | Part | 说明 |
|------|------|------|
| `!pto.vreg<64xf32>` → `!pto.vreg<128xf16>` | 支持 | f32 → f16 |
| `!pto.vreg<64xf32>` → `!pto.vreg<128xbf16>` | 支持 | f32 → bf16 |
| `!pto.vreg<128xf16>` → `!pto.vreg<64xf32>` | 支持 | f16 → f32 |
| `!pto.vreg<128xbf16>` → `!pto.vreg<64xf32>` | 支持 | bf16 → f32 |

### 整数转浮点

| 形式 | 舍入 | 说明 |
|------|------|------|
| `!pto.vreg<256xui8>` → `!pto.vreg<128xf16>` | 否 | ui8 → f16 |
| `!pto.vreg<256xsi8>` → `!pto.vreg<128xf16>` | 否 | si8 → f16 |
| `!pto.vreg<128xsi16>` → `!pto.vreg<128xf16>` | 支持 | i16 → f16 |
| `!pto.vreg<128xsi16>` → `!pto.vreg<64xf32>` | 支持 | i16 → f32 |
| `!pto.vreg<64xsi32>` → `!pto.vreg<64xf32>` | 支持 | i32 → f32 |
| `!pto.vreg<64xui32>` → `!pto.vreg<64xf32>` | 支持 | ui32 → f32 |

### 整数转整数

| 形式 | 饱和 | Part | 说明 |
|------|------|------|------|
| `!pto.vreg<256xui8>` → `!pto.vreg<128xui16>` | 否 | 支持 | ui8 → ui16 |
| `!pto.vreg<256xui8>` → `!pto.vreg<64xui32>` | 否 | 支持 | ui8 → ui32 |
| `!pto.vreg<256xsi8>` → `!pto.vreg<128xsi16>` | 否 | 支持 | si8 → si16 |
| `!pto.vreg<256xsi8>` → `!pto.vreg<64xsi32>` | 否 | 支持 | si8 → si32 |
| `!pto.vreg<128xui16>` → `!pto.vreg<256xui8>` | 支持 | 支持 | ui16 → ui8 |
| `!pto.vreg<128xui16>` → `!pto.vreg<64xui32>` | 否 | 支持 | ui16 → ui32 |
| `!pto.vreg<128xsi16>` → `!pto.vreg<256xui8>` | 支持 | 支持 | si16 → ui8 |
| `!pto.vreg<128xsi16>` → `!pto.vreg<64xui32>` | 否 | 支持 | si16 → ui32 |
| `!pto.vreg<128xsi16>` → `!pto.vreg<64xsi32>` | 否 | 支持 | si16 → si32 |
| `!pto.vreg<64xui32>` → `!pto.vreg<256xui8>` | 支持 | 支持 | ui32 → ui8 |
| `!pto.vreg<64xui32>` → `!pto.vreg<128xui16>` | 支持 | 支持 | ui32 → ui16 |
| `!pto.vreg<64xui32>` → `!pto.vreg<128xsi16>` | 支持 | 支持 | ui32 → si16 |
| `!pto.vreg<64xsi32>` → `!pto.vreg<256xui8>` | 支持 | 支持 | si32 → ui8 |
| `!pto.vreg<64xsi32>` → `!pto.vreg<128xui16>` | 支持 | 支持 | si32 → ui16 |
| `!pto.vreg<64xsi32>` → `!pto.vreg<128xsi16>` | 支持 | 支持 | si32 → si16 |
| `!pto.vreg<64xsi32>` → `!pto.vreg<32xsi64>` | 否 | 支持 | si32 → si64 |

## 支持类型矩阵

下表是概览；真正的源 / 目标配对仍以上面的逐项表格为准。

| `src \\ dst` | `ui8` | `si8` | `ui16` | `si16` | `ui32` | `si32` | `si64` | `f16` | `f32` | `bf16` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ui8` | — | — | Y | — | Y | — | — | Y | — | — |
| `si8` | — | — | — | Y | — | Y | — | Y | — | — |
| `ui16` | Y | — | — | — | Y | — | — | — | — | — |
| `si16` | Y | — | — | — | Y | Y | — | Y | Y | — |
| `ui32` | Y | — | Y | Y | — | — | — | — | — | — |
| `si32` | Y | — | Y | Y | — | — | Y | — | Y | — |
| `si64` | — | — | — | — | — | — | — | — | — | — |
| `f16` | Y | Y | — | Y | — | Y | — | — | Y | — |
| `f32` | — | — | — | Y | — | Y | Y | Y | — | Y |
| `bf16` | — | — | — | — | — | Y | — | — | Y | — |

## 宽度变化型转换模式

当转换会改变 lane 宽度时，通常要用 even / odd 两个 part 组合：

```mlir
%even = pto.vcvt %in0, %mask {rnd = "R", sat = "SAT", part = "EVEN"}
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>
%odd  = pto.vcvt %in1, %mask {rnd = "R", sat = "SAT", part = "ODD"}
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>
%result = pto.vor %even, %odd, %mask
    : !pto.vreg<128xf16>, !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<128xf16>
```

## 性能

### A5

| PTO 操作 | RV（A5） | 说明 | 时延 |
|----------|----------|------|------|
| `pto.vcvt` | `RV_VCVT_F2F` | f32 → f16 | **7** 周期 |
| `pto.vcvt` | — | 其他转换对 | 由目标定义 |

只有代表性 trace 在手册里给出。其他转换对在 A5 上的具体 RV lowering 取决于 trace。CPU 模拟器和 A2/A3 的吞吐仍由目标定义。

## 示例

### C 语义

```c
// 浮点转整数：最近舍入 + 饱和
float src[64];
int dst[64];
int mask[64];

for (int i = 0; i < 64; i++) {
    if (mask[i]) {
        dst[i] = (int)roundf(src[i]);
    } else {
        dst[i] = 0;
    }
}
```

### MLIR SSA

```mlir
%result = pto.vcvt %input_f32, %mask {rnd = "R", sat = "SAT"}
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xi32>

%even = pto.vcvt %in0, %mask {rnd = "R", sat = "SAT", part = "EVEN"}
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>
%odd  = pto.vcvt %in1, %mask {rnd = "R", sat = "SAT", part = "ODD"}
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>

%float_result = pto.vcvt %input_i32, %mask
    : !pto.vreg<64xi32>, !pto.mask<b32> -> !pto.vreg<64xf32>
```

### 典型场景：量化

```mlir
%scaled = pto.vmul %input, %scale, %mask
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%quantized = pto.vcvt %scaled, %mask {rnd = "R", sat = "SAT"}
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xi32>
```

### 典型场景：混合精度

```mlir
%f32_vec = pto.vcvt %bf16_input, %mask {part = "EVEN"}
    : !pto.vreg<128xbf16>, !pto.mask<b16> -> !pto.vreg<64xf32>

%bf16_out = pto.vcvt %f32_result, %mask {rnd = "R", part = "EVEN"}
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xbf16>
```

## 详细说明

### 舍入模式选择建议

| 场景 | 推荐模式 |
|------|----------|
| 通用神经网络推理 | `"R"` |
| 向下取整 | `"F"` |
| 向上取整 | `"C"` |
| 向零截断 | `"Z"` |
| 需要确定性奇数舍入 | `"O"` |

### 属性使用建议

- `rnd`：当转换需要显式舍入规则时使用，尤其是浮点转整数、浮点收窄或整数转浮点。
- `mask`：控制哪些源 lane 参与转换；宽度变化型转换时，它会与 `part` 一起决定输出位置。
- `sat`：当目标类型范围可能溢出时使用。
- `part`：只用于宽度变化型转换，选择 even 或 odd 半区布局。

## 相关页面

- 指令集总览：[转换操作](../../conversion-ops_zh.md)
- 上一条指令：[pto.vci](./vci_zh.md)
- 下一条指令：[pto.vtrc](./vtrc_zh.md)
- 相关舍入指令：[pto.vtrc](./vtrc_zh.md)
