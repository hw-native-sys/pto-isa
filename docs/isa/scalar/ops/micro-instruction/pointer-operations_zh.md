# PTO 微指令：指针操作

本页记录 PTO 微指令层的指针相关操作：如何构造带类型的 PTO 指针，以及如何在 SSA 里做指针算术。当前主要对应 A5（Ascend 950）profile。

## 概览

PTO 微指令的内存操作数使用 `!pto.ptr<element-type, space>` 来区分地址空间。常见的空间有：

- GM：`!pto.ptr<T, gm>`
- 向量 tile buffer（当前硬件实现为 UB）：`!pto.ptr<T, ub>`

这些操作负责把“裸地址”变成带类型、带地址空间的 PTO 指针，并在其上执行元素级偏移与标量内存访问。

## 机制

这组操作把指针类型和指针算术显式建模到 SSA 里：

- `pto.castptr` 把标量地址解释为带类型的 PTO 指针
- `pto.addptr` 按“元素数”而不是按“字节数”做偏移
- `pto.load_scalar` / `pto.store_scalar` 负责单元素的标量访问

它们不会自动切换到向量 load/store 指令族，也不会隐式引入 DMA 或 tile 语义。

## 地址空间约定

| 空间 | 解释 |
|------|------|
| `gm` | 全局内存（GM），片外 HBM / DDR |
| `ub` | 向量 tile buffer（当前硬件实现为 Unified Buffer / UB） |

PTO 指针运算一律以 **元素数** 为单位，而不是以字节数为单位。

## `pto.castptr`

**语法**：`%result = pto.castptr %addr : i64 -> !pto.ptr<T, space>`

**语义**：把一个标量地址解释为目标地址空间中的带类型 PTO 指针。

```c
result = (ptr<T, space>)addr;
```

### 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%addr` | `i64` | 要转换的标量地址值 |

### 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%result` | `!pto.ptr<T, space>` | 目标地址空间中的带类型 PTO 指针 |

### 约束

- `pto.castptr` 只负责构造指针，不代表任何 load/store 副作用。
- `T` 与 `space` 必须和后续使用方式一致。
- 不同地址空间之间的合法性由后续操作和 target profile 决定。

### 示例

```mlir
%0 = pto.castptr %c0 : i64 -> !pto.ptr<f32, ub>
%gm_ptr = pto.castptr %gm_base : i64 -> !pto.ptr<bf16, gm>
```

## `pto.addptr`

**语法**：`%result = pto.addptr %ptr, %offset : !pto.ptr<T, space> -> !pto.ptr<T, space>`

**语义**：按元素偏移量计算新的指针值。

```c
result = ptr + offset;  // offset 以元素为单位，而不是字节
```

### 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%ptr` | `!pto.ptr<T, space>` | 基地址指针 |
| `%offset` | `index` 或 `i64` | 元素级偏移量 |

### 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%result` | `!pto.ptr<T, space>` | 前移后的新指针 |

### 约束

- `pto.addptr` 会保留原始元素类型 `T` 和地址空间 `space`。
- 偏移量按元素数解释。例如，`!pto.ptr<f32, ub>` 加 `1024`，实际会前进 `1024 × 4 = 4096` 字节。
- 超出边界后的地址行为由 target profile 决定。

### 示例

```mlir
%0 = pto.castptr %c0 : i64 -> !pto.ptr<f32, ub>
%1 = pto.addptr %0, %c1024 : !pto.ptr<f32, ub> -> !pto.ptr<f32, ub>
%row_ptr = pto.addptr %base_ptr, %row_offset : !pto.ptr<f32, ub> -> !pto.ptr<f32, ub>
```

## `pto.load_scalar`

**语法**：`%value = pto.load_scalar %ptr[%offset] : !pto.ptr<T, space> -> T`

**语义**：从指针型操作数中加载一个标量元素。

```c
value = ptr[offset];
```

### 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%ptr` | `!pto.ptr<T, space>` | 带类型 PTO 指针 |
| `%offset` | `index` | 以元素为单位的位移 |

### 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%value` | `T` | 读出的标量元素 |

### 约束

- 结果类型必须与 `%ptr` 的元素类型一致。
- 这是标量内存辅助操作，不会像 `pto.vlds` 那样产生 `vreg`。
- 主要合法空间是 `ub`；GM 标量 load 是否可用由目标实现决定。

### 示例

```mlir
%val = pto.load_scalar %ub_ptr[%c4] : !pto.ptr<f32, ub> -> f32
```

## `pto.store_scalar`

**语法**：`pto.store_scalar %value, %ptr[%offset] : !pto.ptr<T, space>, T`

**语义**：向指针型操作数中写入一个标量元素。

```c
ptr[offset] = value;
```

### 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%value` | `T` | 要写入的标量值 |
| `%ptr` | `!pto.ptr<T, space>` | 带类型 PTO 指针 |
| `%offset` | `index` | 以元素为单位的位移 |

### 约束

- 存储值类型必须与 `%ptr` 的元素类型一致。
- 这是标量内存辅助操作，不会像 `pto.vsts` 那样带 mask，也不属于向量 store `dist` 家族。
- 主要合法空间是 `ub`；GM 标量 store 是否可用由目标实现决定。

### 示例

```mlir
pto.store_scalar %val, %ub_ptr[%c8] : !pto.ptr<f32, ub>, f32
```

## 指针驱动的向量访问模式

下面的例子展示 PTO 指针如何和标量算术、结构化控制流、PTO 内存操作一起工作：

```mlir
%0 = pto.castptr %c0 : i64 -> !pto.ptr<f32, ub>
%1 = pto.addptr %0, %c1024 : !pto.ptr<f32, ub> -> !pto.ptr<f32, ub>

pto.vecscope {
  %16 = scf.for %arg3 = %c0 to %11 step %c64 iter_args(%arg4 = %12) -> (i32) {
    %mask, %scalar_out = pto.plt_b32 %arg4 : i32 -> !pto.mask<b32>, i32
    %s = pto.load_scalar %1[%c4] : !pto.ptr<f32, ub> -> f32
    pto.store_scalar %s, %1[%c8] : !pto.ptr<f32, ub>, f32
    %17 = pto.vlds %1[%arg3] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
    %18 = pto.vabs %17, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
    pto.vsts %18, %10[%arg3], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
    scf.yield %scalar_out : i32
  }
}
```

## 相关页面

- [BlockDim 与运行时查询](./block-dim-query_zh.md)
- [向量加载存储](../../../vector/vector-load-store_zh.md)
- [共享标量算术](../../shared-arith_zh.md)
