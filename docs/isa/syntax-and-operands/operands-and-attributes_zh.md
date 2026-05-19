# 操作数与属性

PTO VISA 操作围绕少量操作数类别展开：tile、全局内存视图、标量、谓词和同步值。属性与修饰符会细化操作行为，但不会取代操作数本身的合法性规则。

## 操作数类别

PTO 定义七种操作数类别：

| 类别 | SSA 类型 | C++ API | 说明 |
| --- | --- | --- | --- |
| **Tile** | `!pto.tile<...>` / `!pto.tile_buf<...>` | `Tile<...>` | 带 shape、layout、valid-region 元数据的 tile |
| **GlobalTensor** | `!pto.partition_tensor_view<...>` / `!pto.memref<...>` | `GlobalTensor<...>` | 面向 GM 的视图 |
| **Scalar** | `i8`–`i64`, `u8`–`u64`, `f16`, `bf16`, `f32` | 标准 C++ 标量类型 | 立即数或运行时标量 |
| **Predicate** | `!pto.mask<G>` | IR 层 | 控制向量 lane 参与的 mask |
| **Event** | `!pto.event` | `RecordEvent` | 顺序令牌 |
| **UB Pointer** | `!pto.ptr<T, ub>` | IR 层 | 指向 UB 的指针 |
| **GM Pointer** | `!pto.ptr<T, gm>` | `__gm__ T*` | 指向 GM 的指针 |

## 各类操作数

### Tile

Tile 是 `pto.t*` 的主要有效载荷类型。

```text
!pto.tile<loc=LOC, DTYPE, ROWS, COLS, BLAYOUT, SLAYOUT, FRACTAL, PAD>
```

### GlobalTensor

`GlobalTensor` 描述 GM 存储视图：

```text
!pto.partition_tensor_view<BxHxWxRxCxdtype>
```

### Scalar

标量可以是立即数或运行时值，出现在：

- PTO-AS 的直接立即数
- SSA 中的 `i32`、`i64`、`f32` 等
- C++ intrinsic 的标准标量参数

### Predicate

谓词 `!pto.mask<G>` 控制向量操作中哪些 lane 参与。

### UB Pointer

`!pto.ptr<T, ub>` 用于：

- `vlds`、`vsld`、`vgather2`、`vsts`、`vscatter`
- `copy_gm_to_ubuf`、`copy_ubuf_to_gm`

### GM Pointer

`!pto.ptr<T, gm>` 或 `__gm__ T*` 用于：

- 标量 load/store
- DMA copy

## 属性

属性会改变操作行为，但不会改变操作数类别。

### Compare 属性

| 属性 | 取值 | 说明 |
| --- | --- | --- |
| `cmp` | `"eq"`, `"ne"`, `"lt"`, `"le"`, `"gt"`, `"ge"` | 比较模式 |
| `cmpS` | 同上 | 标量比较变体 |

### 舍入模式

| 属性 | 取值 | 说明 |
| --- | --- | --- |
| `rnd` | `"rne"`, `"rz"`, `"rp"`, `"rm"` | nearest-even / toward-zero / toward +inf / toward -inf |

### Atomic 模式

| 属性 | 取值 | 说明 |
| --- | --- | --- |
| `atomic` | `"none"`, `"add"`, `"max"`, `"min"` | `pto.tstore` 的原子模式 |

### Transform 模式

| 属性 | 取值 | 说明 |
| --- | --- | --- |
| `mode` | `"hw"`, `"wh"`, `"cubic"` 等 | 由具体操作决定取值域 |

### Matmul Phase

| 属性 | 取值 | 说明 |
| --- | --- | --- |
| `phase` | `"relu"`, `"none"` | matmul 后处理阶段 |

### Distribution Mode

| 属性 | 取值 | 说明 |
| --- | --- | --- |
| `dist` | `"NORM"`, `"BRC_B8/B16/B32"`, `"US_B8/B16"`, `"DS_B8/B16"`, `"UNPK_B8/B16/B32"`, `"DINTLV_B32"`, `"SPLT2CHN_B8/B16"`, `"SPLT4CHN_B8"` | `vlds` / `vsts` 的分布模式 |

### Mask 相关属性

| 属性 | 取值 | 说明 |
| --- | --- | --- |
| `mask` | `"POST_UPDATE"`, `"NO_POST_UPDATE"` | masked store 后是否更新 alignment state |

## 操作数约束规则

### Tile 操作数约束

对二元 tile 操作 `optile(dst, src0, src1)`：

1. **类型兼容**：`dtype(src0) == dtype(src1) == dtype(dst)`，除非是显式转换操作
2. **形状兼容**：`shape(src0) == shape(src1) == shape(dst)`，不存在隐式广播
3. **布局兼容**：`BLayout + SLayout + Fractal` 必须满足当前指令集要求
4. **位置意图**：源和目标的 role 必须与指令匹配

### GlobalTensor 约束

对 `TLOAD(tile, tensor)`：

1. `sizeof(tile.dtype) == sizeof(tensor.dtype)`
2. `tensor.Layout` 必须与 tile 的布局要求兼容
3. shape 各维必须为正

### 谓词约束

对带 mask 的向量操作：

1. 谓词宽度必须匹配目标向量宽度
2. 谓词必须来自合法的谓词生成操作

### 立即数 / 标量约束

1. 立即数必须在其类型的可表示范围内
2. shift amount 必须非负且小于元素位宽
3. 广播必须由显式支持该行为的操作承担，例如 `tadds`

## 规则示例

当某条指令接受 tile 加一个标量属性时，合法性仍然同时取决于：

- tile tuple 是否合法
- 属性取值是否在文档化取值域内

合法 tile 不能修复非法属性；合法属性也不能修复非法 tile tuple。

## 契约说明

- 每个必需属性都必须定义允许的取值域
- 非法属性值必须产生确定性诊断
- 操作数角色与属性语义必须在 intrinsic、PTO-AS 和 per-op 页面之间保持一致
- 不存在隐式类型提升；需要时必须显式写出 `TCVT` / `vcvt`
- 广播必须由操作本身显式定义，例如 `tadds` 会广播标量，`tadd` 不会

## 相关页面

- [汇编拼写与操作数](./assembly-model_zh.md)
- [类型系统](../state-and-types/type-system_zh.md)
- [Tile 与有效区域](../programming-model/tiles-and-valid-regions_zh.md)
- [GlobalTensor 与数据搬运](../programming-model/globaltensor-and-data-movement_zh.md)
- [指令描述格式](../reference/format-of-instruction-descriptions_zh.md)
