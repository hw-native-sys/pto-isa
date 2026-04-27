# 布局参考

**BLayout**、**SLayout**、**Fractal Layout**、**GlobalTensor Layout** 和 **Compact Mode** 构成 PTO 的标准布局参考。编程模型背景和 valid-region 语义见 [Tile 与有效区域](../programming-model/tiles-and-valid-regions_zh.md)。

## 两层布局维度

PTO 的布局分两层：

1. **GlobalTensor Layout**
   GlobalTensor 在 GM 中的布局方式，即 `GlobalTensor` 上的 `Layout::ND / DN / NZ`
2. **Tile Layout**
   Tile buffer 在 UB 或 TRF 中的内部组织方式，即 `BLayout + SLayout + Fractal`

两层布局在 `TLOAD` / `TSTORE` 时都必须兼容。

## GlobalTensor Layout（GM 视图）

`GlobalTensor` 是对 GM 的视图，其 layout 参数决定 GM 中的 stride 模式：

| Layout | Stride Pattern | 说明 | 用途 |
| --- | --- | --- | --- |
| `Layout::ND` | 行优先、C contiguous | 标准 row-major | 常规张量 |
| `Layout::DN` | 列优先、Fortran contiguous | 标准 column-major | 列优先张量 |
| `Layout::NZ` | 行优先分形布局 | 为分形 tile 兼容而设计 | A5 matmul LHS |

GM 布局必须与 tile 的内部布局兼容，具体规则见 [TLOAD](../tile/ops/memory-and-data-movement/tload_zh.md) 和 [TSTORE](../tile/ops/memory-and-data-movement/tstore_zh.md)。

## Block Layout（BLayout）

`BLayout` 描述 tile buffer 在行列方向上的存储顺序。

| BLayout | 行方向跨度 | 列方向跨度 | 心智模型 |
| --- | --- | --- | --- |
| `RowMajor` | `Cols` | `1` | C/C++/PyTorch |
| `ColMajor` | `1` | `Rows` | Fortran/Julia |

对形状 `(R, C)` 的 `RowMajor` tile：

$$ \mathrm{offset}(r, c) = (r \times C + c) \times \mathrm{sizeof(DType)} $$

对 `ColMajor` tile：

$$ \mathrm{offset}(r, c) = (c \times R + r) \times \mathrm{sizeof(DType)} $$

## Stripe Layout（SLayout）

`SLayout` 决定 tile 子元素是均匀矩形布局还是分形/跨步布局：

| SLayout | 说明 | 需要 |
| --- | --- | --- |
| `NoneBox` | 标准矩形 tile | 默认形式 |
| `RowMajor` | 行方向分形/跨步布局 | `Fractal ∈ {NZ, FR}` |
| `ColMajor` | 列方向分形/跨步布局 | `Fractal ∈ {ZN, RN}` |

## Fractal Layout

当 `SLayout != NoneBox` 时，`Fractal` 指定精确的分形或跨步模式。

对 `Fractal = NZ` 且 `SLayout = RowMajor`：

$$ \mathrm{offset}(r, c) = \mathrm{zigzag\_index}(r, c) \times \mathrm{sizeof(DType)} $$

zigzag 索引把二维坐标映射到一维 Z-order 顺序。它是硬件定义的，前端应负责地址生成，而不是在源码中手工计算分形偏移。

### Fractal 取值

| Fractal | SLayout | BLayout | 模式 | 常见用途 |
| --- | --- | --- | --- | --- |
| `None` | `NoneBox` | 任意 | 标准矩形布局 | 逐元素和常规计算 |
| `NZ` | `RowMajor` | `ColMajor` | Z-order row-major fractal | A5 matmul 左操作数 |
| `ZN` | `ColMajor` | `RowMajor` | Z-order col-major fractal | `NZ` 对称形式 |
| `FR` | `RowMajor` | `ColMajor` | Row-fractal | CUBE 专用 |
| `RN` | `ColMajor` | `RowMajor` | Row-N-fractal | CUBE 专用 |

## Compact Mode

Compact mode 处理物理 tile 维度大于 valid region 的情况，尤其常见于边界 matmul 和 `TEXTRACT` / `TINSERT`。

### 为什么重要

当矩阵维度不是 tile 尺寸的整数倍时，最后一块 tile 会包含 padding。Compact mode 决定：

1. padding 是否参与运算
2. 分形地址生成是否只覆盖有效元素
3. `TEXTRACT` / `TINSERT` 如何处理 partial tile

### TEXTRACT 中的模式

| Mode | 说明 | 行为 |
| --- | --- | --- |
| `ND2NZ` | 普通布局 → NZ 分形 | 有效数据按 Z-order 紧凑打包 |
| `NZ2ND` | NZ 分形 → 普通布局 | 有效数据解包回 row-major |
| `ND` | 普通布局 → 普通布局 | 直接拷贝 |
| `ND2NZ2` | 普通布局 → NZ，按 2 行分组 | 适用于特定 CUBE 访问模式 |

### TMATMUL_MX 中的 compact 行为

对 MX 格式 matmul，Left tile 使用带 compact addressing 的 NZ 分形布局。当矩阵在边界处不足完整 tile 尺寸时，地址生成只覆盖有效行，padding 行不会进入 CUBE 处理。

## TileType–Layout 兼容矩阵

并非所有 `TileType + BLayout + SLayout + Fractal` 组合都合法。

| TileType | 支持的 BLayout | 支持的 SLayout | 支持的 Fractal | 常见操作 |
| --- | --- | --- | --- | --- |
| `Vec` | `RowMajor`, `ColMajor` | `NoneBox` | `None` | `TADD`, `TMUL`, `TCVT`, `TLOAD/TSTORE` |
| `Mat` | `RowMajor`, `ColMajor` | `NoneBox` | `None` | `TGEMV`, `TGEMV_ACC`, `TGEMV_BIAS` |
| `Acc` | `RowMajor`, `ColMajor` | `NoneBox` | `None` | `TMATMUL`, `TMATMUL_ACC` 输出 |
| `Left` | `RowMajor` | `RowMajor` | `NZ` | `TMATMUL_MX` 左操作数 |
| `Right` | `RowMajor` | `NoneBox` | `NN`（隐式） | `TMATMUL_MX` 右操作数 |
| `Scalar` | `RowMajor` | `NoneBox` | `None` | 单元素 scalar tile |

不在此矩阵中的组合应视为非法 PTO 程序。

## Padding

`Pad` 参数控制 valid region 之外元素的填充值：

| Pad | 含义 |
| --- | --- |
| `Zero` | 域外元素置零 |
| `Null` | 域外元素未定义，不应读取 |
| `Invalid` | 元素标记为无效，读取无定义 |

## 布局转换模式

### Normal → Fractal

```cpp
using SrcTile = Tile<TileType::Vec, int8_t, 16, 16, RowMajor, NoneBox, None, Null>;
using DstTile = Tile<TileType::Left, int8_t, 16, 16, RowMajor, RowMajor, NZ, Null>;
TEXTRACT(dstLeft, srcVec, ExtractMode::ND2NZ);
```

### Fractal → Normal

```cpp
using SrcTile = Tile<TileType::Left, int8_t, 16, 16, RowMajor, RowMajor, NZ, Null>;
using DstTile = Tile<TileType::Vec, int8_t, 16, 16, RowMajor, NoneBox, None, Null>;
TINSERT(dstVec, srcLeft, InsertMode::NZ2ND);
```

## 常量参考

| 常量 | 值 | 单位 | 用途 |
| --- | --- | --- | --- |
| `BLOCK_BYTE_SIZE` | 32 | bytes | DMA block 传输单位 |
| `FIXP_BURST_UNIT_LEN` | 64 | half-words | DMA burst 长度 |
| `FRACTAL_NZ_ROW` | 16 | elements | NZ/ZN 分形行尺寸 |
| `CUBE_BLOCK_SIZE` | 512 | bytes | CUBE 分形块 |
| `MX_COL_LEN` | 2 | elements | MX matmul 列块长度 |
| `MX_ROW_LEN` | 16 | elements | MX matmul 行块长度 |
| `MX_BLOCK_SIZE` | 32 | elements | MX matmul 块大小 |

## 相关页面

- [Tile 与有效区域](../programming-model/tiles-and-valid-regions_zh.md)
- [类型系统](./type-system_zh.md)
- [TEXTRACT](../tile/ops/layout-and-rearrangement/textract_zh.md)
- [TINSERT](../tile/ops/layout-and-rearrangement/tinsert_zh.md)
- [Tile 指令集](../instruction-families/tile-families_zh.md)
