# Tile 与有效区域

Tile 是 PTO 的主要有效载荷对象。大多数 `pto.t*` 语义都定义在 tile 上，因此 tile 的 shape、layout、角色和 valid-region 元数据都是架构可见的。

真实 kernel 很少把整个物理矩形都填满：边界 tile、partial block 和 padding 都很常见。如果 ISA 假装存储矩形中的每个元素都一样有意义，backend 与程序就会在沉默中分叉。PTO 因此显式携带 **valid rows / valid columns**（`Rv`, `Cv`），先在真正有意义的域上定义语义和合法性。

## 机制

### Tile 模板签名

```text
Tile<TileType, DType, Rows, Cols, BLayout, SLayout, Fractal, Pad>
```

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `TileType` | enum | 存储角色：`Vec`、`Mat`、`Acc`、`Scalar`、`Left`、`Right` |
| `DType` | C++ type | 元素类型 |
| `Rows` | 正整数 | 物理行数 |
| `Cols` | 正整数 | 物理列数 |
| `BLayout` | enum | block layout：`RowMajor` 或 `ColMajor` |
| `SLayout` | enum | stripe layout：`NoneBox` / `RowMajor` / `ColMajor` |
| `Fractal` | enum | 分形编码：`None`、`NZ`、`ZN`、`FR`、`RN` |
| `Pad` | enum | 域外填充值：`Zero`、`Null`、`Invalid` |

### TileType

`TileType` 决定该 tile 将由哪个执行流水处理：

| TileType | Pipeline | 常见用途 |
| --- | --- | --- |
| `Vec` | Vector Pipe | 通用逐元素、归约、转换、搬运 |
| `Mat` | Matrix / CUBE | matmul / gemv 输入 |
| `Acc` | Matrix accumulator | matmul 输出累加 |
| `Scalar` | Scalar Unit | `1×1` 标量 tile |
| `Left` | Matrix Pipe | `TMATMUL_MX` 左操作数 |
| `Right` | Matrix Pipe | `TMATMUL_MX` 右操作数 |

### Valid Region

Valid region 是架构可见的“哪些元素真的有意义”的声明。它由 `(Rv, Cv)` 表示，可通过 `tile.GetValidRow()` 和 `tile.GetValidCol()` 读取。

对任意 tile 操作，如果 `0 ≤ i < dst.Rv` 且 `0 ≤ j < dst.Cv`，则 `dst[i, j]` 才有架构定义。除非单独文档化，域外元素没有架构含义。

默认迭代域是 **目标 tile 的 valid region**：

```text
for i in [0, dst.Rv):
    for j in [0, dst.Cv):
        dst[i, j] = f(src0[i, j], src1[i, j], ...)
```

源 tile 在域外坐标上的读取是否发生，以及读到什么值，由具体操作定义；在没有明确说明时，这类值属于 implementation-defined。

### Block Layout

`BLayout` 决定 tile 内部的存储顺序。

| BLayout | 行方向跨度 | 列方向跨度 |
| --- | --- | --- |
| `RowMajor` | `Cols` | `1` |
| `ColMajor` | `1` | `Rows` |

### Stripe Layout

`SLayout` 决定 tile 是普通矩形布局还是分形/跨步布局：

| SLayout | 说明 |
| --- | --- |
| `NoneBox` | 标准矩形 tile |
| `RowMajor` | 行方向分形/跨步布局 |
| `ColMajor` | 列方向分形/跨步布局 |

### Fractal Layout

当 `SLayout != NoneBox` 时，`Fractal` 描述分形寻址模式：

| Fractal | 用途 |
| --- | --- |
| `None` | 普通矩形 tile |
| `NZ` | A5 matmul 左输入常见格式 |
| `ZN` | `NZ` 对称形式 |
| `FR` | CUBE 专用行分形 |
| `RN` | CUBE 专用列分形 |

### 按 TileType 的布局组合

| TileType | 支持的 BLayout | 支持的 SLayout | 支持的 Fractal | 常见操作 |
| --- | --- | --- | --- | --- |
| `Vec` | `RowMajor`, `ColMajor` | `NoneBox` | `None` | `TADD`, `TMUL`, `TCVT`, `TLOAD/TSTORE` |
| `Mat` | `RowMajor`, `ColMajor` | `NoneBox` | `None` | `TGEMV*` |
| `Acc` | `RowMajor`, `ColMajor` | `NoneBox` | `None` | `TMATMUL*` 输出 |
| `Left` | `RowMajor` | `RowMajor` | `NZ` | `TMATMUL_MX` 左输入 |
| `Right` | `RowMajor` | `NoneBox` | `NN`（隐式） | `TMATMUL_MX` 右输入 |
| `Scalar` | `RowMajor` | `NoneBox` | `None` | 标量 tile |

### Padding

| Pad | 含义 |
| --- | --- |
| `Zero` | 域外元素置零 |
| `Null` | 域外元素未定义，不应读取 |
| `Invalid` | 元素标记无效，读取无定义 |

## Compact Mode

当物理 tile 大于 valid region 时，compact mode 决定 padding 如何处理，这在边界 matmul 与 `TEXTRACT` / `TINSERT` 中尤为重要。

### TEXTRACT 中的模式

| Mode | 说明 |
| --- | --- |
| `ND2NZ` | 普通布局 → NZ 分形 |
| `NZ2ND` | NZ 分形 → 普通布局 |
| `ND` | 不做布局转换 |
| `ND2NZ2` | 类似 `ND2NZ`，但按 2 行分组 |

### TMATMUL_MX 中的 compact 行为

对 MX matmul，Left tile 的 NZ 分形布局在边界时只为有效行生成地址，padding 不参与 CUBE 处理。

## 输入

编程模型要求前端或程序提供：

- 合法的 tile 类型与布局组合
- 在边界或 partial tile 场景下明确的 `Rv / Cv`
- 彼此角色能正确组合的操作数

## 输出

产生 tile 的操作会输出带有 payload、valid region 和合法性约束的目标 tile。目标 `TileType` 与布局必须与指令兼容。

## 约束

- 语义只在 valid region 内成立，除非指令页另有定义
- 多输入 tile 操作默认按目标 valid region 迭代
- 单独一个合法的 `TileType` 还不够；shape、layout、location intent 和 target profile 都会参与判断
- `TileType + BLayout + SLayout + Fractal` 组合必须落在文档化支持集合内

## 不允许的情形

- 把域外元素当作稳定语义数据使用
- 假设 backend 会自动修补不兼容的 valid-region 用法
- 使用当前指令集或 target profile 不允许的 tile role / layout

## 示例

### 边界 tile

```cpp
using EdgeTile = Tile<TileType::Vec, half, 16, 16, RowMajor, NoneBox, None, Zero>;
EdgeTile tile;
tile.SetValidRegion(5, 9);
// 只有 tile[0..4][0..8] 在架构上有意义
```

### Matmul 角色

```cpp
using A = Tile<TileType::Left, int8_t, 16, 16, RowMajor, RowMajor, NZ, Null>;
using B = Tile<TileType::Right, int8_t, 16, 16, RowMajor, NoneBox, NN, Null>;
using C = Tile<TileType::Acc, int32_t, 16, 16, RowMajor, NoneBox, None, Zero>;
A a; B b; C c;
TMATMUL(c, a, b);
```

### 按目标 valid region 迭代的逐元素加法

```cpp
Tile<TileType::Vec, float, 16, 16> dst, src0, src1;
dst.SetValidRegion(8, 8);
TADD(dst, src0, src1);
```

## 相关页面

- [GlobalTensor 与数据搬运](./globaltensor-and-data-movement_zh.md)
- [类型系统](../state-and-types/type-system_zh.md)
- [布局参考](../state-and-types/layout_zh.md)
- [Tile 指令集](../instruction-families/tile-families_zh.md)
