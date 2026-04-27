# 位置意图与合法性

PTO 的合法性不只取决于元素类型和 shape。许多操作还依赖值打算处于哪里、承担什么角色。位置意图分类和合法性检查流水如下。

## 位置意图分类

每个 tile 操作数都带有 **location intent**。它决定该 tile 由哪个执行流水处理，以及哪些操作对它是合法的。位置意图编码在 tile 类型中的 `loc=` 字段里。

### 位置意图取值

| 位置意图 | Pipeline | 说明 | 常见用途 |
| --- | --- | --- | --- |
| `loc=vec` | Vector Pipeline | 通用向量 tile | `TADD`、`TMUL`、`TCVT`、`TLOAD/TSTORE` |
| `loc=mat` | Matrix / CUBE | 矩阵乘输入 | `TGEMV` 等 |
| `loc=acc` | Matrix / CUBE | 累加器 / 输出 tile | `TMATMUL*` 输出 |
| `loc=left` | Matrix / CUBE | MX matmul 左操作数 | `TMATMUL_MX` 左输入 |
| `loc=right` | Matrix / CUBE | MX matmul 右操作数 | `TMATMUL_MX` 右输入 |
| `loc=scalar` | Scalar Unit | 标量 tile | 标量型 tile 操作 |

### 在类型中的写法

```text
!pto.tile<loc=vec, f32, 16, 16, RowMajor, NoneBox, None, Zero>
!pto.tile_buf<loc=left, int8, 16, 16, RowMajor, RowMajor, NZ, Null>
!pto.tile_buf<loc=acc, int32, 16, 16, RowMajor, NoneBox, None, Zero>
```

在 C++ API 中，位置意图由 `TileType` 模板参数表达：

```cpp
using VecTile = Tile<TileType::Vec, float, 16, 16>;
using AccTile = Tile<TileType::Acc, float, 16, 16>;
using LeftTile = Tile<TileType::Left, int8_t, 16, 16, RowMajor, RowMajor, NZ, Null>;
```

## 合法性检查流水

PTO 的合法性按四个阶段依次检查。程序只有通过四个阶段才算合法：

```text
Stage 1: TYPE CHECK
Stage 2: SHAPE CHECK
Stage 3: LAYOUT CHECK
Stage 4: TARGET PROFILE CHECK
```

### Stage 1: Type Check

元素类型必须与操作要求兼容。

对二元 tile 操作：

```text
dtype(src0) == dtype(src1) == dtype(dst)
```

对显式转换类操作，例如 `TCVT`，允许源和目标类型不同，但必须属于文档化的转换组。

### Stage 2: Shape Check

物理 shape 和 valid region 必须在该指令和目标 profile 允许的范围内：

```text
1 <= Rows <= MAX_ROWS(profile)
1 <= Cols <= MAX_COLS(profile)
0 <= Rv <= Rows
0 <= Cv <= Cols
```

### Stage 3: Layout Check

`BLayout + SLayout + Fractal` 组合必须对当前 `TileType` 和当前指令合法。

示例：

- `Vec` tile 使用 NZ 布局：非法
- `Left` tile 使用 `ColMajor`：非法
- `Mat` tile 使用分形布局：非法

### Stage 4: Target Profile Check

操作数的 `TileType`、元素类型和布局还必须被所选 target profile 支持。

示例：

- A2/A3 上使用 FP8：非法
- CPU 或 A2/A3 上使用 `Left/Right` 的 MX tile：非法

## 按指令集的合法性要求

### 逐元素 Tile-Tile

- 所有操作数必须为 `loc=vec`
- 布局组合必须与 `Vec` 兼容
- `dtype` 必须在该指令集支持列表内

### Matmul

- 左输入：`TileType::Left` 或 `TileType::Mat`
- 右输入：`TileType::Right` 或 `TileType::Mat`
- 累加器：`TileType::Acc`
- 形状必须满足 matmul 维度兼容关系

### 向量计算

- 操作数必须为 `!pto.vreg<NxDTYPE>`
- mask 宽度必须匹配向量宽度
- `dtype` 必须在目标 profile 支持列表内

## GM 侧操作数

GlobalTensor 操作数遵循单独的合法性路径：

| 检查 | 规则 |
| --- | --- |
| Dtype 大小 | `sizeof(tile.dtype) == sizeof(gtensor.dtype)` |
| 布局兼容性 | `gtensor.Layout` 必须与 tile 布局兼容 |
| Shape 正值 | 所有 shape 维度 > 0 |
| Valid region | `Rv > 0` 且 `Cv > 0` |

## 不允许的情形

- 在 tile 指令上偷用 vector-buffer 假设而没有显式桥接
- 把 location-sensitive 指令集写成“所有本地存储角色都等价”
- 用模糊的 implementation-defined 掩盖其实是 profile 缩窄的限制
- 把 CPU 模拟器的宽松行为当成 A5 合法性的证据

## 相关页面

- [类型系统](./type-system_zh.md)
- [Tile 与有效区域](../programming-model/tiles-and-valid-regions_zh.md)
- [Tile 指令集](../instruction-families/tile-families_zh.md)
- [向量指令集](../vector/README_zh.md)
- [可移植性与目标 Profile](../reference/portability-and-target-profiles_zh.md)
