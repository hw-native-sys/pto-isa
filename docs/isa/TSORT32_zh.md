# TSort32

## 指令示意图

![TSort32 tile operation](../figures/isa/TSort32.svg)

## 简介

对 `src` 的每个 32 元素块，与 `idx` 中对应的索引一起进行排序，并将排序后的值-索引对写入 `dst`。底层 SFU 指令为 **VBS32**（`vbitsort`），单次调用可排序一个或多个独立的 32 元素列表。

## 硬件：VBS32（`vbitsort`）

VBS32 运行在 **SFU**（非向量流水线）上。一次调用排序 `repeat` 个连续的 32 元素块，每个块由 32 个值 + 32 个索引组成，打包为值-索引对：

```cpp
void vbitsort(__ubuf__ T *dst,        // 排序后的值-索引对输出
              __ubuf__ T *src0,        // 每块 32 个值 × repeat
              __ubuf__ uint32_t *src1, // 每块 32 个索引 × repeat
              uint8_t repeat);         // 32 元素块的数量（1..255）
```

- `repeat`（上限 `REPEAT_MAX = 255`）打包到 `config[63:56]`。
- 块在内存中**连续分布，步长为 32 个元素**：块 `b` 读取 `src0[b*32 : b*32+32]` 和 `src1[b*32 : b*32+32]`，写入 `dst[b*32*coef : ...]`，其中 `coef` = 2（float）或 4（half）——值-索引对的扩展因子。
- 排序顺序：按值**降序**；相同值时索引小者优先。

## 数学语义

对每一行 `r`，`src` 按独立的 32 元素块处理。设块 `b` 覆盖列 `32b … 32b+31`，`n_b = min(32, C - 32b)` 为其有效元素数。

$$
(v_k, i_k) = (\mathrm{src}_{r,32b+k},\; \mathrm{idx}_{r,32b+k}), \quad 0 \le k < n_b
$$

按值降序排序，输出重排后的序列：

$$
[(v_{\pi(0)}, i_{\pi(0)}),\; (v_{\pi(1)}, i_{\pi(1)}),\; \ldots]
$$

其中 `π` 为该块的排序置换。

注：
- `idx` 是输入 Tile（索引随值一起被重排），不是输出。
- `dst` 存储排序后的值-索引对，而非仅排序后的值。

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
// 3 参数：src 必须 32 对齐（validCol % 32 == 0）
template <typename DstTileData, typename SrcTileData, typename IdxTileData>
PTO_INST RecordEvent TSort32(DstTileData &dst, SrcTileData &src, IdxTileData &idx);

// 4 参数：支持非 32 对齐尾部（validCol % 32 != 0），通过 tmp 填充
template <typename DstTileData, typename SrcTileData, typename IdxTileData, typename TmpTileData>
PTO_INST RecordEvent TSort32(DstTileData &dst, SrcTileData &src, IdxTileData &idx, TmpTileData &tmp);
```

## Tile 尺寸与数据类型

对于 `src` 形状为 $R \times C$（有效区域）、块大小 32：

| Tile | dtype | 尺寸（元素数） | 说明 |
|------|-------|----------------|-------|
| `src` | `half` 或 `float`（$T$） | $R \times C$ | 待排序的值 |
| `idx` | `uint32_t` | $R \times C$（或 $1 \times C$ 广播） | 随值重排的索引 |
| `dst` | $T$ | $R \times (2C)$ float，$R \times (4C)$ half | 排序后的值-索引对（见下方扩展因子） |
| `tmp`（仅 4 参数） | $T$ | 见下方 tmp 尺寸公式 | 尾部填充 scratch |

**`dst` 扩展因子**（`typeCoef`）：每个输入元素生成一个 8 字节的 tuple `[value (4 B), index (4 B)]`——`float` 的 value 占满 4 B；`half` 的 2 B value 零扩展至 4 B。因此 `dst` 恒为 $C \times 8$ 字节。

| dtype | 每个 `src` 列对应的 `dst` 列数（dtype 单位） | tuple 布局 | 字节/tuple |
|-------|-------------------------------------------|--------------|------------|
| `float` | ×2（2 个 float 槽位） | `[value_f32, index_u32]` | 8 |
| `half` | ×4（4 个 half 槽位） | `[value_f16, 0x0000, index_u32]` | 8 |

## 约束

| 约束 | 原因 |
|------------|------|
| `dst`/`src` dtype = `half` 或 `float`（须一致）；`idx` = `uint32_t` | VBS32 类型分派 |
| 所有 Tile 为 `TileType::Vec`、`BLayout::RowMajor` | SFU 寻址 |
| `validCol % 32 == 0`（3 参数） | 每块恰为 32 个元素 |
| `validCol` 任意（4 参数） | 尾块通过 `tmp` 填充至 32，填充值为 $-\infty$ |
| `repeat = validCol/32`（3 参数）或 `ceil(validCol/32)`（4 参数） | VBS32 repeat 计数，每次调用 ≤ 255；更大的 `validCol` 拆分为多次 `vbitsort` 调用 |
| `tmp`（4 参数）≥ `tmpSize` 元素（见下方公式） | 保存填充后的行/尾块副本 |
| 无 `WaitEvents&...` / 无内部 `TSYNC` | 如需同步须显式调用 |

### `tmp` 尺寸公式（4 参数）

设 $C$ = `validCol`，$B$ = `sizeof(T)` 字节数，$G$ = 32（块大小）。实现根据整行（字节数）是否满足 `MAX_UB_TMP = 8160` 进行分支：

$$
\mathrm{tmpSize} =
\begin{cases}
\mathrm{ceil}_{G}(C) & \text{A2A3：} C \le 8160 \text{（元素数）} \quad \text{或} \quad \text{A5：} C \cdot b \le 8160 \text{（字节）} \\
G = 32 & \text{A2A3：} C > 8160 \text{（元素数）} \quad \text{或} \quad \text{A5：} C \cdot b > 8160 \text{（字节）}
\end{cases}
$$

- `ceil_G(C)` = $C$ 向上取整到 32 的倍数。
- **A2A3**：阈值单位为**元素数**（`srcShapeBytesPerRow / sizeof(T) <= MAX_UB_TMP`），即 $C \le 8160$，与 dtype 无关（float → $C \le 8160$，half → $C \le 8160$）。
- **A5**：阈值单位为**字节**（`srcShapeBytesPerRow <= MAX_UB_TMP`），即 $C \cdot b \le 8160$（float → $C \le 2040$，half → $C \le 4080$）。该阈值为 `pto_copy_ubuf_to_ubuf`（MOV_UB_TO_UB）的 repeat 上限 = 255 块 × 32 B。
- 尾块 = $t = C \bmod G$ 个元素（末尾不完整块），扩展至 $G$ 并以 $-\infty$ 填充。
- Path A（$C \cdot b \le 8160$，小行）：从行首**整行**复制到 tmp，然后原地填充最后 32 个元素。
- Path B（$C \cdot b > 8160$，大行）：仅复制**尾块**到 tmp；完整块直接从 `src` 排序。
- VBS32 硬件上限：每次调用 `repeat ≤ REPEAT_MAX = 255` 块（≤ 8160 元素）；超过 255 块的行拆分为多次 `vbitsort` 调用。
- **UB 布局：** `tmp` 应放置在 `dst` 之后（32 B 对齐），大小为 `ceil(ALIGN_C·b, 32)` 字节——不应使用固定的 8 KB 偏移，因为 Path A（A2A3）在接近阈值时对 float 需要最多 ~32 KB（$C \le 8160$ 元素 = float 32 KB）。

### 4 参数尾部处理

当 `validCol % 32 != 0` 时，末尾不完整块（$t = C \bmod 32$ 个元素）须填充为完整的 32 元素块后才能送入 `vbitsort`。两条路径：

- **A2A3：$C \le 8160$（元素数）** / **A5：$C \cdot b \le 8160$（字节）**（小行）：**整行**复制到 `tmp`，然后通过 `vdup` 原地覆盖最后 32 个元素为 $-\infty$ 填充；从 `tmp` 排序整行。
- **A2A3：$C > 8160$（元素数）** / **A5：$C \cdot b > 8160$（字节）**（大行）：仅复制**尾块**到 `tmp` 并填充；完整块直接从 `src` 排序，仅尾块从 `tmp` 排序。

填充值（$-\infty$ = `-1.0/0.0` 或 `std::numeric_limits<T>::lowest()`）落在降序排序的底部。若 `validCol > 32 × 255`，行按 `REPEAT_MAX` 大小的组拆分，每组通过独立的 `vbitsort` 调用排序。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

// 32 对齐：每行单个块
using SrcT = Tile<TileType::Vec, float, 1, 32>;
using IdxT = Tile<TileType::Vec, uint32_t, 1, 32>;
using DstT = Tile<TileType::Vec, float, 1, 64>;   // 2× src 列数（float）
SrcT src; IdxT idx; DstT dst;
TSort32(dst, src, idx);

// 非 32 对齐尾部：4 参数 + tmp
using SrcT2 = Tile<TileType::Vec, half, 1, 100>;
using IdxT2 = Tile<TileType::Vec, uint32_t, 1, 100>;
using DstT2 = Tile<TileType::Vec, half, 1, 400>;  // 4× src 列数（half）
using TmpT  = Tile<TileType::Vec, half, 1, 128>;  // ≥ ceil32(100)=128
TSort32(dst2, src2, idx2, tmp);
```
