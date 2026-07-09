# TQUANT DN — Axis-0 分组量化与 DN→ZZ

## 指令示意图

![TQUANT tile operation](../figures/isa/TQUANT.svg)

## 简介

**ND**（"normal direction"）是默认的 MX 分组方式：沿 **axis 1**（列方向）每 32 个连续元素为一组。**DN** 表示转置式分组，沿 **axis 0**（行方向）每 32 个连续行为一组。两种模式都产生 RowMajor Tile；"DN"/"ND"仅指**分组轴**，而非存储布局。DN 用于 FlashAttention 场景，其中 softmax 输出（P 矩阵）的自然分组沿 M（行）维度。

DN 量化后，FP8 数据通过标准 `TMOV(ND→NZ)` 转换为 NZ 格式，E8M0 指数通过新的 `TMOV<grp_axis=0>(DN→ZZ)` 转换为 ZZ 格式。

## C++ 内建接口

主接口为 `<grp_axis, mx_alg>` 模板形式：

```cpp
template <int grp_axis, auto mx_alg, typename TileDataOut = void, typename TileDataSrc = void,
          typename TileDataExp = void, typename TileDataMax = void, typename TileDataScaling = void,
          typename... WaitEvents>
PTO_INST RecordEvent TQUANT(TileDataOut &dst, TileDataSrc &src, TileDataExp *exp, TileDataMax *max,
                            TileDataScaling *scaling, WaitEvents &... events);
```

### 参数

| 参数 | 说明 |
|-----------|-------------|
| `grp_axis` | **0** = DN（axis 0 / 行方向分组）；**1** = ND（axis 1 / 列方向分组，默认） |
| `mx_alg` | 组合的目标格式 + 缩放算法标签（`MxQuantAlg` 枚举） |
| `dst` | 输出的 FP8/FP4 Tile（RowMajor，形状与 `src` 相同） |
| `src` | 输入的 fp32/bf16/fp16 Tile（RowMajor `M×N`） |
| `exp` | 输出的 E8M0 指数 Tile：DN 时形状为 `M̂×N`，ND 时为 `M×Γ` |
| `max` | Scratch 的每组 abs-max Tile |
| `scaling` | Scratch 的每组缩放 Tile |

### `MxQuantAlg` 取值

```cpp
enum class MxQuantAlg {
    OcpMxFp8E4M3 = 0, // MXFP8 E4M3 + OCP scale
    NvMxFp8E4M3  = 1, // MXFP8 E4M3 + NV scale
    OcpMxFp4E2M1 = 2, // MXFP4 E2M1 + OCP scale
    NvMxFp4E2M1 = 3,  // MXFP4 E2M1 + NV scale
};
```

> **向后兼容性：** 旧版 `TQUANT<QuantType::MXFP8, ...>` 接口保持不变。`<grp_axis, mx_alg>` 形式为首选接口，无需移除任何旧接口。

## DN 输出形状

对于源 Tile `M×N`，设 `M̂ = M/32`，`Γ = N/32`：

| 输出 | ND（`grp_axis=1`） | DN（`grp_axis=0`） |
|--------|-------------------|-------------------|
| FP8/FP4 数据 | `M×N` RowMajor | `M×N` RowMajor（相同） |
| E8M0 指数 | `M×Γ` | `M̂×N` |
| Max / Scaling | `M×Γ` | `M̂×N` |

**数据 Tile 在 ND 和 DN 之间完全相同**（相同的 `(r,c)` 地址）；仅指数/max/scaling Tile 形状不同。因此数据的 `TMOV(ND→NZ)` 可以不变复用。只有指数需要新的转换：**DN→ZZ**。

## Cube 消费约定

从 A5 sim 日志验证（`LOAD_2Dv2` + `LOAD_MX_2Dv2` + `MMAD_MX`）：

```
FP8 data   → L0A/L0B  作为 NZ fractal（LOAD_2Dv2  Dtype:B8）
E8M0 scale → L0AMX/L0BMX 作为 ZZ fractal（LOAD_MX_2Dv2 Dtype:B16）
MMAD_MX 通过 fractal 字节位置将其配对。
```

Cube 始终要求数据为 NZ 格式、scale 为 ZZ 格式，与量化分组轴无关。DN 量化操作数的唯一区别是指数 Tile 形状（`M̂×N` 而非 `M×Γ`）以及应用的转换（`DN→ZZ` 而非 `ND→ZZ`）。

## DN→ZZ 转换

### 数学证明

对于形状为 `M̂×N` 的 DN 指数 Tile `E_DN[hat_r][c]`（RowMajor，平坦索引为 `hat_r·N + c`）：

**定理（DN→ZZ = 转置 ⊕ ND→ZZ）：**

$$E_{ZZ}[c_b, p, q, \delta] = E_{DN}^T[16c_b + q][2p + \delta] = E_{DN}[2p + \delta][16c_b + q]$$

其中 `c_b ∈ [0, N/16)`，`p ∈ [0, M̂/2)`，`q ∈ [0,16)`，`δ ∈ {0,1}`。

**推论（直接源索引）：**

$$\text{src\_idx}(c_b, p, q, \delta) = (2p + \delta) \cdot N + 16c_b + q$$

**推论（无需 gather）：** 对于固定的 `(c_b, p)`，ZZ 块的 32 字节来自两个连续的 16 字节运行：`E_DN[2p][16c_b:16c_b+16]` 和 `E_DN[2p+1][16c_b:16c_b+16]`。通过 `vintlv` 将它们交错即可得到 ZZ fractal 所需的 `qδ` 交错顺序。因此 DN→ZZ 比 ND→ZZ 更高效（连续加载，无需 `vgather2`/`BLK`/`E2B`）。

### 对齐约束

- `N mod 16 = 0`（始终满足，因为 `N mod 32 = 0`）。
- `M̂ mod 2 = 0`，即 **`M mod 64 = 0`**（用于 δ 配对）。比 ND→ZZ 的 `M mod 16 = 0` 更严格。
- `M = 32`（`M̂ = 1`）：退化恒等（无配对）。

### 与 vshls+vor 的关系

FA 融合 softmax 宏中的 `vshls+vor` 字节打包在 **仅 `M̂=4, N=64`** 时，其数学结果与此转换的转置步骤相同。它无法泛化（需要 `M̂≤4` 以适配 B32 字，且 `N≤64` 以适配单次 VL）。`TMovDnTo2Zz` 是通用替代方案。

### 实现说明

`TMovDnTo2Zz` 使用 `vlds + vintlv + vsstb`（块步长 scatter，`blockStride = numPairs`）配合 `CreatePredicate` 尾部自动递减 — 无需标量计算、无分支、无静态谓词。必须使用 `vsstb` 的 **5-arg POST_UPDATE** 形式；4-arg 形式将 `offset` 解释为源寄存器偏移而非步长配置。参见 `include/pto/npu/a5/TMov.hpp`（`GenerateB8IndicesDN2ZZToUB`）。

## TMOV 接口

### DN→ZZ（新增）

```cpp
template <int grp_axis, typename DstTileData, typename SrcTileData, typename TmpTileData, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, TmpTileData &tmp, WaitEvents &... events);
```

`TMOV<0>(zzTile, e8DnTile, tmpTile)` 选择 `TMovDnTo2Zz`。标准的 `TMOV(zzTile, e8Tile, tmpTile)`（无 `<grp_axis>`）仍为 ND→ZZ（`grp_axis` 默认为 1）。

### ND→NZ（数据，不变）

```cpp
TMOV(fp8NZTile, fp8Tile);   // 标准 2-arg ND→NZ；对 DN 数据同样正确（RowMajor，相同地址）
```

## 流水线（完整 DN 流程）

```
src[M×N] (fp32)
  ──TQUANT<0, MxQuantAlg::OcpMxFp8E4M3>──▶ fp8[M×N] + e8[M̂×N] (DN 指数)
  ──TMOV(ND→NZ)──────────────────────────▶ fp8NZ
  ──TMOV<0>(DN→ZZ)───────────────────────▶ e8ZZ
  ──feed to cube MMAD_MX─────────────────▶ C[M×N]
```

## 示例

```cpp
// DN 量化（axis 0 分组）
TQUANT<0, MxQuantAlg::OcpMxFp8E4M3>(fp8Tile, srcTile, &e8DnTile, &maxTile, &scalingTile);
// 数据 ND→NZ（标准）
TMOV(fp8NZTile, fp8Tile);
// 指数 DN→ZZ（新增）
TMOV<0>(e8ZzTile, e8DnTile, tmpTile);
```

完整 ST 示例（阶段 1–3）见 `tests/npu/a5/src/st/testcase/tquant_dn/`。
