<!-- Generated from `docs/isa/tile/ops/memory-and-data-movement/tstore_zh.md` -->

# TSTORE

## 指令示意图

![TSTORE tile operation](../figures/isa/TSTORE.svg)

## 简介

将 Tile 中的数据存储到 GlobalTensor (GM)，可选使用原子写入或量化参数。

## 数学语义

Notation depends on the `GlobalTensor` shape/stride and the `Tile` layout. Conceptually (2D view, with a base offset):

$$ \mathrm{dst}_{r_0 + i,\; c_0 + j} = \mathrm{src}_{i,j} $$

## 汇编语法

PTO-AS 形式：参见 [PTO-AS Specification](../assembly/PTO-AS.md).

同步形式：

```text
tstore %t1, %sv_out[%c0, %c0]
```

### AS Level 1（SSA）

```text
pto.tstore %src, %mem : (!pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

### AS Level 2（DPS）

```text
pto.tstore ins(%src : !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp` and `include/pto/common/constants.hpp`:

```cpp
template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
          typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, WaitEvents &... events);

template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
          typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, uint64_t preQuantScalar, WaitEvents &... events);

template <typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType = AtomicType::AtomicNone,
          typename... WaitEvents>
PTO_INST RecordEvent TSTORE_FP(GlobalData &dst, TileData &src, FpTileData &fp, WaitEvents &... events);
```

The `preQuantScalar` and `TSTORE_FP` quantized-store overloads are only legal for `TileType::Acc` on current A2/A3 and A5 backends. They do not provide a native vec-tile quantized store contract.

## 约束

- **实现检查 (A2A3)**:
  - Source tile location must be one of: `TileType::Vec`, `TileType::Mat`, `TileType::Acc`.
  - Runtime: all `dst.GetShape(dim)` values and `src.GetValidRow()/GetValidCol()` must be `> 0`.
  - For `TileType::Vec` / `TileType::Mat`:
    - `TileData::DType` must be one of: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `half`, `bfloat16_t`, `float`.
    - `sizeof(TileData::DType) == sizeof(GlobalData::DType)`.
    - Layouts must match ND/DN/NZ (or a special case where `TileData::Rows == 1` or `TileData::Cols == 1`).
    - For `int64_t/uint64_t`, only ND->ND or DN->DN are supported.
    - A2/A3 does not expose a native vec quantized-store path. Frontends that need `vec -> GM` dtype conversion or quantization MUST first materialize the converted vec tile (for example via `TCVT`) and then issue a same-dtype `TSTORE`.
  - For `TileType::Acc` (including quantized/atomic variants):
    - Destination layout must be ND or NZ.
    - Source dtype must be `int32_t` or `float`.
    - When not using quantization, destination dtype must be `__gm__ int32_t/float/half/bfloat16_t`.
    - Static shape constraints: `1 <= TileData::Cols <= 4095`; if ND then `1 <= TileData::Rows <= 8192`; if NZ then `1 <= TileData::Rows <= 65535` and `TileData::Cols % 16 == 0`.
    - Runtime: `1 <= src.GetValidCol() <= 4095`.
- **实现检查 (A5)**:
  - Source tile location must be `TileType::Vec` or `TileType::Acc` (no `Mat` store on this target).
  - For `TileType::Vec`:
    - `sizeof(TileData::DType) == sizeof(GlobalData::DType)`.
    - `TileData::DType` must be one of: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `half`, `bfloat16_t`, `float`, `float8_e4m3_t`, `float8_e5m2_t`, `hifloat8_t`, `float4_e1m2x2_t`, `float4_e2m1x2_t`.
    - Layouts must match ND/DN/NZ (or a special case where `TileData::Rows == 1` or `TileData::Cols == 1`).
    - Additional alignment constraints are enforced (e.g., for ND the row-major width in bytes must be a multiple of 32; for DN the column-major height in bytes must be a multiple of 32, with special-case exceptions).
  - For `TileType::Acc`:
    - Destination layout must be ND or NZ; source dtype must be `int32_t` or `float`.
    - When not using quantization, destination dtype must be `__gm__ int32_t/float/half/bfloat16_t`.
    - Static shape constraints match A2A3 for rows/cols; `AtomicAdd` additionally restricts destination dtype to supported atomic types.
- **有效区域**:
  - The implementation uses `src.GetValidRow()` / `src.GetValidCol()` as the transfer size.

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_auto(__gm__ T* out) {
  using TileT = Tile<TileType::Vec, T, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
  using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

  GTensor gout(out);
  TileT t;
  TSTORE(gout, t);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_manual(__gm__ T* out) {
  using TileT = Tile<TileType::Vec, T, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
  using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

  GTensor gout(out);
  TileT t;
  TASSIGN(t, 0x1000);
  TSTORE<TileT, GTensor, AtomicType::AtomicAdd>(gout, t);
}
```
