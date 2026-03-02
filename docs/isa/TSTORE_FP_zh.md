# TSTORE_FP

## 指令示意图

![TSTORE_FP tile operation](../figures/isa/TSTORE_FP.svg)

## 简介

使用缩放 (`fp`) Tile 作为向量量化参数，将累加器 Tile 存储到全局内存。

## 数学语义

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. Conceptually (2D view, with a base offset), for `0 <= i < R` and `0 <= j < C`:

$$ \mathrm{dst}_{r_0 + i,\; c_0 + j} = \mathrm{Convert}\!\left(\mathrm{src}_{i,j};\ \mathrm{fp}\right) $$

## 汇编语法

PTO-AS 形式：参见 [docs/grammar/PTO-AS_zh.md](../grammar/PTO-AS_zh.md)。

同步形式：

```text
tstore.fp %src, %fp, %sv_out[%c0, %c0]
```

### IR Level 1（SSA）

```text
pto.tstore.fp %src, %fp, %mem : (!pto.tile<...>, !pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

### IR Level 2（DPS）

```text
pto.tstore.fp ins(%src, %fp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp` and `include/pto/common/constants.hpp`:

```cpp
template <typename TileData, typename GlobalData, typename FpTileData,
          AtomicType atomicType = AtomicType::AtomicNone, typename... WaitEvents>
PTO_INST RecordEvent TSTORE_FP(GlobalData& dst, TileData& src, FpTileData& fp, WaitEvents&... events);
```

## 约束

- **实现检查 (A2A3)**:
  - The fp store path is implemented via `TSTORE_IMPL(dst, src, fp)` and uses the same accumulator-to-GM legality checks as quantized accumulator stores:
    - Destination layout must be ND or NZ.
    - Source dtype must be `int32_t` or `float`.
    - Static shape constraints: `1 <= TileData::Cols <= 4095`; if ND then `1 <= TileData::Rows <= 8192`; if NZ then `1 <= TileData::Rows <= 65535` and `TileData::Cols % 16 == 0`.
    - Runtime: `1 <= src.GetValidCol() <= 4095`.
  - No explicit `static_assert` is enforced on `FpTileData` (the implementation uses `fp` to set FPC state).
- **实现检查 (A5)**:
  - Implemented via `TSTORE_IMPL(dst, src, fp)` and validated by `CheckStaticAcc<..., true>()` for the accumulator path (ND/NZ only, `int32_t/float` source dtype, rows/cols ranges).
  - No explicit `static_assert` is enforced on `FpTileData` (the implementation uses `fp` to set FPC state).

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto(__gm__ int8_t* out) {
  using AccT = TileAcc<float, 16, 16>;
  using FpT = Tile<TileType::Scaling, uint64_t, 1, 16, BLayout::RowMajor, 1, DYNAMIC, SLayout::NoneBox>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<int8_t, 16, 16, Layout::ND>;
  using GT = GlobalTensor<int8_t, GShape, GStride, Layout::ND>;

  GT gout(out);
  AccT acc;
  FpT fp(16);
  TSTORE_FP(gout, acc, fp);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual(__gm__ int8_t* out) {
  using AccT = TileAcc<float, 16, 16>;
  using FpT = Tile<TileType::Scaling, uint64_t, 1, 16, BLayout::RowMajor, 1, DYNAMIC, SLayout::NoneBox>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<int8_t, 16, 16, Layout::ND>;
  using GT = GlobalTensor<int8_t, GShape, GStride, Layout::ND>;

  GT gout(out);
  AccT acc;
  FpT fp(16);
  TASSIGN(acc, 0x1000);
  TASSIGN(fp,  0x2000);
  TSTORE_FP(gout, acc, fp);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
pto.tstore.fp %src, %fp, %mem : (!pto.tile<...>, !pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
pto.tstore.fp %src, %fp, %mem : (!pto.tile<...>, !pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

### PTO 汇编形式

```text
tstore.fp %src, %fp, %sv_out[%c0, %c0]
# IR Level 2 (DPS)
pto.tstore.fp ins(%src, %fp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```

