<!-- Generated from `docs/isa/tile/matrix-and-matrix-vector.md` -->

# Matrix And Matrix-Vector Family

Matrix and matrix-vector operations perform tiled linear algebra on `TileType::Mat` tiles (cube tiles). They use specialized matrix multiply units and may have dedicated DMA paths.

## Operations

| Operation | Description | Variants | TileType | C++ Intrinsic |
|-----------|-------------|----------|----------|---------------|
| [pto.tgemv](./ops/matrix-and-matrix-vector/tgemv.md) | General matrix-vector product | basic, acc, bias, mx | `Mat` | `TGEMV(C, A, x)` |
| [pto.tgemv_acc](./ops/matrix-and-matrix-vector/tgemv-acc.md) | GEMV with accumulation | — | `Mat` | `TGEMV_ACC(C, A, x)` |
| [pto.tgemv_bias](./ops/matrix-and-matrix-vector/tgemv-bias.md) | GEMV with bias addition | — | `Mat` | `TGEMV_BIAS(C, A, x, bias)` |
| [pto.tgemv_mx](./ops/matrix-and-matrix-vector/tgemv-mx.md) | MX-format GEMV | — | `Mat` | `TGEMV_MX(C, A, x, scale)` |
| [pto.tmatmul](./ops/matrix-and-matrix-vector/tmatmul.md) | General matrix-matrix multiply | basic, acc, bias, mx | `Mat` | `TMATMUL(C, A, B)` |
| [pto.tmatmul_acc](./ops/matrix-and-matrix-vector/tmatmul-acc.md) | Matmul with accumulation | — | `Mat` | `TMATMUL_ACC(C, A, B)` |
| [pto.tmatmul_bias](./ops/matrix-and-matrix-vector/tmatmul-bias.md) | Matmul with bias addition | — | `Mat` | `TMATMUL_BIAS(C, A, B, bias)` |
| [pto.tmatmul_mx](./ops/matrix-and-matrix-vector/tmatmul-mx.md) | MX-format matrix multiply | — | `Mat` | `TMATMUL_MX(C, A, B, scale)` |

## Mechanism

### GEMV

$$ \mathbf{y} = A \times \mathbf{x} + \mathbf{b} $$

- Matrix tile `A`: shape `(M, K)`
- Vector tile `x`: shape `(K)`
- Bias tile `b` (optional): shape `(M)`
- Result tile `y`: shape `(M)`

### Matmul

$$ C = A \times B + D $$

- Tile `A`: shape `(M, K)`
- Tile `B`: shape `(K, N)`
- Bias tile `D` (optional): shape `(M, N)`
- Result tile `C`: shape `(M, N)`

### MX Format

MX (Matrix Multiply) format is Huawei's hardware-optimized data format. It separates scale tensors and may use compressed data representation. The `*_mx` variants require:

1. MX-formatted input tiles (`A`, `B`) with matching MX layout.
2. A scale tensor (`scale`) with compatible shape.
3. Accumulator tile (`C`) with shape matching the output shape.

## Tile Type

Matrix operations **require** `TileType::Mat` (cube tiles). `TileType::Vec` tiles MUST NOT be used with matrix operations. Cube tiles use a different physical layout optimized for matrix multiplication and have different valid-region semantics from vector tiles.

## Type Support by Target Profile

| Element Type | CPU Simulator | A2/A3 | A5 |
|------------|:-------------:|:------:|:--:|
| f32 (float) | Yes | Yes | Yes |
| f16 (half) | Yes | Yes | Yes |
| bf16 (bfloat16_t) | Yes | Yes | Yes |
| i8 / u8 | No | Yes | Yes |
| f8e4m3 / f8e5m2 | No | No | Yes |
| MX format | No | No | Yes |

## Constraints

- **Tile type MUST be `TileType::Mat`** — `TileType::Vec` tiles MUST NOT be used.
- Shape compatibility: `(M, K) × (K) → (M)` for GEMV; `(M, K) × (K, N) → (M, N)` for matmul.
- MX format operations require matching MX layout between `A` and `B` tiles.
- Bias variants require compatible bias tensor shape.
- Accumulation variants require matching accumulator tile shape.
- On A2/A3, int8/i8 matrix multiply requires `shape[0..1] % 16 == 0`.
- On A5, FP8 matmul is supported but requires scale tensors.

## Cases That Are Not Allowed

- **MUST NOT** use `TileType::Vec` tiles with matrix operations.
- **MUST NOT** use incompatible shape combinations (e.g., `(M, K) × (N, K) → …`).
- **MUST NOT** mix MX and non-MX tiles in the same operation.
- **MUST NOT** use FP8 matmul on CPU simulator or A2/A3.

## C++ Intrinsic

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

// Basic matrix-vector
template <typename TileC, typename TileA, typename TileX>
PTO_INST RecordEvent TGEMV(TileC& C, TileA& A, TileX& x);

// Matrix-vector with accumulation
template <typename TileC, typename TileA, typename TileX>
PTO_INST RecordEvent TGEMV_ACC(TileC& C, TileA& A, TileX& x);

// Matrix-vector with bias
template <typename TileC, typename TileA, typename TileX, typename TileBias>
PTO_INST RecordEvent TGEMV_BIAS(TileC& C, TileA& A, TileX& x, TileBias& bias);

// Basic matrix multiply
template <typename TileC, typename TileA, typename TileB>
PTO_INST RecordEvent TMATMUL(TileC& C, TileA& A, TileB& B);

// Matrix multiply with accumulation
template <typename TileC, typename TileA, typename TileB>
PTO_INST RecordEvent TMATMUL_ACC(TileC& C, TileA& A, TileB& B);

// Matrix multiply with bias
template <typename TileC, typename TileA, typename TileB, typename TileBias>
PTO_INST RecordEvent TMATMUL_BIAS(TileC& C, TileA& A, TileB& B, TileBias& bias);

// MX-format matrix multiply
template <typename TileC, typename TileA, typename TileB, typename TileScale>
PTO_INST RecordEvent TMATMUL_MX(TileC& C, TileA& A, TileB& B, TileScale& scale);
```

## See Also

- [Tile families](../instruction-families/tile-families.md) — Family overview
- [Tile instruction surface](../instruction-surfaces/tile-instructions.md) — Surface description
