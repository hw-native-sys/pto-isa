# include/pto/

This is the primary public header entry for PTO Tile Library. It contains:

- Tile type system and shared utilities
- PTO instruction API declarations (Auto/Manual forms)
- CPU simulation / stub support
- NPU instruction implementations (split by SoC generation)

## Recommended Include

| Scenario | Recommended Header |
|----------|-------------------|
| Upper-layer code (general) | `include/pto/pto-inst.hpp` — unified entry |
| CPU simulation stub injection | `__CPU_SIM` defined → auto-imports `pto/common/cpu_stub.hpp` |

## Layout

```
include/pto/
├── pto-inst.hpp              # Unified entry header (recommended)
├── pto.hpp                    # Core header (includes pto-inst.hpp)
│
├── common/                    # Platform-independent infrastructure
│   ├── pto_tile.hpp          # Core Tile types and layout
│   ├── pto_instr.hpp         # PTO instruction declarations
│   ├── pto_instr_impl.hpp    # PTO instruction shared implementations
│   ├── memory.hpp             # Memory operations
│   ├── constants.hpp          # Constant definitions
│   ├── utils.hpp              # General utilities
│   ├── type.hpp               # Type definitions
│   └── cpu_stub.hpp           # CPU simulation stub
│
├── cpu/                       # CPU-side simulation (if enabled)
│
└── npu/                      # NPU-side implementations (split by SoC)
    ├── a2a3/                 # Ascend A2/A3 series
    │   ├── TAdd.hpp          # TADD implementation
    │   ├── TMatmul.hpp       # TMATMUL implementation
    │   ├── TLoad.hpp         # TLOAD implementation
    │   └── ...               # Other instruction implementations
    └── a5/                   # Ascend A5 series
        ├── TAdd.hpp
        ├── TMatmul.hpp
        ├── TLoad.hpp
        └── ...
```

## Common TileType ↔ Hardware Buffer Mapping

| TileType | Hardware Buffer | Capacity | Typical Use |
|----------|----------------|----------|-------------|
| `Vec` | Unified Buffer (UB) | 256 KB | General elementwise operations |
| `Mat` | L1 | 512 KB | Matrix multiply operands |
| `Left` | L0A | 64 KB | Matmul A operand |
| `Right` | L0B | 64 KB | Matmul B operand |
| `Acc` | L0C | 256 KB | Matmul accumulator |

## Typical Usage

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

// Define GlobalTensor
using DynShape = Shape<1, 1, 1, kGRows_, kGCols_>;
using DynStride = Stride<1, 1, 1, kGCols_, 1>;
using GlobalData = GlobalTensor<T, DynShape, DynStride>;
GlobalData srcGlobal(src);

// Define Tile
using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
TileData srcTile(kTRows_, kTCols_);

// PTO instruction calls
TLOAD(srcTile, srcGlobal);
TADD(dstTile, src0Tile, src1Tile);
TSTORE(dstGlobal, dstTile);
```

## Related Docs

- [ISA Instruction Reference](../docs/isa/) — Full PTO ISA instruction semantics
- [Tile Programming Model](../../docs/coding/Tile.md) — Tile type system in depth
- [include/README_zh.md](./README_zh.md) — 中文版入口
