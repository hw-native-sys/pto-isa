# Location Intent And Legality

PTO legality depends on more than element type and shape. Many operations also care about where a value is intended to live or what role it plays in the selected instruction set. The location-intent taxonomy and the legality checking pipeline are defined below.

## Location Intent Taxonomy

Every tile operand in PTO carries a **location intent** — a declared role that determines which execution pipeline processes it and what operations are legal on it. The location intent is encoded in the `loc=` field of the tile type.

### Location Intent Values

| Location Intent | Pipeline | Description | Typical Use |
|----------------|----------|-------------|-------------|
| `loc=vec` | Vector Pipeline (V) | General-purpose vector tile | Elementwise ops, `TADD`, `TMUL`, `TCVT`, `TLOAD/TSTORE` |
| `loc=mat` | Matrix Multiply (M/CUBE) | Matrix multiply operand (A or B) | `TGEMV`, `TGEMV_ACC`, `TGEMV_BIAS` |
| `loc=acc` | Matrix Multiply (M/CUBE) | Accumulator / output tile | `TMATMUL`, `TMATMUL_ACC`, `TMATMUL_BIAS` output |
| `loc=left` | Matrix Multiply (M/CUBE) | Left-hand operand of MX-format matmul | `TMATMUL_MX` LHS (NZ layout, `SLayout::RowMajor`) |
| `loc=right` | Matrix Multiply (M/CUBE) | Right-hand operand of MX-format matmul | `TMATMUL_MX` RHS (`SLayout::NoneBox`, `NN` fractal) |
| `loc=scalar` | Scalar Unit | Scalar tile (1×1) | Scalar operations on tile instructions |

### Location Intent in Tile Type

In SSA/IR form, location intent is part of the tile type:

```
!pto.tile<loc=vec, f32, 16, 16, RowMajor, NoneBox, None, Zero>
!pto.tile_buf<loc=left, int8, 16, 16, RowMajor, RowMajor, NZ, Null>
!pto.tile_buf<loc=acc, int32, 16, 16, RowMajor, NoneBox, None, Zero>
```

In C++ API, location intent is expressed via the `TileType` template parameter:

```cpp
using VecTile = Tile<TileType::Vec, float, 16, 16>;
using AccTile = Tile<TileType::Acc, float, 16, 16>;
using LeftTile = Tile<TileType::Left, int8_t, 16, 16, RowMajor, RowMajor, NZ, Null>;
```

## Legality Checking Pipeline

PTO performs legality checking in four sequential stages. A program is legal only if it passes all four stages:

```
┌─────────────────────────────────────────┐
│  Stage 1: TYPE CHECK                    │
│  Element types match? Sizes compatible?  │
│  → If fail: type error (diagnostic)     │
└─────────────────┬───────────────────────┘
                  │ PASS
                  ▼
┌─────────────────────────────────────────┐
│  Stage 2: SHAPE CHECK                   │
│  Physical shape (Rows, Cols) legal?     │
│  Valid region (Rv, Cv) within bounds?   │
│  → If fail: shape error (diagnostic)    │
└─────────────────┬───────────────────────┘
                  │ PASS
                  ▼
┌─────────────────────────────────────────┐
│  Stage 3: LAYOUT CHECK                  │
│  BLayout+SLayout+Fractal combo legal    │
│  for this TileType and instruction?     │
│  → If fail: layout error (diagnostic)   │
└─────────────────┬───────────────────────┘
                  │ PASS
                  ▼
┌─────────────────────────────────────────┐
│  Stage 4: TARGET PROFILE CHECK          │
│  TileType + dtype supported on target?   │
│  MX format, FP8, fractal legal on A5?   │
│  → If fail: profile error (diagnostic)  │
└─────────────────┴───────────────────────┘
                  │ PASS
                  ▼
              LEGAL PROGRAM
```

### Stage 1: Type Check

**Rule**: The element type of all operands MUST be compatible with the operation.

For binary tile operations (`TADD`, `TMUL`, etc.):
```
dtype(src0) == dtype(src1) == dtype(dst)
```

For type-converting operations (`TCVT`):
```
dtype(src) and dtype(dst) must be in the same conversion group (see Type System page)
sizeof(dtype(src)) may != sizeof(dtype(dst)) for converting ops
```

**Diagnostic**: `type mismatch: expected f32 but found f16 in operand 1`

### Stage 2: Shape Check

**Rule**: The physical shape of all operands MUST be within the legal bounds for the instruction and target profile.

```
1 <= Rows <= MAX_ROWS(profile)    -- e.g., 65535 on A5, 8192 on A2/A3
1 <= Cols <= MAX_COLS(profile)    -- e.g., 4095 on all profiles
0 <= Rv <= Rows                   -- valid region within physical bounds
0 <= Cv <= Cols
```

**Diagnostic**: `shape out of range: Cols=8192 exceeds maximum of 4095 for TDIV on A2/A3`

### Stage 3: Layout Check

**Rule**: The combination of `BLayout`, `SLayout`, and `Fractal` MUST be a supported combination for the operand's `TileType` and the instruction.

See the Layout Combinations table in the [Tiles And Valid Regions](../programming-model/tiles-and-valid-regions.md) page for the complete list of supported combinations.

**Examples**:
```
Vec tile with NZ layout:          ILLEGAL (Vec tiles do not support fractal layouts)
Left tile with ColMajor layout:   ILLEGAL (Left tiles must be RowMajor)
Mat tile with ColMajor NZ fractal: ILLEGAL (Mat tiles must use standard layouts)
```

**Diagnostic**: `layout mismatch: Vec tile with fractal layout not supported by TADD`

### Stage 4: Target Profile Check

**Rule**: The operand's `TileType`, element type, and layout MUST be supported on the selected target profile.

Examples:
```
FP8 e4m3 type on A2/A3:          ILLEGAL (FP8 not supported on A2/A3)
vstu (unaligned vector store):    ILLEGAL on CPU and A2/A3 (A5 only)
Left/Right MX format tiles:       ILLEGAL on CPU and A2/A3 (A5 only)
```

**Diagnostic**: `profile restriction: FP8 types require A5 profile`

## Legality by Instruction Set

Different instruction sets have different legality rules beyond the four-stage pipeline:

### Elementwise Tile-Tile (TADD, TMUL, etc.)

- All operands MUST be `loc=vec`.
- `BLayout`, `SLayout`, `Fractal` MUST be compatible with `Vec`.
- `dtype` MUST be in the elementwise instruction set type list (varies by profile).

### Matmul (TMATMUL, TGEMV, etc.)

- Left operand: `TileType::Left` (for MX format) or `TileType::Mat`
- Right operand: `TileType::Right` (for MX format) or `TileType::Mat`
- Accumulator: `TileType::Acc`
- Shape constraints: `Rows_A == Rows_C`, `Cols_A == Rows_B`, `Cols_B == Cols_C`

### Vector Compute (vadd, vmul, etc.)

- Operands MUST be `!pto.vreg<NxDTYPE>`.
- Mask operand MUST be `!pto.mask` with matching width.
- `dtype` MUST be in the vector instruction set type list (varies by profile).

## GM-Facing Operands (GlobalTensor)

GlobalTensor operands follow a separate legality path:

| Check | Rule |
|-------|------|
| Dtype size | `sizeof(tile.dtype) == sizeof(gtensor.dtype)` |
| Layout compatibility | `gtensor.Layout` (ND/DN/NZ) must be compatible with `tile.SLayout` |
| Shape positive | All shape dimensions > 0 |
| Valid region | `Rv > 0` and `Cv > 0` |

## Cases That Are Not Allowed

- Using vector-buffer assumptions on a tile-instruction set operand without an explicit bridge.
- Documenting location-sensitive instruction sets as though any local storage role were equivalent.
- Hiding target-profile narrowing inside generic "hardware-specific" wording.
- Relying on the CPU simulator's permissive legality checking as evidence of A5 legality.

## See Also

- [Type System](./type-system.md)
- [Tiles And Valid Regions](../programming-model/tiles-and-valid-regions.md)
- [Tile Instruction Set](../instruction-families/tile-families.md)
- [Vector Instruction Set](../instruction-families/vector-families.md)
- [Portability And Target Profiles](../reference/portability-and-target-profiles.md)
