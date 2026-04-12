<!-- Generated from `docs/isa/instruction-surfaces/vector-instructions.md` -->

# Vector Instruction Surface

`pto.v*` is the vector micro-instruction surface of PTO ISA. It exposes the vector pipeline directly for fine-grained control over lane-level operations, vector registers, predicates, and alignment state.

## Surface Overview

Vector instructions are the fine-grained compute layer beneath tile instructions. While tile instructions operate on architecturally visible tiles with valid-region semantics, vector instructions operate on vector registers (`!pto.vreg<NxT>`), scalar values, and predicate masks. The full register width is always meaningful — there is no valid-region abstraction at the vector level.

**Vector operands** (`!pto.vreg<NxT>`) represent fixed-length vector registers. The width `N` is determined by the element type:

| Element Type | Vector Width N | Register Size |
|-------------|:-------------:|:------------:|
| f32 | 64 | 256 B |
| f16, bf16 | 128 | 256 B |
| i16, u16 | 128 | 256 B |
| i8, u8 | 256 | 256 B |
| f8e4m3, f8e5m2 | 256 | 256 B |

### Data Flow

```
UnifiedBuffer (UB)
    │
    │  vlds / vsld / vgather2 (UB → vreg)
    ▼
Vector Registers (!pto.vreg<NxT>) ──► Vector Compute (pto.v*) ──► Vector Registers
    │                                                         │
    │  vsts / vsst / vscatter (vreg → UB)                   │
    └─────────────────────────────────────────────────────────┘
                    │
                    ▼
             UnifiedBuffer (UB) ──► copy_ubuf_to_gm ──► GlobalMemory
```

## Instruction Classes

| Class | Description | Examples |
|-------|-------------|----------|
| Vector Load/Store | UB↔vector register transfer with distribution modes | `vlds`, `vldas`, `vldus`, `vgather2`, `vsld`, `vsst`, `vscatter` |
| Predicate and Materialization | Vector broadcast and duplication | `vbr`, `vdup` |
| Unary Vector Ops | Single-operand lane-wise operations | `vabs`, `vneg`, `vexp`, `vsqrt`, `vrec`, `vrelu`, `vnot` |
| Binary Vector Ops | Two-operand lane-wise operations | `vadd`, `vsub`, `vmul`, `vdiv`, `vmax`, `vmin` |
| Vec-Scalar Ops | Vector combined with scalar operand | `vadds`, `vmuls`, `vshls`, `vlrelu` |
| Conversion Ops | Type conversion between numeric types | `vci`, `vcvt`, `vtrc` |
| Reduction Ops | Cross-lane reductions (channelled) | `vcadd`, `vcmax`, `vcmin`, `vcgadd`, `vcgmax` |
| Compare and Select | Comparison and conditional lane selection | `vcmp`, `vcmps`, `vsel`, `vselr`, `vselrv2` |
| Data Rearrangement | Lane permutation, interleaving, packing | `vintlv`, `vdintlv`, `vslide`, `vshift`, `vpack`, `vzunpack` |
| SFU and DSA Ops | Special function units and DSA-style operations | `vprelu`, `vexpdiff`, `vaxpy`, `vtranspose`, `vsort32` |

## Inputs

Vector instructions consume combinations of:

- Vector registers (`!pto.vreg<NxT>`)
- Scalar registers or immediate operands
- Predicate masks (`!pto.mask`) — selects which lanes participate
- Memory addresses (`!pto.ptr<T, ub>`) — for load/store ops
- Rounding-mode or distribution-mode attributes

## Expected Outputs

Vector instructions produce:

- Vector register payloads
- Scalar register values (e.g., reduction results)
- Memory writes (via vector store)
- Predicate masks (from compare operations)

## Side Effects

Most vector instructions are pure compute operations with no architectural side effects. Side-effecting vector instructions include:

| Class | Side Effects |
|-------|-------------|
| Vector Load/Store | Reads from or writes to UB-visible memory |
| Compare and Select | Produces predicate mask consumed by subsequent ops |

## Mask Behavior

Vector operations can be gated by a predicate mask. A predicate mask (`!pto.mask`) with width equal to the vector length `N` selects which lanes participate:

- Lanes where the mask bit is **1**: the operation executes normally.
- Lanes where the mask bit is **0**: the operation produces a **defined result** but the specific value depends on the operation:
  - Arithmetic ops: masked lanes produce the identity element (e.g., 0 for add, 1 for mul).
  - Load ops: masked lanes leave the destination register unchanged.
  - Store ops: masked lanes do not write to memory.

Programs MUST NOT rely on the identity-element behavior of masked lanes unless the operation explicitly documents it.

## Alignment State

Vector unaligned store operations (`vstu`, `vstus`, `vstur`) maintain an alignment state that evolves across successive stores. The state consists of an alignment offset that is updated after each store:

```
%align_out, %offset_out = pto.vstu %align_in, %offset_in, %value, %base
    : !pto.align, index, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align, index
```

A trailing flush form (`vstar` / `vstas`) is required to commit any buffered tail bytes. These ops are **A5-only**.

## Constraints

- **Vector length** (`N`) is determined by the element type (see table above); programs do not choose `N` directly.
- **Predicate width** must match the vector length `N` for predicate-gated operations.
- **Alignment requirements** vary by operation and target profile:
  - `vlds` / `vsld`: A5 requires 32B alignment for NORM mode; other profiles may be more permissive.
  - `vstu` / `vstus` / `vstur`: **A5-only**; not supported on CPU or A2/A3.
- **Type combinations** for conversion and arithmetic operations are defined per-op.
- No implicit type promotion: all operands must have compatible types.

## Cases That Are Not Allowed

- Using a predicate mask whose width does not match the target vector length.
- Accessing memory with an illegal alignment for the target profile.
- Relying on undefined lane behavior when predicates mask some lanes (must not depend on the identity-element value).
- Using `vstu` / `vstus` / `vstur` on CPU or A2/A3 (A5-only ops).
- Mixing vector types within a single operation unless the operation explicitly supports it.
- Issuing a vector store before the corresponding DMA copy is complete without an intervening `wait_flag`.

## Syntax

### Assembly Form (PTO-AS)

```asm
vadd %vdst, %vsrc0, %vsrc1 : !pto.vreg<f32, 64>
vlds %vreg, %ub_ptr[%offset] {dist = "NORM"} : !pto.ptr<f32, ub>
```

### SSA Form (AS Level 1)

```mlir
%vdst = pto.vadd %vsrc0, %vsrc1, %mask
    : (!pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask) -> !pto.vreg<64xf32>
```

### DPS Form (AS Level 2)

```mlir
pto.vadd ins(%vsrc0, %vsrc1, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask)
          outs(%vdst : !pto.vreg<64xf32>)
```

See [Assembly Spelling And Operands](../syntax-and-operands/assembly-model.md) for the full syntax specification.

## C++ Intrinsic

Vector instructions are available as C++ intrinsics declared in `include/pto/common/pto_instr.hpp`:

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

// Binary vector addition
PTO_INST void VADD(VecDst& dst, VecSrc0& src0, VecSrc1& src1);

// Vector load
PTO_INST void VLDS(VecData& dst, PtrType addr);

// Masked vector load
PTO_INST void VLDS(VecData& dst, PtrType addr, MaskType pred);
```

## See Also

- [Vector ISA reference](./vector/README.md) — Full vector family reference
- [Vector families](./instruction-families/vector-families.md) — Family-level contracts
- [Instruction families](./instruction-families/README.md) — All family groups
- [Format of instruction descriptions](../reference/format-of-instruction-descriptions.md) — Per-op page standard
