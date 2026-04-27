# Predicate Generation And Algebra

Predicate generation and algebra operations create, combine, pack, unpack, and interleave `!pto.mask` values on the scalar and control instructions. The `!pto.mask` type is the lane-masking mechanism that `pto.v*` vector operations consume.

## The `!pto.mask` Type

`!pto.mask` is a predicate mask type whose width is tied to the active element type rather than being a fixed number of bits:

| Element Type | Vector Width N | Predicate Width |
|-------------|:-------------:|:--------------:|
| f32 | 64 | 64 bits |
| f16 / bf16 | 128 | 128 bits |
| i8 / u8 | 256 | 256 bits |

A predicate mask with bit value `1` at position `i` means lane `i` is **active**; bit value `0` means lane `i` is **inactive**. Vector operations execute on active lanes only; inactive lanes produce zero on A2/A3 and A5, and may produce zero or undefined values on the CPU simulator.

## Sub-category Overview

| Sub-category | Operations | Description | Static / Dynamic |
|--------------|-----------|-------------|-----------------|
| Pattern-based construction | `pset_b8`, `pset_b16`, `pset_b32` | Build mask from named pattern | Static (compile-time pattern) |
| Comparison generation (≥) | `pge_b8`, `pge_b16`, `pge_b32` | Generate mask: `i < scalar` | Dynamic (runtime scalar) |
| Comparison generation (<) | `plt_b8`, `plt_b16`, `plt_b32` | Generate mask: `i ≥ scalar`; also updates scalar | Dynamic (runtime scalar) |
| Predicate pack | `ppack` | Narrow: pack two N-bit masks into one 2N-bit mask | Static (partition token) |
| Predicate unpack | `punpack` | Widen: extract half from a 2N-bit mask | Static (partition token) |
| Boolean algebra | `pand`, `por`, `pxor`, `pnot` | AND / OR / XOR / NOT | Dynamic (runtime operands) |
| Predicate select | `psel` | `mask0 ? mask1 : mask2` | Dynamic (runtime operands) |
| Deinterleave | `pdintlv_b8` | Split one 2N-bit mask into two N-bit masks | Static |
| Interleave | `pintlv_b16` | Combine two N-bit masks into one 2N-bit mask | Static |

## Pattern Tokens

`pset_*` operations accept pattern tokens that encode compile-time-known mask shapes:

| Pattern | Predicate Width | Meaning |
|---------|:--------------:|---------|
| `PAT_ALL` | All N | All lanes active |
| `PAT_ALLF` | All N | All lanes inactive |
| `PAT_H` | N/2 | High half active (upper N/2 lanes) |
| `PAT_Q` | N/4 | Upper quarter active |
| `PAT_VL1` … `PAT_VL128` | N | First N lanes active |
| `PAT_M3` | N | Modular pattern: repeat every 3 lanes |
| `PAT_M4` | N | Modular pattern: repeat every 4 lanes |

## Partition Tokens

`ppack` and `punpack` use partition tokens to specify which half of the predicate register is accessed:

| Token | Meaning |
|-------|---------|
| `LOWER` | Lower N bits of the 2N-bit predicate register |
| `HIGHER` | Upper N bits of the 2N-bit predicate register |

## Shared Constraints

All predicate generation and algebra operations MUST satisfy:

1. **Operand type**: All predicate operands MUST be `!pto.mask`. Mixing predicate operands with scalar or vector register operands is **illegal**.
2. **Predicate width consistency**: All operands in a single operation MUST share the same predicate width. Operations that mix N-bit and 2N-bit predicates MUST use explicit pack/unpack.
3. **Pattern token validity**: Pattern tokens MUST be supported by the target profile. Using a pattern token outside its supported width context is **illegal**.
4. **Scalar operand type**: For `pge_*` and `plt_*` operations, the scalar operand type MUST match the variant suffix (`_b8` → i8, `_b16` → i16, `_b32` → i32).
5. **Side effect**: No predicate generation or algebra operation writes to UB or modifies architectural state beyond producing a predicate result.

## Relationship Between pset, pge, and plt

- `pset_*` → **static** mask, fully determined at compile time from the pattern token
- `pge_*` → **dynamic** mask, depends on a runtime scalar value; predicate lane `i` is active iff `i < scalar`
- `plt_*` → **dynamic** mask AND scalar update; predicate lane `i` is active iff `i < scalar`, and `scalar_out = scalar - N`

`plt_*` operations are designed for software-pipelined remainder loops where the scalar counter is decremented by the vector length each iteration.

## Per-Op Pages

### Pattern-based Construction
- [pto.pset_b8](./ops/predicate-generation-and-algebra/pset-b8.md)
- [pto.pset_b16](./ops/predicate-generation-and-algebra/pset-b16.md)
- [pto.pset_b32](./ops/predicate-generation-and-algebra/pset-b32.md)

### Comparison Generation (Greater-or-Equal)
- [pto.pge_b8](./ops/predicate-generation-and-algebra/pge-b8.md)
- [pto.pge_b16](./ops/predicate-generation-and-algebra/pge-b16.md)
- [pto.pge_b32](./ops/predicate-generation-and-algebra/pge-b32.md)

### Comparison Generation (Less-Than)
- [pto.plt_b8](./ops/predicate-generation-and-algebra/plt-b8.md)
- [pto.plt_b16](./ops/predicate-generation-and-algebra/plt-b16.md)
- [pto.plt_b32](./ops/predicate-generation-and-algebra/plt-b32.md)

### Pack / Unpack
- [pto.ppack](./ops/predicate-generation-and-algebra/ppack.md)
- [pto.punpack](./ops/predicate-generation-and-algebra/punpack.md)

### Boolean Algebra
- [pto.pand](./ops/predicate-generation-and-algebra/pand.md)
- [pto.por](./ops/predicate-generation-and-algebra/por.md)
- [pto.pxor](./ops/predicate-generation-and-algebra/pxor.md)
- [pto.pnot](./ops/predicate-generation-and-algebra/pnot.md)
- [pto.psel](./ops/predicate-generation-and-algebra/psel.md)

### Interleave / Deinterleave
- [pto.pdintlv_b8](./ops/predicate-generation-and-algebra/pdintlv-b8.md)
- [pto.pintlv_b16](./ops/predicate-generation-and-algebra/pintlv-b16.md)

## Related Material

- [Control and configuration](./control-and-configuration.md)
- [Vector Instruction Set: Predicate And Materialization](../vector/predicate-and-materialization.md)
- [Predicate Load Store](./predicate-load-store.md)
