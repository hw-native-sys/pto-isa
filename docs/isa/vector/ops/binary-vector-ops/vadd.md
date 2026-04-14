# pto.vadd

`pto.vadd` is part of the [Binary Vector Instructions](../../binary-vector-ops.md) instruction set.

## Summary

Lane-wise addition of two vector registers, producing a result vector register. Only lanes selected by the predicate mask are active; inactive lanes do not participate in the computation and their destination elements are left unchanged.

## Mechanism

`pto.vadd` is a `pto.v*` compute operation. It reads two source vector registers lane-by-lane, adds the corresponding elements, and writes the result to the destination vector register. The iteration domain covers all N lanes; the predicate mask determines which lanes are active.

For each lane `i` where the predicate is true:

$$ \mathrm{dst}_i = \mathrm{lhs}_i + \mathrm{rhs}_i $$

Lanes where the predicate is false are **inactive**: the destination register element at that lane is unmodified.

## Syntax

### PTO Assembly Form

```mlir
%result = pto.vadd %lhs, %rhs, %mask : (!pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask) -> !pto.vreg<NxT>
```

### AS Level 1 (SSA)

```mlir
%result = pto.vadd %lhs, %rhs, %mask : (!pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask) -> !pto.vreg<NxT>
```

### AS Level 2 (DPS)

```mlir
pto.vadd ins(%lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask)
          outs(%result : !pto.vreg<NxT>)
```

## C++ Intrinsic

```cpp
vector_f32 dst;
vector_f32 src0;
vector_f32 src1;
vector_bool mask;
vadd(dst, src0, src1, mask);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%lhs` | `!pto.vreg<NxT>` | Left-hand source vector register |
| `%rhs` | `!pto.vreg<NxT>` | Right-hand source vector register |
| `%mask` | `!pto.mask` | Predicate mask; lanes where mask bit is 1 are active |

Both source registers MUST have the same element type and the same vector width `N`. The mask width MUST match `N`.

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%dst` | `!pto.vreg<NxT>` | Lane-wise sum on active lanes; inactive lanes are unmodified |

## Side Effects

This operation has no architectural side effect beyond producing its destination vector register. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

- **Type match**: `%lhs`, `%rhs`, and `%dst` MUST have identical element types.
- **Width match**: All three registers MUST have the same vector width `N`.
- **Mask width**: `%mask` MUST have width equal to `N`.
- **Active lanes**: Only lanes where the mask bit is 1 (true) participate in the addition.
- **Inactive lanes**: Destination elements at inactive lanes are unmodified.

## Exceptions

- The verifier rejects illegal operand type mismatches, width mismatches, or mask width mismatches.
- Any additional illegality stated in the [Binary Vector Instructions](../../binary-vector-ops.md) instruction set page is also part of the contract.

## Target-Profile Restrictions

| Element Type | CPU Simulator | A2/A3 | A5 |
|------------|:-------------:|:------:|:--:|
| `f32` | Simulated | Simulated | Supported |
| `f16` / `bf16` | Simulated | Simulated | Supported |
| `i8`–`i64`, `u8`–`u64` | Simulated | Simulated | Supported |

A5 is the primary concrete profile for the vector instructions. CPU simulation and A2/A3-class targets emulate `pto.v*` operations using scalar loops while preserving the visible PTO contract. Code that depends on specific performance characteristics or latency should treat those dependencies as target-profile-specific.

## Performance

### A5 Latency

| Element Type | Latency (cycles) | A5 RV |
|---|---|---|
| `f32` | 7 | `RV_VADD` |
| `f16` | 7 | `RV_VADD` |
| `i32` | 7 | `RV_VADD` |
| `i16` | 7 | `RV_VADD` |
| `i8` | 7 | `RV_VADD` |

### A2/A3 Throughput

| Metric | Value | Applies To |
|--------|-------|-----------|
| Startup latency | 14 (`A2A3_STARTUP_BINARY`) | all FP/INT binary ops |
| Completion: FP32 | 19 (`A2A3_COMPL_FP_BINOP`) | f32, i32 |
| Completion: INT16 | 17 (`A2A3_COMPL_INT_BINOP`) | int16 |
| Per-repeat throughput | 2 (`A2A3_RPT_2`) | all binary ops |
| Pipeline interval | 18 (`A2A3_INTERVAL`) | all vector ops |
| Cycle model | `14 + C + 2R + (R-1)×18` | C=completion, R=repeats |

**Example**: 1024 f32 elements with 16 iterations (`R=16`):

```
A5 total (pipelined): 7 + 15×2 = 37 cycles
A2/A3 total: 14 + 19 + 32 + 270 = 335 cycles
```

### Execution Throughput

`vadd` has 2× the per-repeat throughput of unary ops, making it efficient for element-wise kernels. For batched processing, the vecscope loop hides latency through pipelining.

---

## Examples

### Full-vector addition (all lanes active)

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

// All lanes active: mask set to all-ones
Mask<64> mask;
mask.set_all(true);  // predicate all-true

VADD(vdst, va, vb, mask);
```

### Partial predication

```mlir
// Only lanes where %cond is true participate in addition
%result = pto.vadd %va, %vb, %cond : (!pto.vreg<128xf16>, !pto.vreg<128xf16>, !pto.mask) -> !pto.vreg<128xf16>
```

### Complete vector-load / compute / vector-store pipeline

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void vector_add(Ptr<ub_space_t, ub_t> ub_a, Ptr<ub_space_t, ub_t> ub_b,
                Ptr<ub_space_t, ub_t> ub_out, size_t count) {
    VReg<64, float> va, vb, vdst;
    Mask<64> mask;
    mask.set_all(true);

    VLDS(va, ub_a, "NORM");
    VLDS(vb, ub_b, "NORM");

    VADD(vdst, va, vb, mask);

    VSTS(vdst, ub_out);
}
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Binary Vector Instructions](../../binary-vector-ops.md)
- Next op in instruction set: [pto.vsub](./vsub.md)
- Vector instruction overview: [Vector Instructions](../../../instruction-surfaces/vector-instructions.md)
- Type system: [Type System](../../../state-and-types/type-system.md)
