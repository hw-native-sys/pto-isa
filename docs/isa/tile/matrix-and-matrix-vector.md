# Matrix And Matrix-Vector Instruction Set

This family covers the cube-pipeline instructions that evaluate matrix products on tile buffers. The basic forms produce a new accumulator tile, the `_acc` forms continue accumulation on an existing accumulator tile, the `_bias` forms inject a bias tile, and the `*_mx` forms add explicit scale tiles for block-scale MX formats.

These instructions are not generic vector-tile operations. Their legality depends on dedicated matrix roles such as `Left`, `Right`, `Acc`, `Bias`, `ScaleLeft`, and `ScaleRight`, plus the target profile's layout and datatype rules.

## Operations

| Operation | Purpose | C++ intrinsic | Notes |
| --- | --- | --- | --- |
| [pto.tmatmul](./ops/matrix-and-matrix-vector/tmatmul.md) | Matrix multiply producing a fresh accumulator tile | `TMATMUL(C, A, B)` | New result tile |
| [pto.tmatmul_acc](./ops/matrix-and-matrix-vector/tmatmul-acc.md) | Matrix multiply that continues accumulation | `TMATMUL_ACC(C, A, B)` | K-loop body form |
| [pto.tmatmul_bias](./ops/matrix-and-matrix-vector/tmatmul-bias.md) | Matrix multiply with column bias | `TMATMUL_BIAS(C, A, B, bias)` | Bias tile is one row |
| [pto.tmatmul_mx](./ops/matrix-and-matrix-vector/tmatmul-mx.md) | Matrix multiply in MX block-scale format | `TMATMUL_MX(C, A, AScale, B, BScale)` | A5 only |
| [pto.tgemv](./ops/matrix-and-matrix-vector/tgemv.md) | Matrix-vector multiply producing a fresh accumulator tile | `TGEMV(C, A, B)` | `m = 1` GEMV shape |
| [pto.tgemv_acc](./ops/matrix-and-matrix-vector/tgemv-acc.md) | GEMV that continues accumulation | `TGEMV_ACC(C, A, B)` | Accumulating form |
| [pto.tgemv_bias](./ops/matrix-and-matrix-vector/tgemv-bias.md) | GEMV with bias add | `TGEMV_BIAS(C, A, B, bias)` | Bias tile is one row |
| [pto.tgemv_mx](./ops/matrix-and-matrix-vector/tgemv-mx.md) | GEMV in MX block-scale format | `TGEMV_MX(C, A, AScale, B, BScale)` | A5 only |

## Why This Family Exists

PTO keeps matrix-product instructions separate from ordinary tile arithmetic because the cube path has different operand roles, different legality checks, and different target constraints from the vector path. A reader needs one place that answers:

- which tile roles are legal,
- how accumulation differs from fresh output generation,
- how bias is injected,
- and which profile-specific layout rules apply on A2A3 versus A5.

## Mechanism

### TMATMUL

For `M = a.GetValidRow()`, `K = a.GetValidCol()`, and `N = b.GetValidCol()`:

$$ \mathrm{C}_{i,j} = \sum_{k=0}^{K-1} \mathrm{A}_{i,k} \cdot \mathrm{B}_{k,j} $$

`pto.tmatmul` treats the destination accumulator as a newly produced output tile.

### TMATMUL_ACC

$$ \mathrm{C1}_{i,j} = \mathrm{C0}_{i,j} + \sum_{k=0}^{K-1} \mathrm{A}_{i,k} \cdot \mathrm{B}_{k,j} $$

This form exists for split-K and blocked GEMM loops where a partial accumulator must be carried across iterations.

### TMATMUL_BIAS

$$ \mathrm{C}_{i,j} = \sum_{k=0}^{K-1} \mathrm{A}_{i,k} \cdot \mathrm{B}_{k,j} + \mathrm{Bias}_{0,j} $$

Bias is a one-row tile and is broadcast by output column.

### TGEMV

GEMV is the `m = 1` specialization of the same cube contract. PTO still exposes it as a separate instruction family because it has its own operand spelling and its own common usage pattern.

### MX Variants

`*_mx` uses block-scale MX formats such as MXFP4 and MXMP8. Those forms require:

- one left operand tile in `Left`,
- one right operand tile in `Right`,
- one left scale tile in `ScaleLeft`,
- one right scale tile in `ScaleRight`,
- and an accumulator/output tile in `Acc`.

MX is not "one extra scale tensor". It is a paired scale-tile contract on both sides of the product.

## Tile Roles And Buffer Mapping

The architectural tile roles are abstractions over target tile buffers:

- `Left` is the left matrix operand tile and corresponds to the L0A-backed operand path.
- `Right` is the right matrix operand tile and corresponds to the L0B-backed operand path.
- `Acc` is the accumulator/output tile.
- `Bias` is the one-row bias tile used by `*_bias`.
- `ScaleLeft` and `ScaleRight` are the scale tiles used by MX block-scale variants.

Programs should not assume one portable physical layout for `Right`. A2A3 and A5 both use the `Right` role, but the legal right-tile layout details differ by target profile.

## Target Profiles

`A2A3` in this manual means the Ascend 910B and Ascend 910C class targets. `A5` means the Ascend 950 PR and Ascend 950 DT class targets.

| Capability | CPU simulator | A2A3 | A5 |
| --- | :---: | :---: | :---: |
| `TMATMUL`, `TMATMUL_ACC`, `TMATMUL_BIAS` | Yes | Yes | Yes |
| `TGEMV`, `TGEMV_ACC`, `TGEMV_BIAS` | Yes | Yes | Yes |
| int8 cube path | No | Yes | Yes |
| fp16 / bf16 / fp32 cube path | Yes | Yes | Yes |
| fp8 cube path | No | No | Yes |
| MX block-scale path | No | No | Yes |

### Common Legality

- Shapes must satisfy `(M, K) x (K, N) -> (M, N)` for matmul.
- GEMV uses the same contract with `m = 1`.
- Left, Right, Acc, Bias, and MX scale-tile roles must match the operation being issued.
- Valid-region values outside the legal output domain are not repaired implicitly.

### A2A3 Notes

- The base cube path supports the repository's documented triples such as `(int32, int8, int8)` and `(float, half, half)`.
- Dynamic `m`, `k`, and `n` are constrained to `[1, 4095]`.
- The backend checks the `Left`/`Right`/`Acc` role combination explicitly.

### A5 Notes

- The base cube path accepts `int32` accumulators for int8 input pairs and `float` accumulators for fp16, bf16, fp32, and selected fp8 pairs.
- The `Right` role has A5-specific layout/fractal constraints; do not copy an A2A3 right-tile layout assumption onto A5.
- MX variants are A5-only and require both `ScaleLeft` and `ScaleRight`.

## Performance And Throughput

The repository currently exposes an A2A3 cost-model formula for the shared `mad/mmad` cube instruction used by `TMATMUL`, `TMATMUL_ACC`, `TMATMUL_BIAS`, `TGEMV`, `TGEMV_ACC`, and `TGEMV_BIAS`.

For A2A3:

- startup cost: `14` cycles,
- repeat count: `ceil(M/16) * ceil(N/16) * ceil(K / baskK)`,
- `baskK = 32 / sizeof(left_element_type)`,
- steady-state cost per repeat:
  - `1` cycle for int8 and fp16 buckets,
  - `2` cycles for fp32 buckets.

So the published A2A3 model is:

```text
cycles = 14 + repeat_count * repeat_cost
```

Examples backed by `tests/costmodel/st/testcase/tmatmul/tmatmul_kernel.cpp` include:

- half `40x50 * 50x60`: `62` cycles,
- int8 `6x7 * 7x8`: `15` cycles,
- float `120x110 * 110x50`: `910` cycles.

The current repository does not publish an equivalent A5 latency or throughput table for this family. A5 legality is specified, but cycle figures are not single-listed in the public cost-model headers.

## See Also

- [Tile instruction families](../instruction-families/tile-families.md)
- [Tile instruction surface](../instruction-families/tile-families.md)
- [Location intent and legality](../state-and-types/location-intent-and-legality.md)
- [Layout](../state-and-types/layout.md)
