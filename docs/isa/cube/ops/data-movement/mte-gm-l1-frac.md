# pto.mte_gm_l1_frac

`pto.mte_gm_l1_frac` is part of [Cube Data Movement Ops](../../README.md#cube-data-movement-ops).

## Summary

Load a logical 2-D GM region and write one or more L1 **NZ fractal** matrix groups. `nd2nz` reads a logical `src[n, d]` matrix; `dn2nz` reads a logical `src[d, n]` matrix and writes the same logical `N x D` result into NZ layout.

This is the canonical entry point for staging row-major or column-major GM operands into the cube's [NZ Fractal Layout](../../nz-fractal-layout.md). After `pto.mte_gm_l1_frac`, the L1 tile can feed [`pto.mte_l1_l0a`](./mte-l1-l0a.md) / [`pto.mte_l1_l0b`](./mte-l1-l0b.md).

## Mechanism

Reference addressing:

```text
for g in 0 .. group_count-1:
  src_g = src + g * src_outer_stride
  dst_g = dst + g * dst_loop4_stride * 32

  for n in 0 .. n_value-1:
    for d in 0 .. d_value-1:
      if mode == nd2nz:
        value = load(src_g + n * src_inner_stride + d * sizeof(T))
      else:
        value = load(src_g + d * src_inner_stride + n * sizeof(T))
      store value into NZ position for logical [n, d] under dst_g

  invalid lanes in the final C0 group are written as zero
```

## Syntax

```mlir
pto.mte_gm_l1_frac %src, %dst, nd2nz|dn2nz,
  shape(%n_value, %d_value),
  src_layout(%src_inner_stride[, %src_outer_stride]),
  dst_group(%group_count, %dst_loop2_stride, %dst_loop3_stride, %dst_loop4_stride),
  ctrl(%l2_cache_ctrl, %smallc0_en)
  : !pto.ptr<T, gm>, !pto.ptr<T, l1>, ...
```

## Inputs

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | GM source base pointer |
| `%dst` | ptr | L1 NZ destination base pointer (`!pto.ptr<T, l1>`) |
| `nd2nz` / `dn2nz` | keyword | Source logical layout mode |
| `shape(%n_value, %d_value)` | i64 pair | Logical output shape before NZ packing |
| `src_layout(%src_inner_stride[, %src_outer_stride])` | i64 / optional i64 | Source row/matrix byte strides |
| `dst_group(...)` | i64 tuple | Destination group count and placement strides in C0-size units (1 unit = 32 bytes) |
| `ctrl(%l2_cache_ctrl, %smallc0_en)` | i64, i1 | Cache hint and small-C0 packing enable |

`src_layout(%src_inner_stride)` describes one logical source matrix. For `nd2nz`, `%src_inner_stride` is the byte distance from `src[n, 0]` to `src[n + 1, 0]`. For `dn2nz`, it is the byte distance from `src[d, 0]` to `src[d + 1, 0]`. When `%src_outer_stride` is present, it is the byte distance between adjacent source matrices; omitted means 0.

`dst_group(%group_count, %dst_loop2_stride, %dst_loop3_stride, %dst_loop4_stride)` writes `%group_count` logical matrices. Destination strides are measured in C0-size units. These strides place generated NZ blocks relative to `%dst`; they do not select a separate memory block.

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | Writes one or more NZ matrix groups into L1. |

## Side Effects

Reads GM-visible storage; writes L1-visible storage. Engages the AIC MTE2 pipe.

## Constraints

!!! warning "Constraints"
    - Source strides are bytes. For row-major 16×16 f16 input, `src_layout(32)` describes consecutive rows.
    - Destination strides are C0-size units, **not** bytes and **not** elements.
    - `smallc0_en = true` is valid only for target-supported small-C0 cases. The current contract rejects `d_value > 4` in small-C0 mode.
    - In normal C0 mode, each destination C0 burst is padded to 32 bytes. In small-C0 mode, each destination burst is padded to 4 logical channels; the generated inner-N and C0 destination placement is fixed by that small-C0 packing rule. `%dst_loop4_stride` still places adjacent matrix groups.
    - In small-C0 mode, missing logical `N` rows and invalid `D` lanes are written as zero, and the tail of a generated NZ matrix is padded to the 32-byte C0 boundary.
    - Destination regions selected by `%dst` and `dst_group(...)` must not overlap. If two generated writes target the same bytes, the final value is not a stable program result.

## Examples

```mlir
pto.mte_gm_l1_frac %src, %dst, nd2nz,
  shape(%c32_i64, %c16_i64),
  src_layout(%c32_i64, %c1024_i64),
  dst_group(%c2_i64, %c1_i64, %c16_i64, %c64_i64),
  ctrl(%c0_i64, %false)
  : !pto.ptr<f16, gm>, !pto.ptr<f16, l1>, nd2nz, shape i64, i64,
    src_layout(i64, i64), dst_group i64, i64, i64, i64, ctrl i64, i1
```

## Related Ops

- Direct GM→L1 copy (no repack): [pto.mte_gm_l1](./mte-gm-l1.md)
- Consume the NZ tile: [pto.mte_l1_l0a](./mte-l1-l0a.md), [pto.mte_l1_l0b](./mte-l1-l0b.md)
- Layout reference: [NZ Fractal Layout](../../nz-fractal-layout.md)
