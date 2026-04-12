<!-- Generated from `docs/isa/scalar/ops/dma-copy/copy-ubuf-to-ubuf.md` -->

# pto.copy_ubuf_to_ubuf

Standalone reference page for `pto.copy_ubuf_to_ubuf`. This page belongs to the [DMA Copy](../../dma-copy.md) family in the PTO ISA manual.

## Summary

Copy within Unified Buffer.

## Mechanism

`pto.copy_ubuf_to_ubuf` is a `pto.*` control/configuration operation. It changes ordering, buffer, event, or DMA-visible state that later payload work depends on. The portable guarantee is the dependency/configuration effect, while concrete pipe/event spaces remain target-profile details.

## Syntax


## Inputs

The inputs are the architecture-visible control operands shown in the syntax: pipe ids, event ids, buffer ids, loop/stride values, pointers, or configuration words used to drive later execution.

## Expected Outputs

This form is primarily defined by the side effect it has on control state, predicate state, or memory. It does not publish a new payload SSA result beyond any explicit state outputs shown in the syntax.

## Side Effects

This operation updates control, synchronization, or DMA configuration state. Depending on the form, it may stall a stage, establish a producer-consumer edge, reserve or release a buffer token, or configure later copy behavior.

## Constraints

This operation inherits the legality and operand-shape rules of its family overview. Any target-specific narrowing of element types, distributions, pipe/event spaces, or configuration tuples must be stated by the selected target profile.

## Exceptions

- It is illegal to use unsupported pipe ids, event ids, buffer ids, or configuration tuples for the selected target profile.
- Waiting on state that was never established by a matching producer or prior configuration is an illegal PTO program.

## Target-Profile Restrictions

- CPU simulation preserves the visible dependency/configuration contract, but it may not expose every low-level hazard that motivates the form on hardware targets.
- A2/A3 and A5 profiles may use different concrete pipe, DMA, predicate, or event spaces. Portable code must rely on the documented PTO contract plus the selected target profile.

## Examples

```mlir
pto.copy_ubuf_to_ubuf %source, %dest, %sid, %n_burst, %len_burst, %src_stride, %dst_stride
    : !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64 x5
```

```
burst    = lenBurst contiguous bytes transferred per row
stride   = distance (bytes) from start of row[r] to start of row[r+1]
pad      = ub_stride - lenBurst, padded to the 32B alignment boundary
```

```
GM (source, `!pto.ptr<T, gm>`):

          |<--- src_stride (start-to-start) --->|
          |<- len_burst ->|                     |
Row 0:    [##DATA########]......................|
Row 1:    [##DATA########]......................|
Row 2:    [##DATA########]......................|
          ...
Row N-1:  [##DATA########]

UB (destination, `!pto.ptr<T, ub>`, 32B-aligned):

          |<---------- dst_stride (32B-aligned) ---------->|
          |<- len_burst ->|<- pad (to 32B boundary) ->|    |
Row 0:    [##DATA########][000000 PAD 000000000000000]
Row 1:    [##DATA########][000000 PAD 000000000000000]
Row 2:    [##DATA########][000000 PAD 000000000000000]
          ...
Row N-1:  [##DATA########][000000 PAD 000000000000000]

N = n_burst
stride = start of row[r] to start of row[r+1]
pad    = filled with pad_val to 32B boundary (data_select_bit=true)
[DATA] = valid data transferred by DMA
[PAD]  = pad_val fill (set via set_mov_pad_val)
```

```
UB (source, `!pto.ptr<T, ub>`, 32B-aligned start addr):

          |<---------- src_stride (32B-aligned) --------->|
          |<- len_burst ->|<-- pad (ignored on read) -->| |
Row 0:    [##DATA########][000 pad 000000000000000000]
Row 1:    [##DATA########][000 pad 000000000000000000]
Row 2:    [##DATA########][000 pad 000000000000000000]
          ...
Row N-1:  [##DATA########][000 pad 000000000000000000]

GM (destination, `!pto.ptr<T, gm>`):

          |<--- dst_stride (start-to-start) --->|
          |<- len_burst ->|                     |
Row 0:    [##DATA########]......................|
Row 1:    [##DATA########]......................|
Row 2:    [##DATA########]......................|
          ...
Row N-1:  [##DATA########]

N = n_burst
MTE3 reads only len_burst bytes from each UB row (de-padding).
Only len_burst bytes are written to each GM row.
```

```c
// C equivalent of what the HW executes:
for (int j = 0; j < loop2_count; j++) {                // HW outer loop
    uint8_t *gm1 = gm_src + j * loop2_src_stride;
    uint8_t *ub1 = ub_dst + j * loop2_dst_stride;

    for (int k = 0; k < loop1_count; k++) {            // HW inner loop
        uint8_t *gm2 = gm1 + k * loop1_src_stride;
        uint8_t *ub2 = ub1 + k * loop1_dst_stride;

        for (int r = 0; r < n_burst; r++) {            // burst engine
            memcpy(ub2 + r * dst_stride,               //   UB dest row
                   gm2 + r * src_stride,               //   GM src row
                   len_burst);                          //   contiguous bytes
            if (data_select_bit)
                memset(ub2 + r * dst_stride + len_burst,
                       pad_val, dst_stride - len_burst);
        }
    }
}
```

```c
// C equivalent:
for (int j = 0; j < loop2_count; j++) {
    uint8_t *ub1 = ub_src + j * loop2_src_stride;
    uint8_t *gm1 = gm_dst + j * loop2_dst_stride;

    for (int k = 0; k < loop1_count; k++) {
        uint8_t *ub2 = ub1 + k * loop1_src_stride;
        uint8_t *gm2 = gm1 + k * loop1_dst_stride;

        for (int r = 0; r < n_burst; r++) {
            memcpy(gm2 + r * dst_stride,               //   GM dest row
                   ub2 + r * src_stride,               //   UB src row
                   len_burst);                          //   contiguous bytes
        }
    }
}
```

```
GM layout (32 × 32 f32, contiguous):

    |<- len_burst = 128B (32 × 4) ->|
    |<- src_stride = 128B --------->|
    +--[#######TILE#######]--+  row 0
    +--[#######TILE#######]--+  row 1
    ...
    +--[#######TILE#######]--+  row 31

UB layout (32 × 32 f32, 32B-aligned, contiguous):

    |<- dst_stride = 128B (32B-aligned) ->|
    +--[#######TILE#######]--+  row 0
    +--[#######TILE#######]--+  row 1
    ...
    +--[#######TILE#######]--+  row 31

    len_burst   = 32 × 4 = 128 bytes
    src_stride  = 128 bytes (contiguous rows)
    dst_stride  = 128 bytes (already 32B-aligned, no padding)
```

```mlir
// Simple 2D load — no multi-level loops needed
pto.set_loop_size_outtoub %c1_i64, %c1_i64 : i64, i64

pto.copy_gm_to_ubuf %arg0, %ub_in,
    %c0_i64,       // sid = 0
    %c32_i64,      // n_burst = 32 (32 rows)
    %c128_i64,     // len_burst = 128 bytes per row
    %c0_i64,       // left_padding = 0
    %c0_i64,       // right_padding = 0
    %false,        // data_select_bit = false
    %c0_i64,       // l2_cache_ctl = 0
    %c128_i64,     // src_stride = 128 bytes
    %c128_i64      // dst_stride = 128 bytes
    : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64,
      i64, i64, i1, i64, i64, i64
```

```
GM layout (1024 × 512 f16):

    col 0          col 128               col 512
    |              |                     |
    +--[###TILE###]+.....................+  row R
    +--[###TILE###]+.....................+  row R+1
    ...
    +--[###TILE###]+.....................+  row R+63

    |<--------- src_stride = 1024B ----------->|
    |<-len_burst=256B->|

    len_burst   = 128 × 2 = 256 bytes (128 f16 elements)
    src_stride  = 512 × 2 = 1024 bytes (start-to-start, full GM row)

UB layout (64 × 128 f16, 32B-aligned, contiguous):

    +--[###TILE###]--+  row 0  (256 bytes, 32B-aligned, no pad)
    +--[###TILE###]--+  row 1
    ...
    +--[###TILE###]--+  row 63

    dst_stride = 256 bytes (= len_burst, already 32B-aligned, no padding)
```

```mlir
// Simple 2D load — no multi-level loops needed
pto.set_loop_size_outtoub %c1_i64, %c1_i64 : i64, i64
pto.set_loop1_stride_outtoub %c0_i64, %c0_i64 : i64, i64
pto.set_loop2_stride_outtoub %c0_i64, %c0_i64 : i64, i64

pto.copy_gm_to_ubuf %gm_ptr, %ub_ptr,
    %c0_i64,       // sid = 0
    %c64_i64,      // n_burst = 64 (64 rows)
    %c256_i64,     // len_burst = 256 bytes per row
    %c0_i64,       // left_padding = 0
    %c0_i64,       // right_padding = 0
    %false,        // data_select_bit = false
    %c0_i64,       // l2_cache_ctl = 0
    %c1024_i64,    // src_stride = 1024 bytes (full matrix row)
    %c256_i64      // dst_stride = 256 bytes (tile row)
    : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64,
      i64, i64, i1, i64, i64, i64
```

```
GM (100 cols valid, contiguous):

    |<-len_burst=200B->|
    |<- src_stride=200B (start-to-start) ->|
    +--[####DATA####]-+  row 0
    +--[####DATA####]-+  row 1
    ...
    +--[####DATA####]-+  row 63

UB (128 cols wide, 32B-aligned, padded):

    |<--------- dst_stride = 256B (32B-aligned) --------->|
    |<-len_burst=200B->|<---- pad = 56B to 32B boundary ->|
    +--[####DATA####]-+[0000000 PAD 0000000000000000000000]+  row 0
    +--[####DATA####]-+[0000000 PAD 0000000000000000000000]+  row 1
    ...
    +--[####DATA####]-+[0000000 PAD 0000000000000000000000]+  row 63

    len_burst   = 100 × 2 = 200 bytes
    src_stride  = 200 bytes (start-to-start, contiguous in GM)
    dst_stride  = 128 × 2 = 256 bytes (32B-aligned tile width in UB)
    pad         = 256 - 200 = 56 bytes (padded to 32B boundary with pad_val)
```

```mlir
pto.set_loop_size_outtoub %c1_i64, %c1_i64 : i64, i64
pto.set_loop1_stride_outtoub %c0_i64, %c0_i64 : i64, i64
pto.set_loop2_stride_outtoub %c0_i64, %c0_i64 : i64, i64

pto.copy_gm_to_ubuf %gm_ptr, %ub_ptr,
    %c0_i64,       // sid = 0
    %c64_i64,      // n_burst = 64
    %c200_i64,     // len_burst = 200 bytes
    %c0_i64,       // left_padding = 0
    %c0_i64,       // right_padding = 0
    %true,         // data_select_bit = true (enable padding)
    %c0_i64,       // l2_cache_ctl = 0
    %c200_i64,     // src_stride = 200 bytes
    %c256_i64      // dst_stride = 256 bytes (32B-aligned)
    : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64,
      i64, i64, i1, i64, i64, i64
```

```
UB (source, 32B-aligned, 32 × 32 f32):

    |<- src_stride = 128B (32B-aligned) ->|
    |<- len_burst = 128B ->|
    +--[#######TILE#######]---+  row 0
    +--[#######TILE#######]---+  row 1
    ...
    +--[#######TILE#######]---+  row 31

    (no padding here — len_burst == src_stride)

GM (dest, 32 × 32 f32):

    |<- dst_stride = 128B ->|
    |<- len_burst = 128B -->|
    +--[#######TILE#######]---+  row 0
    +--[#######TILE#######]---+  row 1
    ...
    +--[#######TILE#######]---+  row 31
```

```mlir
// Configure MTE3 strides
pto.set_loop_size_ubtoout %c1_i64, %c1_i64 : i64, i64

pto.copy_ubuf_to_gm %ub_out, %arg1,
    %c0_i64,       // sid = 0
    %c32_i64,      // n_burst = 32
    %c128_i64,     // len_burst = 128 bytes
    %c0_i64,       // reserved = 0
    %c128_i64,     // dst_stride = 128 bytes
    %c128_i64      // src_stride = 128 bytes
    : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64
```

```
UB (source, 32B-aligned, 64 × 128 f16):

    |<- src_stride = 256B (32B-aligned) ->|
    |<- len_burst = 256B ->|
    +--[#####TILE#####]---+  row 0
    +--[#####TILE#####]---+  row 1
    ...
    +--[#####TILE#####]---+  row 63

    (no padding here — len_burst == src_stride)

GM (dest, into 1024 × 512 matrix):

    |<----------- dst_stride = 1024B (start-to-start) --------->|
    |<- len_burst = 256B ->|                                    |
    col 0          col 128                              col 512
    +--[#####TILE#####]---+.............................+  row R
    +--[#####TILE#####]---+.............................+  row R+1
    ...
    +--[#####TILE#####]---+.............................+  row R+63

    MTE3 reads len_burst bytes from each 32B-aligned UB row,
    writes only len_burst bytes per GM row (stride controls row spacing).
```

```mlir
// Configure MTE3 strides
pto.set_loop_size_ubtoout %c1_i64, %c1_i64 : i64, i64
pto.set_loop1_stride_ubtoout %c0_i64, %c0_i64 : i64, i64
pto.set_loop2_stride_ubtoout %c0_i64, %c0_i64 : i64, i64

pto.copy_ubuf_to_gm %ub_ptr, %gm_ptr,
    %c0_i64,       // sid = 0
    %c64_i64,      // n_burst = 64
    %c256_i64,     // len_burst = 256 bytes
    %c0_i64,       // reserved = 0
    %c1024_i64,    // dst_stride = 1024 bytes (GM row)
    %c256_i64      // src_stride = 256 bytes (UB row)
    : !pto.ptr<f16, ub>, !pto.ptr<f16, gm>, i64, i64, i64, i64, i64, i64
```

```
GM [4, 8, 128] f16 (contiguous):        UB (4 tiles laid out sequentially):

    batch 0: 8 rows × 256 bytes          [batch 0: 8×128][batch 1: 8×128]
    batch 1: 8 rows × 256 bytes          [batch 2: 8×128][batch 3: 8×128]
    batch 2: 8 rows × 256 bytes
    batch 3: 8 rows × 256 bytes          loop1 src_stride = 2048 bytes (8 × 256)
                                          loop1 dst_stride = 2048 bytes (8 × 256)
    Each batch = 8 × 256 = 2048 bytes     loop1_count = 4 (iterate over batches)
```

```mlir
// loop1_count = 4 batches, loop2_count = 1 (not used)
pto.set_loop_size_outtoub %c4_i64, %c1_i64 : i64, i64

// loop1 stride: advance by one batch (2048 bytes) in both GM and UB
pto.set_loop1_stride_outtoub %c2048_i64, %c2048_i64 : i64, i64
pto.set_loop2_stride_outtoub %c0_i64, %c0_i64 : i64, i64

pto.copy_gm_to_ubuf %gm_ptr, %ub_ptr,
    %c0_i64,       // sid = 0
    %c8_i64,       // n_burst = 8 rows per batch
    %c256_i64,     // len_burst = 256 bytes per row
    %c0_i64,       // left_padding = 0
    %c0_i64,       // right_padding = 0
    %false,        // data_select_bit = false
    %c0_i64,       // l2_cache_ctl = 0
    %c256_i64,     // src_stride = 256 (contiguous rows)
    %c256_i64      // dst_stride = 256 (contiguous rows)
    : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64,
      i64, i64, i1, i64, i64, i64
```

```
loop1 iter 0: gm_ptr + 0×2048 → ub_ptr + 0×2048, DMA 8 rows × 256B
loop1 iter 1: gm_ptr + 1×2048 → ub_ptr + 1×2048, DMA 8 rows × 256B
loop1 iter 2: gm_ptr + 2×2048 → ub_ptr + 2×2048, DMA 8 rows × 256B
loop1 iter 3: gm_ptr + 3×2048 → ub_ptr + 3×2048, DMA 8 rows × 256B
```

## Detailed Notes

```mlir
pto.copy_ubuf_to_ubuf %source, %dest, %sid, %n_burst, %len_burst, %src_stride, %dst_stride
    : !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64 x5
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `%source` | UB source pointer |
| `%dest` | UB destination pointer |
| `%sid` | Stream ID |
| `%n_burst` | Number of bursts |
| `%len_burst` | Length per burst |
| `%src_stride` | Source stride |
| `%dst_stride` | Destination stride |

## Burst / Stride / Pad Model

All A5 DMA addresses are **stride-based**: stride is the distance from the start of one row to the start of the next row (`stride >= lenBurst`). There is no separate "gap" parameter.

### Key Terms

```
burst    = lenBurst contiguous bytes transferred per row
stride   = distance (bytes) from start of row[r] to start of row[r+1]
pad      = ub_stride - lenBurst, padded to the 32B alignment boundary
```

### Alignment Constraints

- **UB addresses** (both source and destination) must be **32-byte aligned**.
- **GM→UB padding**: When `data_select_bit = true`, each UB row is padded from `lenBurst` up to the **32B-aligned boundary** of `ub_stride` with `pad_val` (set via `set_mov_pad_val`). This ensures every UB row starts at a 32B-aligned offset.
- **UB→GM de-padding**: MTE3 reads `lenBurst` bytes from each 32B-aligned UB row (skipping any padding that was added during load), writing only valid data to GM. This effectively strips padding on store.

### 2D Diagram: GM→UB (pto.copy_gm_to_ubuf)

```
GM (source, `!pto.ptr<T, gm>`):

          |<--- src_stride (start-to-start) --->|
          |<- len_burst ->|                     |
Row 0:    [##DATA########]......................|
Row 1:    [##DATA########]......................|
Row 2:    [##DATA########]......................|
          ...
Row N-1:  [##DATA########]

UB (destination, `!pto.ptr<T, ub>`, 32B-aligned):

          |<---------- dst_stride (32B-aligned) ---------->|
          |<- len_burst ->|<- pad (to 32B boundary) ->|    |
Row 0:    [##DATA########][000000 PAD 000000000000000]
Row 1:    [##DATA########][000000 PAD 000000000000000]
Row 2:    [##DATA########][000000 PAD 000000000000000]
          ...
Row N-1:  [##DATA########][000000 PAD 000000000000000]

N = n_burst
stride = start of row[r] to start of row[r+1]
pad    = filled with pad_val to 32B boundary (data_select_bit=true)
[DATA] = valid data transferred by DMA
[PAD]  = pad_val fill (set via set_mov_pad_val)
```

### 2D Diagram: UB→GM (pto.copy_ubuf_to_gm)

```
UB (source, `!pto.ptr<T, ub>`, 32B-aligned start addr):

          |<---------- src_stride (32B-aligned) --------->|
          |<- len_burst ->|<-- pad (ignored on read) -->| |
Row 0:    [##DATA########][000 pad 000000000000000000]
Row 1:    [##DATA########][000 pad 000000000000000000]
Row 2:    [##DATA########][000 pad 000000000000000000]
          ...
Row N-1:  [##DATA########][000 pad 000000000000000000]

GM (destination, `!pto.ptr<T, gm>`):

          |<--- dst_stride (start-to-start) --->|
          |<- len_burst ->|                     |
Row 0:    [##DATA########]......................|
Row 1:    [##DATA########]......................|
Row 2:    [##DATA########]......................|
          ...
Row N-1:  [##DATA########]

N = n_burst
MTE3 reads only len_burst bytes from each UB row (de-padding).
Only len_burst bytes are written to each GM row.
```

## Multi-Level Loop Semantics (C Code)

The full DMA transfer is a nested loop. The HW loop registers (set before the copy) control the outer levels, and the copy instruction parameters control the innermost burst level.

### GM→UB Full Loop

```c
// C equivalent of what the HW executes:
for (int j = 0; j < loop2_count; j++) {                // HW outer loop
    uint8_t *gm1 = gm_src + j * loop2_src_stride;
    uint8_t *ub1 = ub_dst + j * loop2_dst_stride;

    for (int k = 0; k < loop1_count; k++) {            // HW inner loop
        uint8_t *gm2 = gm1 + k * loop1_src_stride;
        uint8_t *ub2 = ub1 + k * loop1_dst_stride;

        for (int r = 0; r < n_burst; r++) {            // burst engine
            memcpy(ub2 + r * dst_stride,               //   UB dest row
                   gm2 + r * src_stride,               //   GM src row
                   len_burst);                          //   contiguous bytes
            if (data_select_bit)
                memset(ub2 + r * dst_stride + len_burst,
                       pad_val, dst_stride - len_burst);
        }
    }
}
```

### UB→GM Full Loop

```c
// C equivalent:
for (int j = 0; j < loop2_count; j++) {
    uint8_t *ub1 = ub_src + j * loop2_src_stride;
    uint8_t *gm1 = gm_dst + j * loop2_dst_stride;

    for (int k = 0; k < loop1_count; k++) {
        uint8_t *ub2 = ub1 + k * loop1_src_stride;
        uint8_t *gm2 = gm1 + k * loop1_dst_stride;

        for (int r = 0; r < n_burst; r++) {
            memcpy(gm2 + r * dst_stride,               //   GM dest row
                   ub2 + r * src_stride,               //   UB src row
                   len_burst);                          //   contiguous bytes
        }
    }
}
```

## Example 1: GM→UB — Load a 32×32 f32 Tile (Simple Case)

Load a 32×32 f32 tile from GM into UB. This matches the `abs_kernel_2d` test case.

```
GM layout (32 × 32 f32, contiguous):

    |<- len_burst = 128B (32 × 4) ->|
    |<- src_stride = 128B --------->|
    +--[#######TILE#######]--+  row 0
    +--[#######TILE#######]--+  row 1
    ...
    +--[#######TILE#######]--+  row 31

UB layout (32 × 32 f32, 32B-aligned, contiguous):

    |<- dst_stride = 128B (32B-aligned) ->|
    +--[#######TILE#######]--+  row 0
    +--[#######TILE#######]--+  row 1
    ...
    +--[#######TILE#######]--+  row 31

    len_burst   = 32 × 4 = 128 bytes
    src_stride  = 128 bytes (contiguous rows)
    dst_stride  = 128 bytes (already 32B-aligned, no padding)
```

```mlir
// Simple 2D load — no multi-level loops needed
pto.set_loop_size_outtoub %c1_i64, %c1_i64 : i64, i64

pto.copy_gm_to_ubuf %arg0, %ub_in,
    %c0_i64,       // sid = 0
    %c32_i64,      // n_burst = 32 (32 rows)
    %c128_i64,     // len_burst = 128 bytes per row
    %c0_i64,       // left_padding = 0
    %c0_i64,       // right_padding = 0
    %false,        // data_select_bit = false
    %c0_i64,       // l2_cache_ctl = 0
    %c128_i64,     // src_stride = 128 bytes
    %c128_i64      // dst_stride = 128 bytes
    : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64,
      i64, i64, i1, i64, i64, i64
```

## Example 2: GM→UB — Load a 2D Tile from a Larger Matrix

Load a 64×128 tile (f16) from a 1024×512 matrix in GM into UB.

```
GM layout (1024 × 512 f16):

    col 0          col 128               col 512
    |              |                     |
    +--[###TILE###]+.....................+  row R
    +--[###TILE###]+.....................+  row R+1
    ...
    +--[###TILE###]+.....................+  row R+63

    |<--------- src_stride = 1024B ----------->|
    |<-len_burst=256B->|

    len_burst   = 128 × 2 = 256 bytes (128 f16 elements)
    src_stride  = 512 × 2 = 1024 bytes (start-to-start, full GM row)

UB layout (64 × 128 f16, 32B-aligned, contiguous):

    +--[###TILE###]--+  row 0  (256 bytes, 32B-aligned, no pad)
    +--[###TILE###]--+  row 1
    ...
    +--[###TILE###]--+  row 63

    dst_stride = 256 bytes (= len_burst, already 32B-aligned, no padding)
```

```mlir
// Simple 2D load — no multi-level loops needed
pto.set_loop_size_outtoub %c1_i64, %c1_i64 : i64, i64
pto.set_loop1_stride_outtoub %c0_i64, %c0_i64 : i64, i64
pto.set_loop2_stride_outtoub %c0_i64, %c0_i64 : i64, i64

pto.copy_gm_to_ubuf %gm_ptr, %ub_ptr,
    %c0_i64,       // sid = 0
    %c64_i64,      // n_burst = 64 (64 rows)
    %c256_i64,     // len_burst = 256 bytes per row
    %c0_i64,       // left_padding = 0
    %c0_i64,       // right_padding = 0
    %false,        // data_select_bit = false
    %c0_i64,       // l2_cache_ctl = 0
    %c1024_i64,    // src_stride = 1024 bytes (full matrix row)
    %c256_i64      // dst_stride = 256 bytes (tile row)
    : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64,
      i64, i64, i1, i64, i64, i64
```

## Example 3: GM→UB — Load with Padding

Load 100 valid columns from GM into a 128-wide UB tile (f16). The remaining 28 columns are zero-padded.

```
GM (100 cols valid, contiguous):

    |<-len_burst=200B->|
    |<- src_stride=200B (start-to-start) ->|
    +--[####DATA####]-+  row 0
    +--[####DATA####]-+  row 1
    ...
    +--[####DATA####]-+  row 63

UB (128 cols wide, 32B-aligned, padded):

    |<--------- dst_stride = 256B (32B-aligned) --------->|
    |<-len_burst=200B->|<---- pad = 56B to 32B boundary ->|
    +--[####DATA####]-+[0000000 PAD 0000000000000000000000]+  row 0
    +--[####DATA####]-+[0000000 PAD 0000000000000000000000]+  row 1
    ...
    +--[####DATA####]-+[0000000 PAD 0000000000000000000000]+  row 63

    len_burst   = 100 × 2 = 200 bytes
    src_stride  = 200 bytes (start-to-start, contiguous in GM)
    dst_stride  = 128 × 2 = 256 bytes (32B-aligned tile width in UB)
    pad         = 256 - 200 = 56 bytes (padded to 32B boundary with pad_val)
```

```mlir
pto.set_loop_size_outtoub %c1_i64, %c1_i64 : i64, i64
pto.set_loop1_stride_outtoub %c0_i64, %c0_i64 : i64, i64
pto.set_loop2_stride_outtoub %c0_i64, %c0_i64 : i64, i64

pto.copy_gm_to_ubuf %gm_ptr, %ub_ptr,
    %c0_i64,       // sid = 0
    %c64_i64,      // n_burst = 64
    %c200_i64,     // len_burst = 200 bytes
    %c0_i64,       // left_padding = 0
    %c0_i64,       // right_padding = 0
    %true,         // data_select_bit = true (enable padding)
    %c0_i64,       // l2_cache_ctl = 0
    %c200_i64,     // src_stride = 200 bytes
    %c256_i64      // dst_stride = 256 bytes (32B-aligned)
    : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64,
      i64, i64, i1, i64, i64, i64
```

## Example 4: UB→GM — Store a 32×32 f32 Tile (Simple Case)

Store a 32×32 f32 tile from UB back to GM. This matches the `abs_kernel_2d` test case.

```
UB (source, 32B-aligned, 32 × 32 f32):

    |<- src_stride = 128B (32B-aligned) ->|
    |<- len_burst = 128B ->|
    +--[#######TILE#######]---+  row 0
    +--[#######TILE#######]---+  row 1
    ...
    +--[#######TILE#######]---+  row 31

    (no padding here — len_burst == src_stride)

GM (dest, 32 × 32 f32):

    |<- dst_stride = 128B ->|
    |<- len_burst = 128B -->|
    +--[#######TILE#######]---+  row 0
    +--[#######TILE#######]---+  row 1
    ...
    +--[#######TILE#######]---+  row 31
```

```mlir
// Configure MTE3 strides
pto.set_loop_size_ubtoout %c1_i64, %c1_i64 : i64, i64

pto.copy_ubuf_to_gm %ub_out, %arg1,
    %c0_i64,       // sid = 0
    %c32_i64,      // n_burst = 32
    %c128_i64,     // len_burst = 128 bytes
    %c0_i64,       // reserved = 0
    %c128_i64,     // dst_stride = 128 bytes
    %c128_i64      // src_stride = 128 bytes
    : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64
```

## Example 5: UB→GM — Store a 2D Tile Back to a Larger Matrix

Store a 64×128 tile (f16) from UB back to a 1024×512 GM matrix at an offset.

```
UB (source, 32B-aligned, 64 × 128 f16):

    |<- src_stride = 256B (32B-aligned) ->|
    |<- len_burst = 256B ->|
    +--[#####TILE#####]---+  row 0
    +--[#####TILE#####]---+  row 1
    ...
    +--[#####TILE#####]---+  row 63

    (no padding here — len_burst == src_stride)

GM (dest, into 1024 × 512 matrix):

    |<----------- dst_stride = 1024B (start-to-start) --------->|
    |<- len_burst = 256B ->|                                    |
    col 0          col 128                              col 512
    +--[#####TILE#####]---+.............................+  row R
    +--[#####TILE#####]---+.............................+  row R+1
    ...
    +--[#####TILE#####]---+.............................+  row R+63

    MTE3 reads len_burst bytes from each 32B-aligned UB row,
    writes only len_burst bytes per GM row (stride controls row spacing).
```

```mlir
// Configure MTE3 strides
pto.set_loop_size_ubtoout %c1_i64, %c1_i64 : i64, i64
pto.set_loop1_stride_ubtoout %c0_i64, %c0_i64 : i64, i64
pto.set_loop2_stride_ubtoout %c0_i64, %c0_i64 : i64, i64

pto.copy_ubuf_to_gm %ub_ptr, %gm_ptr,
    %c0_i64,       // sid = 0
    %c64_i64,      // n_burst = 64
    %c256_i64,     // len_burst = 256 bytes
    %c0_i64,       // reserved = 0
    %c1024_i64,    // dst_stride = 1024 bytes (GM row)
    %c256_i64      // src_stride = 256 bytes (UB row)
    : !pto.ptr<f16, ub>, !pto.ptr<f16, gm>, i64, i64, i64, i64, i64, i64
```

## Example 6: GM→UB with Multi-Level Loop (Batch of Tiles)

Load 4 batches of 8×128 tiles from a [4, 8, 128] f16 tensor using loop1.

```
GM [4, 8, 128] f16 (contiguous):        UB (4 tiles laid out sequentially):

    batch 0: 8 rows × 256 bytes          [batch 0: 8×128][batch 1: 8×128]
    batch 1: 8 rows × 256 bytes          [batch 2: 8×128][batch 3: 8×128]
    batch 2: 8 rows × 256 bytes
    batch 3: 8 rows × 256 bytes          loop1 src_stride = 2048 bytes (8 × 256)
                                          loop1 dst_stride = 2048 bytes (8 × 256)
    Each batch = 8 × 256 = 2048 bytes     loop1_count = 4 (iterate over batches)
```

```mlir
// loop1_count = 4 batches, loop2_count = 1 (not used)
pto.set_loop_size_outtoub %c4_i64, %c1_i64 : i64, i64

// loop1 stride: advance by one batch (2048 bytes) in both GM and UB
pto.set_loop1_stride_outtoub %c2048_i64, %c2048_i64 : i64, i64
pto.set_loop2_stride_outtoub %c0_i64, %c0_i64 : i64, i64

pto.copy_gm_to_ubuf %gm_ptr, %ub_ptr,
    %c0_i64,       // sid = 0
    %c8_i64,       // n_burst = 8 rows per batch
    %c256_i64,     // len_burst = 256 bytes per row
    %c0_i64,       // left_padding = 0
    %c0_i64,       // right_padding = 0
    %false,        // data_select_bit = false
    %c0_i64,       // l2_cache_ctl = 0
    %c256_i64,     // src_stride = 256 (contiguous rows)
    %c256_i64      // dst_stride = 256 (contiguous rows)
    : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64,
      i64, i64, i1, i64, i64, i64
```

Execution trace:

```
loop1 iter 0: gm_ptr + 0×2048 → ub_ptr + 0×2048, DMA 8 rows × 256B
loop1 iter 1: gm_ptr + 1×2048 → ub_ptr + 1×2048, DMA 8 rows × 256B
loop1 iter 2: gm_ptr + 2×2048 → ub_ptr + 2×2048, DMA 8 rows × 256B
loop1 iter 3: gm_ptr + 3×2048 → ub_ptr + 3×2048, DMA 8 rows × 256B
```

## Related Ops / Family Links

- Family overview: [DMA Copy](../../dma-copy.md)
- Previous op in family: [pto.copy_ubuf_to_gm](./copy-ubuf-to-gm.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
