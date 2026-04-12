<!-- Generated from `docs/isa/vector/data-rearrangement.md` -->

# Vector Families: Data Rearrangement

This page documents `pto.v*` rearrangement families. These operations permute or repack vector-visible data without turning into tile movement or DMA, so they stay in the vector surface.

> **Category:** In-register data movement and permutation
> **Pipeline:** PIPE_V (Vector Core)

Operations that rearrange data within or between vector registers without memory access.

## Common Operand Model

- `%lhs` / `%rhs` are source vector register values.
- `%src` is a single source vector register value.
- `%result` is the destination vector register value unless an op explicitly
  returns multiple vectors.
- These families do not access UB directly; they only rearrange register
  contents.

---

## Interleave / Deinterleave

### `pto.vintlv`

- **syntax:** `%low, %high = pto.vintlv %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- **semantics:** Interleave elements from two sources.

```c
// Interleave: merge even/odd elements from two sources
// low  = {src0[0], src1[0], src0[1], src1[1], ...}
// high = {src0[N/2], src1[N/2], src0[N/2+1], src1[N/2+1], ...}
```

- **inputs:** `%lhs` and `%rhs` are the two source vectors.
- **outputs:** `%low` and `%high` are the two destination vectors.
- **constraints and limitations:** The two outputs form a paired interleave
  result. The PTO ISA vector surface representation exposes that pair as two SSA results, and the pair ordering MUST
  be preserved.

---

### `pto.vdintlv`

- **syntax:** `%low, %high = pto.vdintlv %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- **semantics:** Deinterleave elements into even/odd.

```c
// Deinterleave: separate even/odd elements
// low  = {src0[0], src0[2], src0[4], ...}  // even
// high = {src0[1], src0[3], src0[5], ...}  // odd
```

- **inputs:** `%lhs` and `%rhs` represent the interleaved source stream in the
  current PTO ISA vector surface representation.
- **outputs:** `%low` and `%high` are the separated destination vectors.
- **constraints and limitations:** The two outputs form the even/odd
  deinterleave result pair, and their ordering MUST be preserved.

---

## Slide / Shift

### `pto.vslide`

- **syntax:** `%result = pto.vslide %src0, %src1, %amt : !pto.vreg<NxT>, !pto.vreg<NxT>, i16 -> !pto.vreg<NxT>`
- **semantics:** Concatenate two vectors and extract N-element window at offset.

```c
// Conceptually: tmp[0..2N-1] = {src1, src0}
// dst[i] = tmp[amt + i]
if (amt >= 0)
    for (int i = 0; i < N; i++)
        dst[i] = (i >= amt) ? src0[i - amt] : src1[N - amt + i];
```

**Use case:** Sliding window operations, shift register patterns.

- **inputs:** `%src0` and `%src1` provide the concatenated source window and
  `%amt` selects the extraction offset.
- **outputs:** `%result` is the extracted destination window.
- **constraints and limitations:** `pto.vslide` operates on the logical
  concatenation of `%src1` and `%src0`. The source order and extraction offset
  MUST be preserved exactly.

---

### `pto.vshift`

- **syntax:** `%result = pto.vshift %src, %amt : !pto.vreg<NxT>, i16 -> !pto.vreg<NxT>`
- **semantics:** Single-source slide (shift with zero fill).

```c
for (int i = 0; i < N; i++)
    dst[i] = (i >= amt) ? src[i - amt] : 0;
```

- **inputs:** `%src` is the source vector and `%amt` is the slide amount.
- **outputs:** `%result` is the shifted vector.
- **constraints and limitations:** This surface represents the single-source
  slide/shift family. Zero-fill versus other fill behavior MUST match the
  selected form.

---

## Compress / Expand

### `pto.vsqz`

- **syntax:** `%result = pto.vsqz %src, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **semantics:** Compress — pack active lanes to front.

```c
int j = 0;
for (int i = 0; i < N; i++)
    if (mask[i]) dst[j++] = src[i];
while (j < N) dst[j++] = 0;
```

**Use case:** Sparse data compaction, filtering.

- **inputs:** `%src` is the source vector and `%mask` selects which elements are
  kept.
- **outputs:** `%result` is the compacted vector.
- **constraints and limitations:** This is a reduction-style compaction family.
  Preserved element order MUST match source lane order.

---

### `pto.vusqz`

- **syntax:** `%result = pto.vusqz %mask : !pto.mask -> !pto.vreg<NxT>`
- **semantics:** Expand — scatter front elements to active positions.

```c
int j = 0;
for (int i = 0; i < N; i++)
    if (mask[i]) dst[i] = src_front[j++];
    else dst[i] = 0;
```

- **inputs:** `%mask` is the expansion/placement predicate.
- **outputs:** `%result` is the expanded vector image.
- **constraints and limitations:** The source-front stream is implicit in the
  current surface. Lane placement for active and inactive positions MUST be
  preserved exactly.

---

## Permutation

### `pto.vperm`

- **syntax:** `%result = pto.vperm %src, %index : !pto.vreg<NxT>, !pto.vreg<NxI> -> !pto.vreg<NxT>`
- **semantics:** In-register permute (table lookup). **Not** memory gather.

```c
for (int i = 0; i < N; i++)
    dst[i] = src[index[i] % N];
```

**Note:** This operates on register contents, unlike `pto.vgather2` which reads from UB memory.

- **inputs:** `%src` is the source vector and `%index` supplies per-lane source
  indices.
- **outputs:** `%result` is the permuted vector.
- **constraints and limitations:** This is an in-register permutation family.
  `%index` values outside the legal range follow the wrap/clamp behavior of the
  selected form.

---

### `pto.vselr`

- **syntax:** `%result = pto.vselr %src0, %src1 : !pto.vreg<NxT>, !pto.vreg<NxI> -> !pto.vreg<NxT>`
- **semantics:** Register select with reversed mask semantics.

```c
for (int i = 0; i < N; i++)
    dst[i] = mask[i] ? src1[i] : src0[i];
```

- **inputs:** `%src0` and `%src1` are source vectors.
- **outputs:** `%result` is the selected vector.
- **constraints and limitations:** This page records the rearrangement use of
  the family; the compare/select page documents the same name from the predicate
  selection perspective.

---

## Pack / Unpack

### `pto.vpack`

- **syntax:** `%result = pto.vpack %src0, %src1, %part : !pto.vreg<NxT_wide>, !pto.vreg<NxT_wide>, index -> !pto.vreg<2NxT_narrow>`
- **semantics:** Narrowing pack — two wide vectors to one narrow vector.

```c
// e.g., two vreg<64xi32> → one vreg<128xi16>
for (int i = 0; i < N; i++) {
    dst[i]     = truncate(src0[i]);
    dst[N + i] = truncate(src1[i]);
}
```

- **inputs:** `%src0` and `%src1` are wide source vectors and `%part` selects
  the packing submode.
- **outputs:** `%result` is the packed narrow vector.
- **constraints and limitations:** Packing is a narrowing conversion. Source
  values that do not fit the destination width follow the truncation semantics
  of the selected packing mode.

---

### `pto.vsunpack`

- **syntax:** `%result = pto.vsunpack %src, %part : !pto.vreg<NxT_narrow>, index -> !pto.vreg<N/2xT_wide>`
- **semantics:** Sign-extending unpack — narrow to wide (half).

```c
// e.g., vreg<128xi16> → vreg<64xi32> (one half)
for (int i = 0; i < N/2; i++)
    dst[i] = sign_extend(src[part_offset + i]);
```

- **inputs:** `%src` is the packed narrow vector and `%part` selects which half
  is unpacked.
- **outputs:** `%result` is the widened vector.
- **constraints and limitations:** This is the sign-extending unpack family.

---

### `pto.vzunpack`

- **syntax:** `%result = pto.vzunpack %src, %part : !pto.vreg<NxT_narrow>, index -> !pto.vreg<N/2xT_wide>`
- **semantics:** Zero-extending unpack — narrow to wide (half).

```c
for (int i = 0; i < N/2; i++)
    dst[i] = zero_extend(src[part_offset + i]);
```

- **inputs:** `%src` is the packed narrow vector and `%part` selects which half
  is unpacked.
- **outputs:** `%result` is the widened vector.
- **constraints and limitations:** This is the zero-extending unpack family.

---

## Typical Usage

```mlir
// AoS → SoA conversion using deinterleave
%even, %odd = pto.vdintlv %interleaved0, %interleaved1
    : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

// Filter: keep only elements passing condition
%pass_mask = pto.vcmps %values, %threshold, %all, "gt"
    : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.mask
%compacted = pto.vsqz %values, %pass_mask
    : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// Sliding window sum
%prev_window = pto.vslide %curr, %prev, %c1
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, i16 -> !pto.vreg<64xf32>
%window_sum = pto.vadd %curr, %prev_window, %all
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// Type narrowing via pack
%packed_i16 = pto.vpack %wide0_i32, %wide1_i32, %c0
    : !pto.vreg<64xi32>, !pto.vreg<64xi32>, index -> !pto.vreg<128xi16>
```

---

## V2 Interleave Forms

### `pto.vintlvv2`

- **syntax:** `%result = pto.vintlvv2 %lhs, %rhs, "PART" : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **inputs:** `%lhs` and `%rhs` are source vectors and `PART` selects the
  returned half of the V2 interleave result.
- **outputs:** `%result` is the selected interleave half.
- **constraints and limitations:** This op exposes only one half of the V2
  result in SSA form.

### `pto.vdintlvv2`

- **syntax:** `%result = pto.vdintlvv2 %lhs, %rhs, "PART" : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **inputs:** `%lhs` and `%rhs` are source vectors and `PART` selects the
  returned half of the V2 deinterleave result.
- **outputs:** `%result` is the selected deinterleave half.
- **constraints and limitations:** This op exposes only one half of the V2
  result in SSA form.
