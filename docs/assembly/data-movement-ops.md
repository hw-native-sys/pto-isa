# Data Movement / Layout

This document describes data movement and layout transformation operations.

**Total Operations:** 14

---

## Operations

### TEXTRACT

For detailed instruction documentation, see [isa/TEXTRACT](../isa/tile/ops/layout-and-rearrangement/textract.md)

**AS Level 1 (SSA):**

```text
%dst = pto.textract %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.textract ins(%src, %idxrow, %idxcol : !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TEXTRACT_FP

For detailed instruction documentation, see [isa/TEXTRACT_FP](../isa/tile/ops/layout-and-rearrangement/textract.md)

**AS Level 1 (SSA):**

```text
%dst = pto.textract_fp %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.textract_fp ins(%src, %idxrow, %idxcol : !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TIMG2COL

**AS Level 1 (SSA):**

```text
%dst = pto.timg2col %src : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.timg2col ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TINSERT

For detailed instruction documentation, see [isa/TINSERT](../isa/tile/ops/layout-and-rearrangement/tinsert.md)

**AS Level 1 (SSA):**

```text
%dst = pto.tinsert %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.tinsert ins(%src[%r0, %r1] : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TINSERT_FP

For detailed instruction documentation, see [isa/TINSERT_FP](../isa/tile/ops/layout-and-rearrangement/tinsert.md)

**AS Level 1 (SSA):**

```text
%dst = pto.tinsert_fp %src, %fp, %idxrow, %idxcol : (!pto.tile<...>, !pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.tinsert_fp ins(%src, %fp, %idxrow, %idxcol : !pto.tile_buf<...>, !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TFILLPAD

For detailed instruction documentation, see [isa/TFILLPAD](../isa/tile/ops/layout-and-rearrangement/tfillpad.md)

**AS Level 1 (SSA):**

```text
%dst = pto.tfillpad %src : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.tfillpad ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TFILLPAD_INPLACE

For detailed instruction documentation, see [isa/TFILLPAD_INPLACE](../isa/tile/ops/layout-and-rearrangement/tfillpad-inplace.md)

**AS Level 1 (SSA):**

```text
%dst = pto.tfillpad_inplace %src : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.tfillpad_inplace ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TFILLPAD_EXPAND

For detailed instruction documentation, see [isa/TFILLPAD_EXPAND](../isa/tile/ops/layout-and-rearrangement/tfillpad-expand.md)

**AS Level 1 (SSA):**

```text
%dst = pto.tfillpad_expand %src : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.tfillpad_expand ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TMOV

For detailed instruction documentation, see [isa/TMOV](../isa/tile/ops/layout-and-rearrangement/tmov.md)

**AS Level 1 (SSA):**

```text
%dst = pto.tmov.s2d %src  : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.tmov ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TMOV_FP

For detailed instruction documentation, see [isa/TMOV_FP](../isa/tile/ops/layout-and-rearrangement/tmov.md)

**AS Level 1 (SSA):**

```text
%dst = pto.tmov.fp %src, %fp : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.tmov.fp ins(%src, %fp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TRESHAPE

For detailed instruction documentation, see [isa/TRESHAPE](../isa/tile/ops/layout-and-rearrangement/treshape.md)

**AS Level 1 (SSA):**

```text
%dst = pto.treshape %src : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.treshape ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TTRANS

For detailed instruction documentation, see [isa/TTRANS](../isa/tile/ops/layout-and-rearrangement/ttrans.md)

**AS Level 1 (SSA):**

```text
%dst = pto.ttrans %src : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.ttrans ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TIMG2COL

For detailed instruction documentation, see [isa/TIMG2COL](../isa/tile/ops/layout-and-rearrangement/timg2col.md)

**AS Level 1 (SSA):**

```text
%dst = pto.timg2col %src : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.timg2col ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---
