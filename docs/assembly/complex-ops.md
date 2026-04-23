# Complex

This document describes complex operations including sorting, gathering, quantization, and random number generation.

**Total Operations:** 18

---

## Operations

### TPRINT

For detailed instruction documentation, see [isa/TPRINT](../isa/TPRINT.md)


**AS Level 1 (SSA):**
```text
pto.tprint %src : !pto.tile<...> | !pto.partition_tensor_view<MxNxdtype> -> ()
```

**AS Level 2 (DPS):**
```text
pto.tprint ins(%src : !pto.tile_buf<...> | !pto.partition_tensor_view<MxNxdtype>)
```

---

### TMRGSORT

For detailed instruction documentation, see [isa/TMRGSORT](../isa/TMRGSORT.md)


**AS Level 1 (SSA):**
```text
%dst = pto.tmrgsort %src, %blockLen : (!pto.tile<...>, dtype) -> !pto.tile<...>
%dst, %executed = pto.tmrgsort %src0, %src1, %src2, %src3 {exhausted = false}
 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> (!pto.tile<...>, vector<4xi16>)
```

**AS Level 2 (DPS):**
```text
pto.tmrgsort ins(%src, %blockLen : !pto.tile_buf<...>, dtype)  outs(%dst : !pto.tile_buf<...>)
pto.tmrgsort ins(%src0, %src1, %src2, %src3 {exhausted = false} : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
outs(%dst, %executed : !pto.tile_buf<...>, vector<4xi16>)
```

---

### TSORT32

For detailed instruction documentation, see [isa/TSORT32](../isa/TSORT32.md)

**AS Level 1 (SSA):**
```text
%dst, %idx = pto.tsort32 %src : !pto.tile<...> -> (!pto.tile<...>, !pto.tile<...>)
```

**AS Level 2 (DPS):**
```text
pto.tsort32 ins(%src : !pto.tile_buf<...>) outs(%dst, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>)
```

---

### TGATHER

For detailed instruction documentation, see [isa/TGATHER](../isa/TGATHER.md)


**AS Level 1 (SSA):**
```text
%dst = pto.tgather %src, %indices : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
%dst = pto.tgather %src {maskPattern = #pto.mask_pattern<P0101>}: !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS):**
```text
pto.tgather ins(%src, %indices : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
pto.tgather ins(%src, {maskPattern = #pto.mask_pattern<P0101>} : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCI

For detailed instruction documentation, see [isa/TCI](../isa/TCI.md)


**AS Level 1 (SSA):**
```text
%dst = pto.tci %scalar {descending = false} : dtype -> !pto.tile<...>
```

**AS Level 2 (DPS):**
```text
pto.tci ins(%scalar {descending = false} : dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TTRI

For detailed instruction documentation, see [isa/TTRI](../isa/TTRI.md)


**AS Level 1 (SSA):**
```text
%dst = pto.ttri %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS):**
```text
pto.ttri ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TRANDOM

For detailed instruction documentation, see [isa/TRANDOM](../isa/TRANDOM.md)


**AS Level 1 (SSA):**
```text
%dst = pto.trandom %key, %counter {rounds = 10} : -> !pto.tile<...>
```

**AS Level 2 (DPS)：**
```text
pto.trandom ins(%key, %counter {rounds = 10} : dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TPARTADD

For detailed instruction documentation, see [isa/TPARTADD](../isa/TPARTADD.md)


**AS Level 1 (SSA):**
```text
%dst = pto.tpartadd %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS):**
```text
pto.tpartadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TPARTMUL

For detailed instruction documentation, see [isa/TPARTMUL](../isa/TPARTMUL.md)


**AS Level 1 (SSA):**
```text
%dst = pto.tpartmul %src0, %src1 : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS):**
```text
pto.tpartmul ins(%src0, %src1 : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TPARTMAX

For detailed instruction documentation, see [isa/TPARTMAX](../isa/TPARTMAX.md)


**AS Level 1 (SSA):**
```text
%dst = pto.tpartmax %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS):**
```text
pto.tpartmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TPARTMIN

For detailed instruction documentation, see [isa/TPARTMIN](../isa/TPARTMIN.md)


**AS Level 1 (SSA):**
```text
%dst = pto.tpartmin %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS):**
```text
pto.tpartmin ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TPARTARGMAX

For detailed instruction documentation, see [isa/TPARTARGMAX](../isa/TPARTARGMAX.md)


**AS Level 1 (SSA):**
```text
%dstVal, %dstIdx = pto.tpartargmax %src0Val, %src1Val, %src0Idx, %src1Idx : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> (!pto.tile<...>, !pto.tile<...>)
```

**AS Level 2 (DPS):**
```text
pto.tpartargmax ins(%src0Val, %src1Val, %src0Idx, %src1Idx : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dstVal, %dstIdx : !pto.tile_buf<...>, !pto.tile_buf<...>)
```

---

### TPARTARGMIN

For detailed instruction documentation, see [isa/TPARTARGMIN](../isa/TPARTARGMIN.md)


**AS Level 1 (SSA):**
```text
%dstVal, %dstIdx = pto.tpartargmin %src0Val, %src1Val, %src0Idx, %src1Idx : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> (!pto.tile<...>, !pto.tile<...>)
```

**AS Level 2 (DPS):**
```text
pto.tpartargmin ins(%src0Val, %src1Val, %src0Idx, %src1Idx : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dstVal, %dstIdx : !pto.tile_buf<...>, !pto.tile_buf<...>)
```

---

### TGATHERB

For detailed instruction documentation, see [isa/TGATHERB](../isa/TGATHERB.md)


**AS Level 1 (SSA):**
```text
%dst = pto.tgatherb %src, %offsets : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS):**
```text
pto.tgatherb ins(%src, %offsets : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSCATTER

For detailed instruction documentation, see [isa/TSCATTER](../isa/TSCATTER.md)


**AS Level 1 (SSA):**
```text
%dst = pto.tscatter %src, %idx : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS):**
```text
pto.tscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TQUANT

For detailed instruction documentation, see [isa/TQUANT](../isa/TQUANT.md)


**AS Level 1 (SSA):**
```text
%dst = pto.tquant %src, %qp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS):**
```text
pto.tquant ins(%src, %qp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---
