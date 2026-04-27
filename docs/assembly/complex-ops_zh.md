# 复杂操作

本文档描述复杂操作，包括排序、聚集、量化和随机数生成。

**操作总数：** 16

---

## 操作

### TPRINT

该指令的详细介绍请见[isa/TPRINT](../isa/tile/ops/irregular-and-complex/tprint_zh.md)

**AS Level 1 (SSA)：**

```text
pto.tprint %src : !pto.tile<...> | !pto.partition_tensor_view<MxNxdtype> -> ()
```

**AS Level 2 (DPS)：**

```text
pto.tprint ins(%src : !pto.tile_buf<...> | !pto.partition_tensor_view<MxNxdtype>)
```

---

### TMRGSORT

该指令的详细介绍请见[isa/TMRGSORT](../isa/tile/ops/irregular-and-complex/tmrgsort_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tmrgsort %src, %blockLen : (!pto.tile<...>, dtype) -> !pto.tile<...>
%dst, %executed = pto.tmrgsort %src0, %src1, %src2, %src3 {exhausted = false}
 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> (!pto.tile<...>, vector<4xi16>)
```

**AS Level 2 (DPS)：**

```text
pto.tmrgsort ins(%src, %blockLen : !pto.tile_buf<...>, dtype)  outs(%dst : !pto.tile_buf<...>)
pto.tmrgsort ins(%src0, %src1, %src2, %src3 {exhausted = false} : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
outs(%dst, %executed : !pto.tile_buf<...>, vector<4xi16>)
```

---

### TSORT32

该指令的详细介绍请见[isa/TSORT32](../isa/tile/ops/irregular-and-complex/tsort32_zh.md)

**AS Level 1 (SSA)：**

```text
%dst, %idx = pto.tsort32 %src : !pto.tile<...> -> (!pto.tile<...>, !pto.tile<...>)
```

**AS Level 2 (DPS)：**

```text
pto.tsort32 ins(%src : !pto.tile_buf<...>) outs(%dst, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>)
```

---

### TGATHER

该指令的详细介绍请见[isa/TGATHER](../isa/tile/ops/irregular-and-complex/tgather_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tgather %src, %indices : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
%dst = pto.tgather %src {maskPattern = #pto.mask_pattern<P0101>}: !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tgather ins(%src, %indices : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
pto.tgather ins(%src, {maskPattern = #pto.mask_pattern<P0101>} : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCI

该指令的详细介绍请见[isa/TCI](../isa/tile/ops/irregular-and-complex/tci_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tci %scalar {descending = false} : dtype -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tci ins(%scalar {descending = false} : dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TTRI

该指令的详细介绍请见[isa/TTRI](../isa/tile/ops/irregular-and-complex/ttri_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.ttri %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.ttri ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TRANDOM

该指令的详细介绍请见[isa/TRANDOM](../isa/tile/irregular-and-complex_zh.md)


**AS Level 1 (SSA)：**
```text
%dst = pto.trandom %key, %counter {rounds = 10} : -> !pto.tile<...>
```

**AS Level 2 (DPS)：**
```text
pto.trandom ins(%key, %counter {rounds = 10} : dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TPARTADD

该指令的详细介绍请见[isa/TPARTADD](../isa/tile/ops/irregular-and-complex/tpartadd_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tpartadd %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tpartadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TPARTMUL

该指令的详细介绍请见[isa/TPARTMUL](../isa/tile/ops/irregular-and-complex/tpartmul_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tpartmul %src0, %src1 : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tpartmul ins(%src0, %src1 : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TPARTMAX

该指令的详细介绍请见[isa/TPARTMAX](../isa/tile/ops/irregular-and-complex/tpartmax_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tpartmax %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tpartmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TPARTMIN

该指令的详细介绍请见[isa/TPARTMIN](../isa/tile/ops/irregular-and-complex/tpartmin_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tpartmin %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tpartmin ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TPARTARGMAX

该指令的详细介绍请见[isa/TPARTARGMAX](../isa/TPARTARGMAX_zh.md)


**AS Level 1 (SSA)：**
```text
%dstVal, %dstIdx = pto.tpartargmax %src0Val, %src1Val, %src0Idx, %src1Idx : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> (!pto.tile<...>, !pto.tile<...>)
```

**AS Level 2 (DPS)：**
```text
pto.tpartargmax ins(%src0Val, %src1Val, %src0Idx, %src1Idx : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dstVal, %dstIdx : !pto.tile_buf<...>, !pto.tile_buf<...>)
```

---

### TPARTARGMIN

该指令的详细介绍请见[isa/TPARTARGMIN](../isa/TPARTARGMIN_zh.md)


**AS Level 1 (SSA)：**
```text
%dstVal, %dstIdx = pto.tpartargmin %src0Val, %src1Val, %src0Idx, %src1Idx : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> (!pto.tile<...>, !pto.tile<...>)
```

**AS Level 2 (DPS)：**
```text
pto.tpartargmin ins(%src0Val, %src1Val, %src0Idx, %src1Idx : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dstVal, %dstIdx : !pto.tile_buf<...>, !pto.tile_buf<...>)
```

---

### TGATHERB

该指令的详细介绍请见[isa/TGATHERB](../isa/tile/ops/irregular-and-complex/tgatherb_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tgatherb %src, %offsets : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tgatherb ins(%src, %offsets : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSCATTER

该指令的详细介绍请见[isa/TSCATTER](../isa/tile/ops/irregular-and-complex/tscatter_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tscatter %src, %idx : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TQUANT

该指令的详细介绍请见[isa/TQUANT](../isa/tile/ops/irregular-and-complex/tquant_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tquant %src, %qp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tquant ins(%src, %qp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---
