# 复杂操作

本文档描述复杂操作，包括排序、聚集、量化和随机数生成。

**操作总数：** 14

---

## 操作

### TPRINT

该指令的详细介绍请见[isa/TPRINT](../isa/TPRINT_zh.md)

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

该指令的详细介绍请见[isa/TMRGSORT](../isa/TMRGSORT_zh.md)

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

该指令的详细介绍请见[isa/TSORT32](../isa/TSORT32_zh.md)

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

该指令的详细介绍请见[isa/TGATHER](../isa/TGATHER_zh.md)

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

该指令的详细介绍请见[isa/TCI](../isa/TCI_zh.md)

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

该指令的详细介绍请见[isa/TTRI](../isa/TTRI_zh.md)

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

该指令的详细介绍请见[isa/TRANDOM](../isa/TRANDOM_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tci %key, %counter {rounds = 10} : -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.trandom ins(%key, %counter {rounds = 10} : dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TRANDOM

该指令的详细介绍请见[isa/TRANDOM](../isa/TRANDOM_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.trandom %key, %counter : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.trandom ins(%key, %counter : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TPARTADD

该指令的详细介绍请见[isa/TPARTADD](../isa/TPARTADD_zh.md)

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

该指令的详细介绍请见[isa/TPARTMUL](../isa/TPARTMUL_zh.md)

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

该指令的详细介绍请见[isa/TPARTMAX](../isa/TPARTMAX_zh.md)

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

该指令的详细介绍请见[isa/TPARTMIN](../isa/TPARTMIN_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tpartmin %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tpartmin ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TGATHERB

该指令的详细介绍请见[isa/TGATHERB](../isa/TGATHERB_zh.md)

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

该指令的详细介绍请见[isa/TSCATTER](../isa/TSCATTER_zh.md)

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

该指令的详细介绍请见[isa/TQUANT](../isa/TQUANT_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tquant %src, %qp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tquant ins(%src, %qp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---
