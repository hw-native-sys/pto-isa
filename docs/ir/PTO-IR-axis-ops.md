# Axis Reduce / Expand

This document describes row/column reduction and broadcast operations.

**Total Operations:** 25

---

## Operations

### TROWSUM

For detailed instruction documentation, see [isa/TROWSUM](../isa/TROWSUM.md)


**IR Level 1 (SSA):**
```text
%dst = pto.trowsum %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.trowsum ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLSUM

For detailed instruction documentation, see [isa/TCOLSUM](../isa/TCOLSUM.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tcolsum %src : !pto.tile<...> -> !pto.tile<...>
%dst = pto.tcolsum %src, %tmp {isBinary = false} : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcolsum ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
pto.tcolsum ins(%src, %tmp {isBinary = false} : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLPROD

For detailed instruction documentation, see [isa/TCOLPROD](../isa/TCOLPROD.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tcolprod %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcolprod ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLMAX

For detailed instruction documentation, see [isa/TCOLMAX](../isa/TCOLMAX.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tcolmax %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcolmax ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWMAX

For detailed instruction documentation, see [isa/TROWMAX](../isa/TROWMAX.md)


**IR Level 1 (SSA):**
```text
%dst = pto.trowmax %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.trowmax ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWMIN

For detailed instruction documentation, see [isa/TROWMIN](../isa/TROWMIN.md)


**IR Level 1 (SSA):**
```text
%dst = pto.trowmin %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.trowmin ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPAND

For detailed instruction documentation, see [isa/TROWEXPAND](../isa/TROWEXPAND.md)


**IR Level 1 (SSA):**
```text
%dst = pto.trowexpand %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.trowexpand ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDDIV

For detailed instruction documentation, see [isa/TROWEXPANDDIV](../isa/TROWEXPANDDIV.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tcolexpanddiv %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcolexpanddiv ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDMUL

For detailed instruction documentation, see [isa/TROWEXPANDMUL](../isa/TROWEXPANDMUL.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tcolexpandmul %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcolexpandmul ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDSUB

For detailed instruction documentation, see [isa/TROWEXPANDSUB](../isa/TROWEXPANDSUB.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tcolexpandsub %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcolexpandsub ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDADD

For detailed instruction documentation, see [isa/TROWEXPANDADD](../isa/TROWEXPANDADD.md)


**IR Level 1 (SSA):**
```text
%dst = pto.trowexpandadd %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.trowexpandadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDMAX

For detailed instruction documentation, see [isa/TROWEXPANDMAX](../isa/TROWEXPANDMAX.md)


**IR Level 1 (SSA):**
```text
%dst = pto.trowexpandmax %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.trowexpandmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDMIN

For detailed instruction documentation, see [isa/TROWEXPANDMIN](../isa/TROWEXPANDMIN.md)


**IR Level 1 (SSA):**
```text
%dst = pto.trowexpandmin %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.trowexpandmin ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDEXPDIF

For detailed instruction documentation, see [isa/TROWEXPANDEXPDIF](../isa/TROWEXPANDEXPDIF.md)


**IR Level 1 (SSA):**
```text
%dst = pto.trowexpandexpdif %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.trowexpandexpdif ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLMIN

For detailed instruction documentation, see [isa/TCOLMIN](../isa/TCOLMIN.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tcolmin %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcolmin ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPAND

For detailed instruction documentation, see [isa/TCOLEXPAND](../isa/TCOLEXPAND.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tcolexpand %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcolexpand ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDDIV

For detailed instruction documentation, see [isa/TCOLEXPANDDIV](../isa/TCOLEXPANDDIV.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tcolexpanddiv %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcolexpanddiv ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDMUL

For detailed instruction documentation, see [isa/TCOLEXPANDMUL](../isa/TCOLEXPANDMUL.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tcolexpandmul %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcolexpandmul ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDADD

For detailed instruction documentation, see [isa/TCOLEXPANDADD](../isa/TCOLEXPANDADD.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tcolexpandadd %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcolexpandadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDMAX

For detailed instruction documentation, see [isa/TCOLEXPANDMAX](../isa/TCOLEXPANDMAX.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tcolexpandmax %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcolexpandmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDMIN

For detailed instruction documentation, see [isa/TCOLEXPANDMIN](../isa/TCOLEXPANDMIN.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tcolexpandmin %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcolexpandmin ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDSUB

For detailed instruction documentation, see [isa/TCOLEXPANDSUB](../isa/TCOLEXPANDSUB.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tcolexpandsub %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcolexpandsub ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDEXPDIF

For detailed instruction documentation, see [isa/TCOLEXPANDEXPDIF](../isa/TCOLEXPANDEXPDIF.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tcolexpandexpdif %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcolexpandexpdif ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWPROD

For detailed instruction documentation, see [isa/TROWPROD](../isa/TROWPROD.md)

**IR Level 1 (SSA):**
```text
%dst = pto.trowprod %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.trowprod ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---


