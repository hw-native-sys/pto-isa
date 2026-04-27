# 轴归约/扩展操作

本文档描述行/列归约和广播操作。

**操作总数：** 23

---

## 操作

### TROWSUM

该指令的详细介绍请见[isa/TROWSUM](../isa/tile/ops/reduce-and-expand/trowsum_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.trowsum %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.trowsum ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLSUM

该指令的详细介绍请见[isa/TCOLSUM](../isa/tile/ops/reduce-and-expand/tcolsum_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tcolsum %src : !pto.tile<...> -> !pto.tile<...>
%dst = pto.tcolsum %src, %tmp {isBinary = false} : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tcolsum ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
pto.tcolsum ins(%src, %tmp {isBinary = false} : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLPROD

该指令的详细介绍请见[isa/TCOLPROD](../isa/tile/ops/reduce-and-expand/tcolprod_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tcolprod %src : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tcolprod ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLMAX

该指令的详细介绍请见[isa/TCOLMAX](../isa/tile/ops/reduce-and-expand/tcolmax_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tcolmax %src : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tcolmax ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWMAX

该指令的详细介绍请见[isa/TROWMAX](../isa/tile/ops/reduce-and-expand/trowmax_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.trowmax %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.trowmax ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWMIN

该指令的详细介绍请见[isa/TROWMIN](../isa/tile/ops/reduce-and-expand/trowmin_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.trowmin %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.trowmin ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPAND

该指令的详细介绍请见[isa/TROWEXPAND](../isa/tile/ops/reduce-and-expand/trowexpand_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.trowexpand %src : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.trowexpand ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDDIV

该指令的详细介绍请见[isa/TROWEXPANDDIV](../isa/tile/ops/reduce-and-expand/trowexpanddiv_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tcolexpanddiv %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tcolexpanddiv ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDMUL

该指令的详细介绍请见[isa/TROWEXPANDMUL](../isa/tile/ops/reduce-and-expand/trowexpandmul_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tcolexpandmul %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tcolexpandmul ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDSUB

该指令的详细介绍请见[isa/TROWEXPANDSUB](../isa/tile/ops/reduce-and-expand/trowexpandsub_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tcolexpandsub %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tcolexpandsub ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDADD

该指令的详细介绍请见[isa/TROWEXPANDADD](../isa/tile/ops/reduce-and-expand/trowexpandadd_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.trowexpandadd %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.trowexpandadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDMAX

该指令的详细介绍请见[isa/TROWEXPANDMAX](../isa/tile/ops/reduce-and-expand/trowexpandmax_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.trowexpandmax %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.trowexpandmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDMIN

该指令的详细介绍请见[isa/TROWEXPANDMIN](../isa/tile/ops/reduce-and-expand/trowexpandmin_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.trowexpandmin %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.trowexpandmin ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDEXPDIF

该指令的详细介绍请见[isa/TROWEXPANDEXPDIF](../isa/tile/ops/reduce-and-expand/trowexpandexpdif_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.trowexpandexpdif %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.trowexpandexpdif ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLMIN

该指令的详细介绍请见[isa/TCOLMIN](../isa/tile/ops/reduce-and-expand/tcolmin_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tcolmin %src : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tcolmin ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPAND

该指令的详细介绍请见[isa/TCOLEXPAND](../isa/tile/ops/reduce-and-expand/tcolexpand_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tcolexpand %src : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tcolexpand ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDDIV

该指令的详细介绍请见[isa/TCOLEXPANDDIV](../isa/tile/ops/reduce-and-expand/tcolexpanddiv_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tcolexpanddiv %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tcolexpanddiv ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDMUL

该指令的详细介绍请见[isa/TCOLEXPANDMUL](../isa/tile/ops/reduce-and-expand/tcolexpandmul_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tcolexpandmul %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tcolexpandmul ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDADD

该指令的详细介绍请见[isa/TCOLEXPANDADD](../isa/tile/ops/reduce-and-expand/tcolexpandadd_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tcolexpandadd %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tcolexpandadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDMAX

该指令的详细介绍请见[isa/TCOLEXPANDMAX](../isa/tile/ops/reduce-and-expand/tcolexpandmax_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tcolexpandmax %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tcolexpandmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDMIN

该指令的详细介绍请见[isa/TCOLEXPANDMIN](../isa/tile/ops/reduce-and-expand/tcolexpandmin_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tcolexpandmin %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tcolexpandmin ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDSUB

该指令的详细介绍请见[isa/TCOLEXPANDSUB](../isa/tile/ops/reduce-and-expand/tcolexpandsub_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tcolexpandsub %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tcolexpandsub ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDEXPDIF

该指令的详细介绍请见[isa/TCOLEXPANDEXPDIF](../isa/tile/ops/reduce-and-expand/tcolexpandexpdif_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tcolexpandexpdif %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tcolexpandexpdif ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

### TROWPROD

该指令的详细介绍请见[isa/TROWPROD](../isa/tile/ops/reduce-and-expand/trowprod_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.trowprod %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.trowprod ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---
