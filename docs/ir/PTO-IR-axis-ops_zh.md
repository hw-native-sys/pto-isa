# 轴归约/扩展操作

本文档描述行/列归约和广播操作。

**操作总数：** 23

---

## 操作

### TROWSUM

该指令的详细介绍请见[isa/TROWSUM](../isa/TROWSUM_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.trowsum %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.trowsum ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLSUM

该指令的详细介绍请见[isa/TCOLSUM](../isa/TCOLSUM_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tcolsum %src : !pto.tile<...> -> !pto.tile<...>
%dst = pto.tcolsum %src, %tmp {isBinary = false} : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcolsum ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
pto.tcolsum ins(%src, %tmp {isBinary = false} : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLPROD

该指令的详细介绍请见[isa/TCOLPROD](../isa/TCOLPROD_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tcolprod %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcolprod ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLMAX

该指令的详细介绍请见[isa/TCOLMAX](../isa/TCOLMAX_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tcolmax %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcolmax ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWMAX

该指令的详细介绍请见[isa/TROWMAX](../isa/TROWMAX_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.trowmax %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.trowmax ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWMIN

该指令的详细介绍请见[isa/TROWMIN](../isa/TROWMIN_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.trowmin %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.trowmin ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPAND

该指令的详细介绍请见[isa/TROWEXPAND](../isa/TROWEXPAND_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.trowexpand %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.trowexpand ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDDIV

该指令的详细介绍请见[isa/TROWEXPANDDIV](../isa/TROWEXPANDDIV_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tcolexpanddiv %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcolexpanddiv ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDMUL

该指令的详细介绍请见[isa/TROWEXPANDMUL](../isa/TROWEXPANDMUL_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tcolexpandmul %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcolexpandmul ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDSUB

该指令的详细介绍请见[isa/TROWEXPANDSUB](../isa/TROWEXPANDSUB_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tcolexpandsub %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcolexpandsub ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDADD

该指令的详细介绍请见[isa/TROWEXPANDADD](../isa/TROWEXPANDADD_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.trowexpandadd %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.trowexpandadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDMAX

该指令的详细介绍请见[isa/TROWEXPANDMAX](../isa/TROWEXPANDMAX_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.trowexpandmax %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.trowexpandmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDMIN

该指令的详细介绍请见[isa/TROWEXPANDMIN](../isa/TROWEXPANDMIN_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.trowexpandmin %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.trowexpandmin ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDEXPDIF

该指令的详细介绍请见[isa/TROWEXPANDEXPDIF](../isa/TROWEXPANDEXPDIF_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.trowexpandexpdif %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.trowexpandexpdif ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLMIN

该指令的详细介绍请见[isa/TCOLMIN](../isa/TCOLMIN_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tcolmin %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcolmin ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPAND

该指令的详细介绍请见[isa/TCOLEXPAND](../isa/TCOLEXPAND_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tcolexpand %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcolexpand ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDDIV

该指令的详细介绍请见[isa/TCOLEXPANDDIV](../isa/TCOLEXPANDDIV_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tcolexpanddiv %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcolexpanddiv ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDMUL

该指令的详细介绍请见[isa/TCOLEXPANDMUL](../isa/TCOLEXPANDMUL_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tcolexpandmul %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcolexpandmul ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDADD

该指令的详细介绍请见[isa/TCOLEXPANDADD](../isa/TCOLEXPANDADD_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tcolexpandadd %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcolexpandadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDMAX

该指令的详细介绍请见[isa/TCOLEXPANDMAX](../isa/TCOLEXPANDMAX_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tcolexpandmax %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcolexpandmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDMIN

该指令的详细介绍请见[isa/TCOLEXPANDMIN](../isa/TCOLEXPANDMIN_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tcolexpandmin %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcolexpandmin ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDSUB

该指令的详细介绍请见[isa/TCOLEXPANDSUB](../isa/TCOLEXPANDSUB_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tcolexpandsub %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcolexpandsub ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDEXPDIF

该指令的详细介绍请见[isa/TCOLEXPANDEXPDIF](../isa/TCOLEXPANDEXPDIF_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tcolexpandexpdif %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcolexpandexpdif ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
### TROWPROD

该指令的详细介绍请见[isa/TROWPROD](../isa/TROWPROD_zh.md)

**IR Level 1 (SSA)：**
```text
%dst = pto.trowprod %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
pto.trowprod ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---


