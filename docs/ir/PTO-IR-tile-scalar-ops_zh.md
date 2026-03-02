# Tile-标量 / Tile-立即数

本文档描述 tile 与标量值或立即常量之间的操作。

**操作总数：** 19

---

## 操作

### TEXPANDS

该指令的详细介绍请见[isa/TEXPANDS](../isa/TEXPANDS_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.texpands %scalar : dtype -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.texpands ins(%scalar : dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TCMPS

该指令的详细介绍请见[isa/TCMPS](../isa/TCMPS_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tcmps %src, %scalar {cmpMode = #pto<cmp xx>} : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcmps ins(%src, %scalar{cmpMode = #pto<cmp xx>}: !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TSELS

该指令的详细介绍请见[isa/TSELS](../isa/TSELS_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tsels %src0, %src1, %scalar : (!pto.tile<...>, !pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tsels ins(%src0, %src1, %scalar : !pto.tile_buf<...>, !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TMINS

该指令的详细介绍请见[isa/TMINS](../isa/TMINS_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tmins %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tmins ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TADDS

该指令的详细介绍请见[isa/TADDS](../isa/TADDS_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tadds %src, %scalar : (!pto.tile<...>,dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tadds ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TSUBS

该指令的详细介绍请见[isa/TSUBS](../isa/TSUBS_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tsubs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tsubs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TDIVS

该指令的详细介绍请见[isa/TDIVS](../isa/TDIVS_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tdivs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
%dst = pto.tdivs %scalar, %src : (dtype, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tdivs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
pto.tdivs ins(%scalar, %src : dtype, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TMULS

该指令的详细介绍请见[isa/TMULS](../isa/TMULS_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tmuls %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tmuls ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TFMODS

该指令的详细介绍请见[isa/TFMODS](../isa/TFMODS_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tfmods %src, %scalar : !pto.tile<...>, f32
```

**IR Level 2 (DPS)：**
```text
pto.tfmods ins(%src, %scalar : !pto.tile_buf<...>, f32) outs(%dst : !pto.tile_buf<...>)
```

---

### TREMS

该指令的详细介绍请见[isa/TREMS](../isa/TREMS_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.trems %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.trems ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TMAXS

该指令的详细介绍请见[isa/TMAXS](../isa/TMAXS_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tmaxs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tmaxs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TANDS

该指令的详细介绍请见[isa/TANDS](../isa/TANDS_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tands %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tands ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TORS

该指令的详细介绍请见[isa/TORS](../isa/TORS_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tors %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tors ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TSHLS

该指令的详细介绍请见[isa/TSHLS](../isa/TSHLS_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tshls %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tshls ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TSHRS

该指令的详细介绍请见[isa/TSHRS](../isa/TSHRS_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tshrs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tshrs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TXORS

该指令的详细介绍请见[isa/TXORS](../isa/TXORS_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.txors %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.txors ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TLRELU

该指令的详细介绍请见[isa/TLRELU](../isa/TLRELU_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tlrelu %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tlrelu ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TADDSC

该指令的详细介绍请见[isa/TADDSC](../isa/TADDSC_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.taddsc %src0, %scalar, %src1 : (!pto.tile<...>, dtype, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.taddsc ins(%src0, %scalar, %src1 : !pto.tile_buf<...>, dtype, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSUBSC

该指令的详细介绍请见[isa/TSUBSC](../isa/TSUBSC_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tsubsc %src0, %scalar, %src1 : (!pto.tile<...>, dtype, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tsubsc ins(%src0, %scalar, %src1 : !pto.tile_buf<...>, dtype, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
