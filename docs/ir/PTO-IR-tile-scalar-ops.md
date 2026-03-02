# Tile-Scalar / Tile-Immediate

This document describes operations between tiles and scalar values or immediate constants.

**Total Operations:** 19

---

## Operations

### TEXPANDS

For detailed instruction documentation, see [isa/TEXPANDS](../isa/TEXPANDS.md)


**IR Level 1 (SSA):**
```text
%dst = pto.texpands %scalar : dtype -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.texpands ins(%scalar : dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TCMPS

For detailed instruction documentation, see [isa/TCMPS](../isa/TCMPS.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tcmps %src, %scalar {cmpMode = #pto<cmp xx>} : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcmps ins(%src, %scalar{cmpMode = #pto<cmp xx>}: !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TSELS

For detailed instruction documentation, see [isa/TSELS](../isa/TSELS.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tsels %src0, %src1, %scalar : (!pto.tile<...>, !pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tsels ins(%src0, %src1, %scalar : !pto.tile_buf<...>, !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TMINS

For detailed instruction documentation, see [isa/TMINS](../isa/TMINS.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tmins %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tmins ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TADDS

For detailed instruction documentation, see [isa/TADDS](../isa/TADDS.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tadds %src, %scalar : (!pto.tile<...>,dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tadds ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TSUBS

For detailed instruction documentation, see [isa/TSUBS](../isa/TSUBS.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tsubs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tsubs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TDIVS

For detailed instruction documentation, see [isa/TDIVS](../isa/TDIVS.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tdivs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
%dst = pto.tdivs %scalar, %src : (dtype, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tdivs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
pto.tdivs ins(%scalar, %src : dtype, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TMULS

For detailed instruction documentation, see [isa/TMULS](../isa/TMULS.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tmuls %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tmuls ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TFMODS

For detailed instruction documentation, see [isa/TFMODS](../isa/TFMODS.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tfmods %src, %scalar : !pto.tile<...>, f32
```

**IR Level 2 (DPS):**
```text
pto.tfmods ins(%src, %scalar : !pto.tile_buf<...>, f32) outs(%dst : !pto.tile_buf<...>)
```

---

### TREMS

For detailed instruction documentation, see [isa/TREMS](../isa/TREMS.md)


**IR Level 1 (SSA):**
```text
%dst = pto.trems %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.trems ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TMAXS

For detailed instruction documentation, see [isa/TMAXS](../isa/TMAXS.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tmaxs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tmaxs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TANDS

For detailed instruction documentation, see [isa/TANDS](../isa/TANDS.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tands %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tands ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TORS

For detailed instruction documentation, see [isa/TORS](../isa/TORS.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tors %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tors ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TSHLS

For detailed instruction documentation, see [isa/TSHLS](../isa/TSHLS.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tshls %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tshls ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TSHRS

For detailed instruction documentation, see [isa/TSHRS](../isa/TSHRS.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tshrs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tshrs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TXORS

For detailed instruction documentation, see [isa/TXORS](../isa/TXORS.md)


**IR Level 1 (SSA):**
```text
%dst = pto.txors %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.txors ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TLRELU

For detailed instruction documentation, see [isa/TLRELU](../isa/TLRELU.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tlrelu %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tlrelu ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TADDSC

For detailed instruction documentation, see [isa/TADDSC](../isa/TADDSC.md)


**IR Level 1 (SSA):**
```text
%dst = pto.taddsc %src0, %scalar, %src1 : (!pto.tile<...>, dtype, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.taddsc ins(%src0, %scalar, %src1 : !pto.tile_buf<...>, dtype, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSUBSC

For detailed instruction documentation, see [isa/TSUBSC](../isa/TSUBSC.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tsubsc %src0, %scalar, %src1 : (!pto.tile<...>, dtype, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tsubsc ins(%src0, %scalar, %src1 : !pto.tile_buf<...>, dtype, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

