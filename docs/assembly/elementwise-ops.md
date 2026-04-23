# Elementwise (Tile-Tile)

This document describes element-wise operations between two tiles.

**Total Operations:** 28

---

## Operations

### TADD

For detailed instruction documentation, see [isa/TADD](../isa/TADD.md)

**AS Level 1 (SSA):**

```text
%dst = pto.tadd %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.tadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TABS

For detailed instruction documentation, see [isa/TABS](../isa/TABS.md)

**AS Level 1 (SSA):**

```text
%dst = pto.tabs %src : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.tabs ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TAND

For detailed instruction documentation, see [isa/TAND](../isa/TAND.md)

**AS Level 1 (SSA):**

```text
%dst = pto.tand %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.tand ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TOR

For detailed instruction documentation, see [isa/TOR](../isa/TOR.md)

**AS Level 1 (SSA):**

```text
%dst = pto.tor %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.tor ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSUB

For detailed instruction documentation, see [isa/TSUB](../isa/TSUB.md)

**AS Level 1 (SSA):**

```text
%dst = pto.tsub %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.tsub ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TMUL

For detailed instruction documentation, see [isa/TMUL](../isa/TMUL.md)

**AS Level 1 (SSA):**

```text
%dst = pto.tmul %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.tmul ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TMIN

For detailed instruction documentation, see [isa/TMIN](../isa/TMIN.md)

**AS Level 1 (SSA):**

```text
%dst = pto.tmin %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.tmin ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TMAX

For detailed instruction documentation, see [isa/TMAX](../isa/TMAX.md)

**AS Level 1 (SSA):**

```text
%dst = pto.tmax %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.tmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCMP

For detailed instruction documentation, see [isa/TCMP](../isa/TCMP.md)

**AS Level 1 (SSA):**

```text
%dst = pto.tcmp %src0, %src1{cmpMode = #pto<cmp xx>}: (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.tcmp ins(%src0, %src1{cmpMode = #pto<cmp xx>}: !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TDIV

For detailed instruction documentation, see [isa/TDIV](../isa/TDIV.md)

**AS Level 1 (SSA):**

```text
%dst = pto.tdiv %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.tdiv ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSHL

For detailed instruction documentation, see [isa/TSHL](../isa/TSHL.md)

**AS Level 1 (SSA):**

```text
%dst = pto.tshl %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.tshl ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSHR

For detailed instruction documentation, see [isa/TSHR](../isa/TSHR.md)

**AS Level 1 (SSA):**

```text
%dst = pto.tshr %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.tshr ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TXOR

For detailed instruction documentation, see [isa/TXOR](../isa/TXOR.md)

**AS Level 1 (SSA):**

```text
%dst = pto.txor %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.txor ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TLOG

For detailed instruction documentation, see [isa/TLOG](../isa/TLOG.md)

**AS Level 1 (SSA):**

```text
%dst = pto.tlog %src : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.tlog ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TRECIP

For detailed instruction documentation, see [isa/TRECIP](../isa/TRECIP.md)

**AS Level 1 (SSA):**

```text
%dst = pto.trecip %src : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.trecip ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TPRELU

For detailed instruction documentation, see [isa/TPRELU](../isa/TPRELU.md)

**AS Level 1 (SSA):**

```text
%dst = pto.tprelu %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.tprelu ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TADDC

For detailed instruction documentation, see [isa/TADDC](../isa/TADDC.md)

**AS Level 1 (SSA):**

```text
%dst = pto.taddc %src0, %src1, %src2 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.taddc ins(%src0, %src1, %src2 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSUBC

For detailed instruction documentation, see [isa/TSUBC](../isa/TSUBC.md)

**AS Level 1 (SSA):**

```text
%dst = pto.tsubc %src0, %src1, %src2 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.tsubc ins(%src0, %src1, %src2 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCVT

For detailed instruction documentation, see [isa/TCVT](../isa/TCVT.md)

**AS Level 1 (SSA):**

```text
%dst = pto.tcvt %src{rmode = #pto<round_mode xx>}: !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.tcvt ins(%src{rmode = #pto<round_mode xx>}: !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSEL

For detailed instruction documentation, see [isa/TSEL](../isa/TSEL.md)

**AS Level 1 (SSA):**

```text
%dst = pto.tsel %mask, %src0, %src1 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.tsel ins(%mask, %src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TRSQRT

For detailed instruction documentation, see [isa/TRSQRT](../isa/TRSQRT.md)

**AS Level 1 (SSA):**

```text
%dst = pto.trsqrt %src : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.trsqrt ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSQRT

For detailed instruction documentation, see [isa/TSQRT](../isa/TSQRT.md)

**AS Level 1 (SSA):**

```text
%dst = pto.tsqrt %src : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.tsqrt ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TEXP

For detailed instruction documentation, see [isa/TEXP](../isa/TEXP.md)

**AS Level 1 (SSA):**

```text
%dst = pto.texp %src : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.texp ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TNOT

For detailed instruction documentation, see [isa/TNOT](../isa/TNOT.md)

**AS Level 1 (SSA):**

```text
%dst = pto.tnot %src : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.tnot ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TRELU

For detailed instruction documentation, see [isa/TRELU](../isa/TRELU.md)

**AS Level 1 (SSA):**

```text
%dst = pto.trelu %src : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.trelu ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TNEG

For detailed instruction documentation, see [isa/TNEG](../isa/TNEG.md)

**AS Level 1 (SSA):**

```text
%dst = pto.tneg %src : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.tneg ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TREM

For detailed instruction documentation, see [isa/TREM](../isa/TREM.md)

**AS Level 1 (SSA):**

```text
%dst = pto.trem %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.trem ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TFMOD

For detailed instruction documentation, see [isa/TFMOD](../isa/TFMOD.md)

**AS Level 1 (SSA):**

```text
%dst = pto.tfmod %src0, %src1 : !pto.tile<...>
```

**AS Level 2 (DPS):**

```text
pto.tfmod ins(%src0, %src1 : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TPOW

For detailed instruction documentation, see [isa/TPOW](../isa/TPOW.md)

**AS Level 1 (SSA):**
```text
%dst = pto.tpow %base, %exp, %tmp : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS):**
```text
pto.tpow ins(%base, %exp, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---
