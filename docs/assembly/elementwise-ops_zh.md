# 逐元素操作（Tile-Tile）

本文档描述两个 tile 之间的逐元素操作。

**操作总数：** 29

---

## 操作

### TADD

该指令的详细介绍请见[isa/TADD](../isa/tile/ops/elementwise-tile-tile/tadd_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tadd %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TABS

该指令的详细介绍请见[isa/TABS](../isa/tile/ops/elementwise-tile-tile/tabs_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tabs %src : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tabs ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TAND

该指令的详细介绍请见[isa/TAND](../isa/tile/ops/elementwise-tile-tile/tand_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tand %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tand ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TOR

该指令的详细介绍请见[isa/TOR](../isa/tile/ops/elementwise-tile-tile/tor_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tor %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tor ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSUB

该指令的详细介绍请见[isa/TSUB](../isa/tile/ops/elementwise-tile-tile/tsub_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tsub %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tsub ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TMUL

该指令的详细介绍请见[isa/TMUL](../isa/tile/ops/elementwise-tile-tile/tmul_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tmul %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tmul ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TMIN

该指令的详细介绍请见[isa/TMIN](../isa/tile/ops/elementwise-tile-tile/tmin_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tmin %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tmin ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TMAX

该指令的详细介绍请见[isa/TMAX](../isa/tile/ops/elementwise-tile-tile/tmax_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tmax %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCMP

该指令的详细介绍请见[isa/TCMP](../isa/tile/ops/elementwise-tile-tile/tcmp_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tcmp %src0, %src1{cmpMode = #pto<cmp xx>}: (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tcmp ins(%src0, %src1{cmpMode = #pto<cmp xx>}: !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TDIV

该指令的详细介绍请见[isa/TDIV](../isa/tile/ops/elementwise-tile-tile/tdiv_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tdiv %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tdiv ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSHL

该指令的详细介绍请见[isa/TSHL](../isa/tile/ops/elementwise-tile-tile/tshl_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tshl %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tshl ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSHR

该指令的详细介绍请见[isa/TSHR](../isa/tile/ops/elementwise-tile-tile/tshr_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tshr %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tshr ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TXOR

该指令的详细介绍请见[isa/TXOR](../isa/tile/ops/elementwise-tile-tile/txor_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.txor %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.txor ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TLOG

该指令的详细介绍请见[isa/TLOG](../isa/tile/ops/elementwise-tile-tile/tlog_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tlog %src : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tlog ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TRECIP

该指令的详细介绍请见[isa/TRECIP](../isa/tile/ops/elementwise-tile-tile/trecip_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.trecip %src : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.trecip ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TPRELU

该指令的详细介绍请见[isa/TPRELU](../isa/tile/ops/elementwise-tile-tile/tprelu_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tprelu %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tprelu ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TADDC

该指令的详细介绍请见[isa/TADDC](../isa/tile/ops/elementwise-tile-tile/taddc_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.taddc %src0, %src1, %src2 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.taddc ins(%src0, %src1, %src2 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSUBC

该指令的详细介绍请见[isa/TSUBC](../isa/tile/ops/elementwise-tile-tile/tsubc_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tsubc %src0, %src1, %src2 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tsubc ins(%src0, %src1, %src2 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCVT

该指令的详细介绍请见[isa/TCVT](../isa/tile/ops/elementwise-tile-tile/tcvt_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tcvt %src{rmode = #pto<round_mode xx>}: !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tcvt ins(%src{rmode = #pto<round_mode xx>}: !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSEL

该指令的详细介绍请见[isa/TSEL](../isa/tile/ops/elementwise-tile-tile/tsel_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tsel %mask, %src0, %src1 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tsel ins(%mask, %src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TRSQRT

该指令的详细介绍请见[isa/TRSQRT](../isa/tile/ops/elementwise-tile-tile/trsqrt_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.trsqrt %src : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.trsqrt ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSQRT

该指令的详细介绍请见[isa/TSQRT](../isa/tile/ops/elementwise-tile-tile/tsqrt_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tsqrt %src : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tsqrt ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TEXP

该指令的详细介绍请见[isa/TEXP](../isa/tile/ops/elementwise-tile-tile/texp_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.texp %src : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.texp ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TNOT

该指令的详细介绍请见[isa/TNOT](../isa/tile/ops/elementwise-tile-tile/tnot_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tnot %src : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tnot ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TRELU

该指令的详细介绍请见[isa/TRELU](../isa/tile/ops/elementwise-tile-tile/trelu_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.trelu %src : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.trelu ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TNEG

该指令的详细介绍请见[isa/TNEG](../isa/tile/ops/elementwise-tile-tile/tneg_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tneg %src : !pto.tile<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tneg ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TREM

该指令的详细介绍请见[isa/TREM](../isa/tile/ops/elementwise-tile-tile/trem_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.trem %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.trem ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TFMOD

该指令的详细介绍请见[isa/TFMOD](../isa/tile/ops/elementwise-tile-tile/tfmod_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tfmod %src0, %src1 : !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tfmod ins(%src0, %src1 : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TPOW

该指令的详细介绍请见[isa/TPOW](../isa/tile/ops/elementwise-tile-tile/tpow_zh.md)


**AS Level 1 (SSA)：**
```text
%dst = pto.tpow %base, %exp, %tmp : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**AS Level 2 (DPS)：**
```text
pto.tpow ins(%base, %exp, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---
