# 逐元素操作（Tile-Tile）

本文档描述两个 tile 之间的逐元素操作。

**操作总数：** 28

---

## 操作

### TADD

该指令的详细介绍请见[isa/TADD](../isa/TADD_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tadd %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TABS

该指令的详细介绍请见[isa/TABS](../isa/TABS_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tabs %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tabs ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TAND

该指令的详细介绍请见[isa/TAND](../isa/TAND_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tand %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tand ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TOR

该指令的详细介绍请见[isa/TOR](../isa/TOR_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tor %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tor ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSUB

该指令的详细介绍请见[isa/TSUB](../isa/TSUB_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tsub %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tsub ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TMUL

该指令的详细介绍请见[isa/TMUL](../isa/TMUL_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tmul %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tmul ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TMIN

该指令的详细介绍请见[isa/TMIN](../isa/TMIN_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tmin %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tmin ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TMAX

该指令的详细介绍请见[isa/TMAX](../isa/TMAX_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tmax %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCMP

该指令的详细介绍请见[isa/TCMP](../isa/TCMP_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tcmp %src0, %src1{cmpMode = #pto<cmp xx>}: (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcmp ins(%src0, %src1{cmpMode = #pto<cmp xx>}: !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TDIV

该指令的详细介绍请见[isa/TDIV](../isa/TDIV_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tdiv %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tdiv ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSHL

该指令的详细介绍请见[isa/TSHL](../isa/TSHL_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tshl %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tshl ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSHR

该指令的详细介绍请见[isa/TSHR](../isa/TSHR_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tshr %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tshr ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TXOR

该指令的详细介绍请见[isa/TXOR](../isa/TXOR_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.txor %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.txor ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TLOG

该指令的详细介绍请见[isa/TLOG](../isa/TLOG_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tlog %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tlog ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TRECIP

该指令的详细介绍请见[isa/TRECIP](../isa/TRECIP_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.trecip %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.trecip ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TPRELU

该指令的详细介绍请见[isa/TPRELU](../isa/TPRELU_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tprelu %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tprelu ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TADDC

该指令的详细介绍请见[isa/TADDC](../isa/TADDC_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.taddc %src0, %src1, %src2 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.taddc ins(%src0, %src1, %src2 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSUBC

该指令的详细介绍请见[isa/TSUBC](../isa/TSUBC_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tsubc %src0, %src1, %src2 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tsubc ins(%src0, %src1, %src2 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCVT

该指令的详细介绍请见[isa/TCVT](../isa/TCVT_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tcvt %src{rmode = #pto<round_mode xx>}: !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcvt ins(%src{rmode = #pto<round_mode xx>}: !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSEL

该指令的详细介绍请见[isa/TSEL](../isa/TSEL_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tsel %mask, %src0, %src1 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tsel ins(%mask, %src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TRSQRT

该指令的详细介绍请见[isa/TRSQRT](../isa/TRSQRT_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.trsqrt %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.trsqrt ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSQRT

该指令的详细介绍请见[isa/TSQRT](../isa/TSQRT_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tsqrt %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tsqrt ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TEXP

该指令的详细介绍请见[isa/TEXP](../isa/TEXP_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.texp %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.texp ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TNOT

该指令的详细介绍请见[isa/TNOT](../isa/TNOT_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tnot %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tnot ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TRELU

该指令的详细介绍请见[isa/TRELU](../isa/TRELU_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.trelu %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.trelu ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TNEG

该指令的详细介绍请见[isa/TNEG](../isa/TNEG_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tneg %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tneg ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TREM

该指令的详细介绍请见[isa/TREM](../isa/TREM_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.trem %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.trem ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TFMOD

该指令的详细介绍请见[isa/TFMOD](../isa/TFMOD_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tfmod %src0, %src1 : !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tfmod ins(%src0, %src1 : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
