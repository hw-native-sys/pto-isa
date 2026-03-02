# 数据移动/布局

本文档描述数据移动和布局转换操作。

**操作总数：** 12

---

## 操作

### TEXTRACT

该指令的详细介绍请见[isa/TEXTRACT](../isa/TEXTRACT_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.textract %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.textract ins(%src, %idxrow, %idxcol : !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TEXTRACT_FP

该指令的详细介绍请见[isa/TEXTRACT_FP](../isa/TEXTRACT_FP_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.textract_fp %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.textract_fp ins(%src, %idxrow, %idxcol : !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TIMG2COL

**IR Level 1 (SSA)：**
```text
%dst = pto.timg2col %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.timg2col ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TINSERT

该指令的详细介绍请见[isa/TINSERT](../isa/TINSERT_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tinsert %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tinsert ins(%src[%r0, %r1] : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TINSERT_FP

该指令的详细介绍请见[isa/TINSERT_FP](../isa/TINSERT_FP_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tinsert_fp %src, %fp, %idxrow, %idxcol : (!pto.tile<...>, !pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tinsert_fp ins(%src, %fp, %idxrow, %idxcol : !pto.tile_buf<...>, !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TFILLPAD

该指令的详细介绍请见[isa/TFILLPAD](../isa/TFILLPAD_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tfillpad %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tfillpad ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TFILLPAD_INPLACE

该指令的详细介绍请见[isa/TFILLPAD_INPLACE](../isa/TFILLPAD_INPLACE_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tfillpad_inplace %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tfillpad_inplace ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TFILLPAD_EXPAND

该指令的详细介绍请见[isa/TFILLPAD_EXPAND](../isa/TFILLPAD_EXPAND_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tfillpad_expand %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tfillpad_expand ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TMOV

该指令的详细介绍请见[isa/TMOV](../isa/TMOV_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tmov.s2d %src  : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tmov ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TMOV_FP

该指令的详细介绍请见[isa/TMOV_FP](../isa/TMOV_FP_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tmov.fp %src, %fp : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tmov.fp ins(%src, %fp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TRESHAPE

该指令的详细介绍请见[isa/TRESHAPE](../isa/TRESHAPE_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.treshape %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.treshape ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TTRANS

该指令的详细介绍请见[isa/TTRANS](../isa/TTRANS_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.ttrans %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.ttrans ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
### TIMG2COL

该指令的详细介绍请见[isa/TIMG2COL](../isa/TIMG2COL_zh.md)

**IR Level 1 (SSA)：**
```text
%dst = pto.timg2col %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.timg2col ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---


