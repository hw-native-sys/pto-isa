# 矩阵乘法

本文档描述矩阵乘法和矩阵-向量操作。

**操作总数：** 8

---

## 操作

### TGEMV_MX

该指令的详细介绍请见[isa/TGEMV_MX](../isa/TGEMV_MX_zh.md)


**IR Level 1 (SSA)：**
```text
%acc = pto.tgemv.mx %a, %a_scale, %b, %b_scale : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tgemv.mx ins(%a, %a_scale, %b, %b_scale : (!pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)) outs(%acc : !pto.tile_buf<...>)
```

---

### TMATMUL_MX

该指令的详细介绍请见[isa/TMATMUL_MX](../isa/TMATMUL_MX_zh.md)


**IR Level 1 (SSA)：**
```text
%c = pto.tmatmul.mx %a, %a_scale, %b, %b_scale : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>)
-> !pto.tile<...>
%c_out = pto.tmatmul.mx.acc %c_in, %a, %a_scale, %b, %b_scale : (!pto.tile<...>, !pto.tile<...>,
!pto.tile<...>, !pto.tile<...>, !pto.tile<...>)  -> !pto.tile<...>
%c = pto.tmatmul.mx.bias %a, %a_scale, %b, %b_scale, %bias : (!pto.tile<...>, !pto.tile<...>,
!pto.tile<...>, !pto.tile<...>, !pto.tile<...>)  -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tmatmul.mx ins(%a, %a_scale, %b, %b_scale : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
outs(%c :  !pto.tile_buf<...>)
pto.tmatmul.mx.acc ins(%c_in, %a, %a_scale, %b, %b_scale : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>,
!pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c_out : !pto.tile_buf<...>)
pto.tmatmul.mx.bias ins(%a, %a_scale, %b, %b_scale, %bias : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>,
!pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c : !pto.tile_buf<...>)
```

---

### TMATMUL

该指令的详细介绍请见[isa/TMATMUL](../isa/TMATMUL_zh.md)


**IR Level 1 (SSA)：**
```text
%c = pto.tmatmul %a, %b : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tmatmul ins(%a, %b : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c : !pto.tile_buf<...>)
```

---

### TMATMUL_ACC

该指令的详细介绍请见[isa/TMATMUL_ACC](../isa/TMATMUL_ACC_zh.md)


**IR Level 1 (SSA)：**
```text
%c_out = pto.tmatmul.acc %c_in, %a, %b : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tmatmul.acc ins(%c_in, %a, %b : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c_out : !pto.tile_buf<...>)
```

---

### TMATMUL_BIAS

该指令的详细介绍请见[isa/TMATMUL_BIAS](../isa/TMATMUL_BIAS_zh.md)


**IR Level 1 (SSA)：**
```text
%c = pto.tmatmul.bias %a, %b, %bias : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tmatmul.bias ins(%a, %b, %bias : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c : !pto.tile_buf<...>)
```

---

### TGEMV

该指令的详细介绍请见[isa/TGEMV](../isa/TGEMV_zh.md)


**IR Level 1 (SSA)：**
```text
%c = pto.tgemv %a, %b : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
%c_out = pto.tgemv.acc %c_in, %a, %b : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
%c = pto.tgemv.bias %a, %b, %bias : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tgemv ins(%a, %b : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c : !pto.tile_buf<...>)
pto.tgemv.acc ins(%c_in, %a, %b : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c_out : !pto.tile_buf<...>)
pto.tgemv.bias ins(%a, %b, %bias : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c : !pto.tile_buf<...>)
```

---

### TGEMV_ACC

该指令的详细介绍请见[isa/TGEMV_ACC](../isa/TGEMV_ACC_zh.md)


**IR Level 1 (SSA)：**
```text
%c = pto.tgemv %a, %b : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
%c_out = pto.tgemv.acc %c_in, %a, %b : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
%c = pto.tgemv.bias %a, %b, %bias : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tgemv ins(%a, %b : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c : !pto.tile_buf<...>)
pto.tgemv.acc ins(%c_in, %a, %b : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c_out : !pto.tile_buf<...>)
pto.tgemv.bias ins(%a, %b, %bias : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c : !pto.tile_buf<...>)
```

---

### TGEMV_BIAS

该指令的详细介绍请见[isa/TGEMV_BIAS](../isa/TGEMV_BIAS_zh.md)


**IR Level 1 (SSA)：**
```text
%c = pto.tgemv %a, %b : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
%c_out = pto.tgemv.acc %c_in, %a, %b : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
%c = pto.tgemv.bias %a, %b, %bias : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tgemv ins(%a, %b : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c : !pto.tile_buf<...>)
pto.tgemv.acc ins(%c_in, %a, %b : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c_out : !pto.tile_buf<...>)
pto.tgemv.bias ins(%a, %b, %bias : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c : !pto.tile_buf<...>)
```
