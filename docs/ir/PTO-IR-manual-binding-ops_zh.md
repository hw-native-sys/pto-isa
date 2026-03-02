# 手动/资源绑定

本文档描述手动资源绑定和配置操作。

**操作总数：** 6

---

## 操作

### TASSIGN

该指令的详细介绍请见[isa/TASSIGN](../isa/TASSIGN_zh.md)


**IR Level 1 (SSA)：**
```text
pto.tassign %tile, %addr : !pto.tile<...>, dtype
```

**IR Level 2 (DPS)：**
```text
pto.tassign ins(%tile, %addr : !pto.tile_buf<...>, dtype)
```

---

### TSETHF32MODE

该指令的详细介绍请见[isa/TSETHF32MODE](../isa/TSETHF32MODE_zh.md)

**IR Level 1 (SSA)：**
```text
pto.tsethf32mode {enable = true, mode = ...}
```

**IR Level 2 (DPS)：**
```text
pto.tsethf32mode ins({enable = true, mode = ...}) outs()
```

---

### TSETTF32MODE

该指令的详细介绍请见[isa/TSETTF32MODE](../isa/TSETTF32MODE_zh.md)

**IR Level 1 (SSA)：**
```text
pto.tsettf32mode {enable = true, mode = ...}
```

**IR Level 2 (DPS)：**
```text
pto.tsettf32mode ins({enable = true, mode = ...}) outs()
```

---

### TSETFMATRIX

该指令的详细介绍请见[isa/TSETFMATRIX](../isa/TSETFMATRIX_zh.md)


**IR Level 1 (SSA)：**
```text
pto.tsetfmatrix %cfg : !pto.fmatrix_config -> ()
```

**IR Level 2 (DPS)：**
```text
pto.tsetfmatrix ins(%cfg : !pto.fmatrix_config) outs()
```

---

### TSET_IMG2COL_RPT

该指令的详细介绍请见[isa/TSET_IMG2COL_RPT](../isa/TSET_IMG2COL_RPT_zh.md)

**IR Level 1 (SSA)：**
```text
pto.tset_img2col_rpt %cfg : !pto.fmatrix_config -> ()
```

**IR Level 2 (DPS)：**
```text
pto.tset_img2col_rpt ins(%cfg : !pto.fmatrix_config) outs()
```

---

### TSET_IMG2COL_PADDING

该指令的详细介绍请见[isa/TSET_IMG2COL_PADDING](../isa/TSET_IMG2COL_PADDING_zh.md)

**IR Level 1 (SSA)：**
```text
pto.tset_img2col_padding %cfg : !pto.fmatrix_config -> ()
```

**IR Level 2 (DPS)：**
```text
pto.tset_img2col_padding ins(%cfg : !pto.fmatrix_config) outs()
```

---

