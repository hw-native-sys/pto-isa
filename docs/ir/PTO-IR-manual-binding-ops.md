# Manual / Resource Binding

This document describes manual resource binding and configuration operations.

**Total Operations:** 6

---

## Operations

### TASSIGN

For detailed instruction documentation, see [isa/TASSIGN](../isa/TASSIGN.md)


**IR Level 1 (SSA):**
```text
pto.tassign %tile, %addr : !pto.tile<...>, dtype
```

**IR Level 2 (DPS):**
```text
pto.tassign ins(%tile, %addr : !pto.tile_buf<...>, dtype)
```

---

### TSETHF32MODE

For detailed instruction documentation, see [isa/TSETHF32MODE](../isa/TSETHF32MODE.md)

**IR Level 1 (SSA):**
```text
pto.tsethf32mode {enable = true, mode = ...}
```

**IR Level 2 (DPS):**
```text
pto.tsethf32mode ins({enable = true, mode = ...}) outs()
```

---

### TSETTF32MODE

For detailed instruction documentation, see [isa/TSETTF32MODE](../isa/TSETTF32MODE.md)

**IR Level 1 (SSA):**
```text
pto.tsettf32mode {enable = true, mode = ...}
```

**IR Level 2 (DPS):**
```text
pto.tsettf32mode ins({enable = true, mode = ...}) outs()
```

---

### TSETFMATRIX

For detailed instruction documentation, see [isa/TSETFMATRIX](../isa/TSETFMATRIX.md)


**IR Level 1 (SSA):**
```text
pto.tsetfmatrix %cfg : !pto.fmatrix_config -> ()
```

**IR Level 2 (DPS):**
```text
pto.tsetfmatrix ins(%cfg : !pto.fmatrix_config) outs()
```

---

### TSET_IMG2COL_RPT

For detailed instruction documentation, see [isa/TSET_IMG2COL_RPT](../isa/TSET_IMG2COL_RPT.md)

**IR Level 1 (SSA):**
```text
pto.tset_img2col_rpt %cfg : !pto.fmatrix_config -> ()
```

**IR Level 2 (DPS):**
```text
pto.tset_img2col_rpt ins(%cfg : !pto.fmatrix_config) outs()
```

---

### TSET_IMG2COL_PADDING

For detailed instruction documentation, see [isa/TSET_IMG2COL_PADDING](../isa/TSET_IMG2COL_PADDING.md)

**IR Level 1 (SSA):**
```text
pto.tset_img2col_padding %cfg : !pto.fmatrix_config -> ()
```

**IR Level 2 (DPS):**
```text
pto.tset_img2col_padding ins(%cfg : !pto.fmatrix_config) outs()
```

---

