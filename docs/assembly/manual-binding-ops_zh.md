# 手动/资源绑定

本文档描述手动资源绑定和配置操作。

**操作总数：** 4

---

## 操作

### TASSIGN

该指令的详细介绍请见[isa/TASSIGN](../isa/tile/ops/sync-and-config/tassign_zh.md)

**AS Level 1 (SSA)：**

```text
pto.tassign %tile, %addr : !pto.tile<...>, dtype
```

**AS Level 2 (DPS)：**

```text
pto.tassign ins(%tile, %addr : !pto.tile_buf<...>, dtype)
```

---

### SETFMATRIX

该指令的详细介绍请见[pto.setfmatrix](../isa/tile/ops/sync-and-config/setfmatrix.md)

**AS Level 1 (SSA)：**

```text
pto.setfmatrix %cfg : !pto.fmatrix_config -> ()
```

**AS Level 2 (DPS)：**

```text
pto.setfmatrix ins(%cfg : !pto.fmatrix_config) outs()
```

---

### SET_IMG2COL_RPT

该指令的详细介绍请见[pto.set_img2col_rpt](../isa/tile/ops/sync-and-config/set-img2col-rpt.md)

**AS Level 1 (SSA)：**

```text
pto.set_img2col_rpt %cfg : !pto.fmatrix_config -> ()
```

**AS Level 2 (DPS)：**

```text
pto.set_img2col_rpt ins(%cfg : !pto.fmatrix_config) outs()
```

---

### SET_IMG2COL_PADDING

该指令的详细介绍请见[pto.set_img2col_padding](../isa/tile/ops/sync-and-config/set-img2col-padding.md)

**AS Level 1 (SSA)：**

```text
pto.set_img2col_padding %cfg : !pto.fmatrix_config -> ()
```

**AS Level 2 (DPS)：**

```text
pto.set_img2col_padding ins(%cfg : !pto.fmatrix_config) outs()
```

---
