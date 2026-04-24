# 手动/资源绑定

本文档描述手动资源绑定和配置操作。

**操作总数：** 4

---

## 操作

### TASSIGN

该指令的详细介绍请见[isa/TASSIGN](../isa/TASSIGN_zh.md)

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

该指令的详细介绍请见[isa/SETFMATRIX](../isa/SETFMATRIX_zh.md)

**AS Level 1 (SSA)：**

```text
pto.SETFMATRIX %cfg : !pto.fmatrix_config -> ()
```

**AS Level 2 (DPS)：**

```text
pto.SETFMATRIX ins(%cfg : !pto.fmatrix_config) outs()
```

---

### SET_IMG2COL_RPT

该指令的详细介绍请见[isa/SET_IMG2COL_RPT](../isa/SET_IMG2COL_RPT_zh.md)

**AS Level 1 (SSA)：**

```text
pto.SET_IMG2COL_RPT %cfg : !pto.fmatrix_config -> ()
```

**AS Level 2 (DPS)：**

```text
pto.SET_IMG2COL_RPT ins(%cfg : !pto.fmatrix_config) outs()
```

---

### SET_IMG2COL_PADDING

该指令的详细介绍请见[isa/SET_IMG2COL_PADDING](../isa/SET_IMG2COL_PADDING_zh.md)

**AS Level 1 (SSA)：**

```text
pto.SET_IMG2COL_PADDING %cfg : !pto.fmatrix_config -> ()
```

**AS Level 2 (DPS)：**

```text
pto.SET_IMG2COL_PADDING ins(%cfg : !pto.fmatrix_config) outs()
```

---
