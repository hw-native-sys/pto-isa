# Manual / Resource Binding

This document describes manual resource binding and configuration operations.

**Total Operations:** 4

---

## Operations

### TASSIGN

For detailed instruction documentation, see [isa/TASSIGN](../isa/TASSIGN.md)

**AS Level 1 (SSA):**

```text
pto.tassign %tile, %addr : !pto.tile<...>, dtype
```

**AS Level 2 (DPS):**

```text
pto.tassign ins(%tile, %addr : !pto.tile_buf<...>, dtype)
```

---

### SETFMATRIX

For detailed instruction documentation, see [isa/SETFMATRIX](../isa/SETFMATRIX.md)

**AS Level 1 (SSA):**

```text
pto.SETFMATRIX %cfg : !pto.fmatrix_config -> ()
```

**AS Level 2 (DPS):**

```text
pto.SETFMATRIX ins(%cfg : !pto.fmatrix_config) outs()
```

---

### SET_IMG2COL_RPT

For detailed instruction documentation, see [isa/SET_IMG2COL_RPT](../isa/SET_IMG2COL_RPT.md)

**AS Level 1 (SSA):**

```text
pto.SET_IMG2COL_RPT %cfg : !pto.fmatrix_config -> ()
```

**AS Level 2 (DPS):**

```text
pto.SET_IMG2COL_RPT ins(%cfg : !pto.fmatrix_config) outs()
```

---

### SET_IMG2COL_PADDING

For detailed instruction documentation, see [isa/SET_IMG2COL_PADDING](../isa/SET_IMG2COL_PADDING.md)

**AS Level 1 (SSA):**

```text
pto.SET_IMG2COL_PADDING %cfg : !pto.fmatrix_config -> ()
```

**AS Level 2 (DPS):**

```text
pto.SET_IMG2COL_PADDING ins(%cfg : !pto.fmatrix_config) outs()
```

---
