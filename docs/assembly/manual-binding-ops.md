# Manual / Resource Binding

This document describes manual resource binding and configuration operations.

**Total Operations:** 4

---

## Operations

### TASSIGN

For detailed instruction documentation, see [isa/TASSIGN](../isa/tile/ops/sync-and-config/tassign.md)

**AS Level 1 (SSA):**

```text
pto.tassign %tile, %addr : !pto.tile<...>, dtype
```

**AS Level 2 (DPS):**

```text
pto.tassign ins(%tile, %addr : !pto.tile_buf<...>, dtype)
```

---

### pto.setfmatrix

For detailed instruction documentation, see [pto.setfmatrix](../isa/tile/ops/sync-and-config/setfmatrix.md).

**AS Level 1 (SSA):**

```text
pto.setfmatrix %cfg : !pto.fmatrix_config -> ()
```

**AS Level 2 (DPS):**

```text
pto.setfmatrix ins(%cfg : !pto.fmatrix_config) outs()
```

---

### pto.set_img2col_rpt

For detailed instruction documentation, see [pto.set_img2col_rpt](../isa/tile/ops/sync-and-config/set-img2col-rpt.md)

**AS Level 1 (SSA):**

```text
pto.set_img2col_rpt %cfg : !pto.fmatrix_config -> ()
```

**AS Level 2 (DPS):**

```text
pto.set_img2col_rpt ins(%cfg : !pto.fmatrix_config) outs()
```

---

### pto.set_img2col_padding

For detailed instruction documentation, see [pto.set_img2col_padding](../isa/tile/ops/sync-and-config/set-img2col-padding.md)

**AS Level 1 (SSA):**

```text
pto.set_img2col_padding %cfg : !pto.fmatrix_config -> ()
```

**AS Level 2 (DPS):**

```text
pto.set_img2col_padding ins(%cfg : !pto.fmatrix_config) outs()
```

---
