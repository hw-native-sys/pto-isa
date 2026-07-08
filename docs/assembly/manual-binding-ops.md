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
