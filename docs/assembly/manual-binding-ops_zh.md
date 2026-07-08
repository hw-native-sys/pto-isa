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
