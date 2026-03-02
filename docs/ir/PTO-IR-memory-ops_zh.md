# 内存操作（GM <-> Tile）

本文档描述全局内存和 tile 之间的内存操作。

**操作总数：** 6

---

## 操作

### TLOAD

该指令的详细介绍请见[isa/TLOAD](../isa/TLOAD_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tload %mem : !pto.partition_tensor_view<MxNxdtype> ->
!pto.tile<loc, dtype, rows, cols, blayout, slayout, fractal, pad>
```

**IR Level 2 (DPS)：**
```text
pto.tload ins(%mem : !pto.partition_tensor_view<MxNxdtype>) outs(%dst : !pto.tile_buf<...>)
```

---

### TPREFETCH

该指令的详细介绍请见[isa/TPREFETCH](../isa/TPREFETCH_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.tprefetch %src : !pto.global<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tprefetch ins(%src : !pto.global<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSTORE

该指令的详细介绍请见[isa/TSTORE](../isa/TSTORE_zh.md)


**IR Level 1 (SSA)：**
```text
pto.tstore %src, %mem : (!pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

**IR Level 2 (DPS)：**
```text
pto.tstore ins(%src : !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```

---

### TSTORE_FP

该指令的详细介绍请见[isa/TSTORE_FP](../isa/TSTORE_FP_zh.md)


**IR Level 1 (SSA)：**
```text
pto.tstore.fp %src, %fp, %mem : (!pto.tile<...>, !pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

**IR Level 2 (DPS)：**
```text
pto.tstore.fp ins(%src, %fp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```

---

### MGATHER

该指令的详细介绍请见[isa/MGATHER](../isa/MGATHER_zh.md)


**IR Level 1 (SSA)：**
```text
%dst = pto.mgather %mem, %idx : (!pto.partition_tensor_view<MxNxdtype>, pto.tile<...>)
-> !pto.tile<loc, dtype, rows, cols, blayout, slayout, fractal, pad>
```

**IR Level 2 (DPS)：**
```text
pto.mgather ins(%mem, %idx : !pto.partition_tensor_view<MxNxdtype>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### MSCATTER

该指令的详细介绍请见[isa/MSCATTER](../isa/MSCATTER_zh.md)


**IR Level 1 (SSA)：**
```text
pto.mscatter %src, %idx, %mem : (!pto.tile<...>, !pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

**IR Level 2 (DPS)：**
```text
pto.mscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```
