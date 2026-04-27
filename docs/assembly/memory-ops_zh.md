# 内存操作（GM <-> Tile）

本文档描述全局内存和 tile 之间的内存操作。

**操作总数：** 6

---

## 操作

### TLOAD

该指令的详细介绍请见[isa/TLOAD](../isa/tile/ops/memory-and-data-movement/tload_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tload %mem : !pto.partition_tensor_view<MxNxdtype> ->
!pto.tile<loc, dtype, rows, cols, blayout, slayout, fractal, pad>
```

**AS Level 2 (DPS)：**

```text
pto.tload ins(%mem : !pto.partition_tensor_view<MxNxdtype>) outs(%dst : !pto.tile_buf<...>)
```

---

### TPREFETCH

该指令的详细介绍请见[isa/TPREFETCH](../isa/tile/ops/memory-and-data-movement/tprefetch_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.tprefetch %src : !pto.global<...> -> !pto.tile<...>
```

**AS Level 2 (DPS)：**

```text
pto.tprefetch ins(%src : !pto.global<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSTORE

该指令的详细介绍请见[isa/TSTORE](../isa/tile/ops/memory-and-data-movement/tstore_zh.md)

**AS Level 1 (SSA)：**

```text
pto.tstore %src, %mem : (!pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

**AS Level 2 (DPS)：**

```text
pto.tstore ins(%src : !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```

---

### TSTORE_FP

该指令的详细介绍请见[isa/TSTORE_FP](../isa/tile/ops/memory-and-data-movement/tstore_zh.md)

**AS Level 1 (SSA)：**

```text
pto.tstore.fp %src, %fp, %mem : (!pto.tile<...>, !pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

**AS Level 2 (DPS)：**

```text
pto.tstore.fp ins(%src, %fp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```

---

### MGATHER

该指令的详细介绍请见[isa/MGATHER](../isa/tile/ops/memory-and-data-movement/mgather_zh.md)

**AS Level 1 (SSA)：**

```text
%dst = pto.mgather %mem, %idx : (!pto.partition_tensor_view<MxNxdtype>, pto.tile<...>)
-> !pto.tile<loc, dtype, rows, cols, blayout, slayout, fractal, pad>
```

**AS Level 2 (DPS)：**

```text
pto.mgather ins(%mem, %idx : !pto.partition_tensor_view<MxNxdtype>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### MSCATTER

该指令的详细介绍请见[isa/MSCATTER](../isa/tile/ops/memory-and-data-movement/mscatter_zh.md)

**AS Level 1 (SSA)：**

```text
pto.mscatter %src, %idx, %mem : (!pto.tile<...>, !pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

**AS Level 2 (DPS)：**

```text
pto.mscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```
