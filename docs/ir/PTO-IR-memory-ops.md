# Memory (GM <-> Tile)

This document describes memory operations between global memory and tiles.

**Total Operations:** 6

---

## Operations

### TLOAD

For detailed instruction documentation, see [isa/TLOAD](../isa/TLOAD.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tload %mem : !pto.partition_tensor_view<MxNxdtype> ->
!pto.tile<loc, dtype, rows, cols, blayout, slayout, fractal, pad>
```

**IR Level 2 (DPS):**
```text
pto.tload ins(%mem : !pto.partition_tensor_view<MxNxdtype>) outs(%dst : !pto.tile_buf<...>)
```

---

### TPREFETCH

For detailed instruction documentation, see [isa/TPREFETCH](../isa/TPREFETCH.md)


**IR Level 1 (SSA):**
```text
%dst = pto.tprefetch %src : !pto.global<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tprefetch ins(%src : !pto.global<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSTORE

For detailed instruction documentation, see [isa/TSTORE](../isa/TSTORE.md)


**IR Level 1 (SSA):**
```text
pto.tstore %src, %mem : (!pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

**IR Level 2 (DPS):**
```text
pto.tstore ins(%src : !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```

---

### TSTORE_FP

For detailed instruction documentation, see [isa/TSTORE_FP](../isa/TSTORE_FP.md)


**IR Level 1 (SSA):**
```text
pto.tstore.fp %src, %fp, %mem : (!pto.tile<...>, !pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

**IR Level 2 (DPS):**
```text
pto.tstore.fp ins(%src, %fp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```

---

### MGATHER

For detailed instruction documentation, see [isa/MGATHER](../isa/MGATHER.md)


**IR Level 1 (SSA):**
```text
%dst = pto.mgather %mem, %idx : (!pto.partition_tensor_view<MxNxdtype>, pto.tile<...>)
-> !pto.tile<loc, dtype, rows, cols, blayout, slayout, fractal, pad>
```

**IR Level 2 (DPS):**
```text
pto.mgather ins(%mem, %idx : !pto.partition_tensor_view<MxNxdtype>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### MSCATTER

For detailed instruction documentation, see [isa/MSCATTER](../isa/MSCATTER.md)


**IR Level 1 (SSA):**
```text
pto.mscatter %src, %idx, %mem : (!pto.tile<...>, !pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

**IR Level 2 (DPS):**
```text
pto.mscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```

---

