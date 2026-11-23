# TPREFETCH

## Product Support<a name="section73648168211"></a>

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Supported
<!-- end id3 -->



## Tile Operation Diagram

![TPREFETCH tile operation](../../figures/isa/TPREFETCH.svg)

## Introduction

Prefetch data from global memory into a tile-local cache/buffer (implementation-defined). This is typically used to reduce latency before a subsequent `TLOAD`.

Note: unlike most PTO instructions, `TPREFETCH` does **not** implicitly call event waits in the C++ wrapper.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename GlobalData>
PTO_INST RecordEvent TPREFETCH(TileData &dst, GlobalData &src);
```

## Constraints

- Semantics and caching behavior are target/implementation-defined.
- Some targets may ignore prefetches or treat them as hints.

## Math Interpretation

Unless otherwise specified, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

## Examples

See related examples in `docs/en/isa/` and `docs/en/coding/tutorials/`.
