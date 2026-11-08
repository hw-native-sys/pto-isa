# Linx target backend

The Linx backend is a peer PTO NPU target alongside A2/A3, A5, A6, and Kirin.
It uses the Linx compiler for device builds and the Linx TileOP CPU simulator
for host-side validation.

## Linx NPU compile

Use the Linx compiler and define `__linx`:

```text
--target=linx64v5-unknown-linux-musl -mlxbc -fenable-matrix -std=c++20 -D__linx
```

The public entry point remains:

```cpp
#include <pto/pto-inst.hpp>
```

## Linx CPU simulator

Compile with a host C++ compiler using both `PTO_LINX_CPU_SIM` and
`__cpu_sim__`. This selects the Linx CPU-sim implementation while preserving
the same public PTO include.

The Linx implementation is kept under `pto/npu/linx/impl` because its TileOP
headers and inline assembly are target-owned. The compatibility header only
provides empty CANN address-space annotations needed by shared declarations;
it does not select a CANN backend.
