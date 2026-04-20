# include/pto/npu/a5/

Ascend A5 (Ascend 950) series PTO instruction implementation headers.

## Key Files

```
include/pto/npu/a5/
├── TAdd.hpp            # TADD element-wise addition
├── TSub.hpp            # TSUB element-wise subtraction
├── TMul.hpp            # TMUL element-wise multiplication
├── TDiv.hpp            # TDIV element-wise division
├── TMatmul.hpp         # TMATMUL matrix multiply (enhanced CUBE)
├── TMatmulMx.hpp       # TMATMUL_MX matrix multiply with MX format
├── TLoad.hpp           # TLOAD GM → tile buffer
├── TStore.hpp          # TSTORE tile buffer → GM
├── TAssign.hpp         # TASSIGN resource binding
├── TSync.hpp           # TSYNC synchronization
├── ...                  # Other instruction implementations (Reduce, Expand, Layout, etc.)
```

## A5-Specific Features

| Feature | Description |
|---------|-------------|
| MXFP8 / MXFP4 | Hybrid-precision matrix multiply supported on A5 hardware |
| Fractal layouts | Full NZ / ZN / FR / RN fractal layout support |
| Vector hardware | `pto.v*` instructions have full hardware support on A5 (not emulated) |
| FP8 types | `f8e4m3`, `f8e5m2` data types supported |

## Key Differences from A2/A3

| Feature | A2/A3 | A5 |
|---------|:------:|:--:|
| Matrix multiply | Hardware CUBE | Enhanced CUBE |
| MXFP4 / MXFP8 | Not supported | Supported |
| Vector instructions | Emulated | Full hardware support |
| Fractal layouts | Emulated | Full support |
| FP8 types | Not supported | Supported |

## Related

| Document | Content |
|----------|---------|
| [docs/isa/](../../docs/isa/) | ISA semantics and examples |
| [tests/npu/a5/src/st/](../../tests/npu/a5/src/st/) | A5 NPU ST tests |
| [include/pto/npu/](../README.md) | NPU implementation entry |
