# include/pto/npu/a2a3/

Ascend A2/A3 (Ascend 910B / 910C) series PTO instruction implementation headers.

## Key Files

```
include/pto/npu/a2a3/
├── TAdd.hpp            # TADD element-wise addition
├── TSub.hpp            # TSUB element-wise subtraction
├── TMul.hpp            # TMUL element-wise multiplication
├── TDiv.hpp            # TDIV element-wise division
├── TMatmul.hpp         # TMATMUL matrix multiply (hardware CUBE)
├── TLoad.hpp           # TLOAD GM → tile buffer
├── TStore.hpp          # TSTORE tile buffer → GM
├── TAssign.hpp         # TASSIGN resource binding
├── TSync.hpp           # TSYNC synchronization
├── ...                  # Other instruction implementations (Reduce, Expand, Layout, etc.)
```

## Key Differences from A5

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
| [tests/npu/a2a3/src/st/](../../tests/npu/a2a3/src/st/) | A2/A3 NPU ST tests |
| [include/pto/npu/](../README.md) | NPU implementation entry |
