# include/pto/npu/kirin9030/

Kirin 9030 series PTO instruction implementation headers.

Kirin 9030 is an Ascend SoC variant targeting consumer scenarios. Its PTO instruction implementations may differ in certain details from other datacenter SoCs (A2/A3/A5).

## Key Files

```
include/pto/npu/kirin9030/
├── TAdd.hpp
├── TSub.hpp
├── TMul.hpp
├── TDiv.hpp
├── TMatmul.hpp
├── TLoad.hpp
├── TStore.hpp
├── TAssign.hpp
├── TSync.hpp
└── ...                  # Other instruction implementations
```

## Related

| Document | Content |
|----------|---------|
| [docs/isa/](../../docs/isa/) | ISA semantics and examples |
| [include/pto/npu/](../README.md) | NPU implementation entry |
