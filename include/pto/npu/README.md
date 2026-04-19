# include/pto/npu/

NPU-side PTO instruction implementations. Different SoC generations have different optimized implementations and pipeline details.

## Choose by SoC

| SoC Generation | Directory | Description |
|----------------|----------|-------------|
| Ascend A2/A3 | `a2a3/` | Ascend 910B / 910C series implementations |
| Ascend A5 | `a5/` | Ascend 950 series implementations |
| Kirin 9030 | `kirin9030/` | Kirin 9030-specific implementations |
| Kirin X90 | `kirinX90/` | Kirin X90-specific implementations |

## Layout

```
include/pto/npu/
├── a2a3/                      # Ascend A2/A3 (910B/910C) series
│   ├── TAdd.hpp              # TADD implementation
│   ├── TSub.hpp              # TSUB implementation
│   ├── TMul.hpp              # TMUL implementation
│   ├── TMatmul.hpp           # TMATMUL implementation
│   ├── TLoad.hpp             # TLOAD implementation
│   ├── TStore.hpp            # TSTORE implementation
│   └── ...                    # Other instruction implementations
│
├── a5/                        # Ascend A5 (950) series
│   ├── TAdd.hpp
│   ├── TSub.hpp
│   ├── TMul.hpp
│   ├── TMatmul.hpp
│   ├── TLoad.hpp
│   ├── TStore.hpp
│   └── ...                    # Other implementations (A5-specific ops like TMATMUL_MX)
│
├── kirin9030/                 # Kirin 9030
└── kirinX90/                  # Kirin X90
```

## Selecting the SoC Version

SoC selection is controlled by the build system and test scripts:

- `tests/script/run_st.py` / `tests/script/build_st.py`: select via `-v a3|a5`
- `tests/npu/<soc>/src/st/CMakeLists.txt`: builds the corresponding ST targets and dependencies per SoC

## Key A2/A3 vs A5 Differences

| Feature | A2/A3 | A5 |
|---------|:------:|:--:|
| Matrix multiply unit | CUBE | CUBE (enhanced) |
| MXFP4/MXFP8 support | — | Supported |
| Vector instructions | Emulated | Full hardware support |
| Fractal layouts | Emulated | Full support |
| FP8 types | — | Supported |

## Related Docs

| Document | Content |
|----------|---------|
| [include/pto/README.md](../README.md) | PTO header entry point |
| [docs/getting-started.md](../../../docs/getting-started.md) | Complete getting started guide |
| [include/pto/npu/README_zh.md](./README_zh.md) | 中文版 |
