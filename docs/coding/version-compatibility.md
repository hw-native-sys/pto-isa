# Version Compatibility

This document details PTO-ISA's version compatibility strategy, differences between versions, platform support, and migration guides.

## Contents

- [1. Version Strategy](#1-version-strategy)
- [2. Platform Compatibility](#2-platform-compatibility)
- [3. API Compatibility](#3-api-compatibility)
- [4. Version History](#4-version-history)
- [5. Migration Guide](#5-migration-guide)

______________________________________________________________________

## 1. Version Strategy

### 1.1 Semantic Versioning

PTO-ISA follows Semantic Versioning 2.0.0:

```text
MAJOR.MINOR.PATCH

Example: v1.2.3
  ├─ 1: Major version
  ├─ 2: Minor version
  └─ 3: Patch version
```

**Version Increment Rules**:

- **MAJOR**: Incompatible API changes

  - Remove or rename public APIs
  - Change API behavior semantics
  - Modify data structure layouts
  - Example: v1.x.x → v2.0.0

- **MINOR**: Backward-compatible feature additions

  - Add new API functions
  - Add new operator support
  - Performance optimizations (no behavior change)
  - Example: v1.2.x → v1.3.0

- **PATCH**: Backward-compatible bug fixes

  - Bug fixes
  - Documentation updates
  - Internal refactoring (no API impact)
  - Example: v1.2.3 → v1.2.4

### 1.2 Current Version

- **Latest Stable**: v1.0.0
- **Release Date**: 2025-12-27
- **Support Status**: ✅ Long-term Support (LTS)

### 1.3 Support Lifecycle

| Version Type | Support Period | Description |
| -------------- | ---------------- | ------------- |
| **LTS (Long-term Support)** | 2 years | Major versions, full support and security updates |
| **Stable** | 1 year | Minor versions, bug fixes provided |
| **Development** | 3 months | Experimental features, stability not guaranteed |

**Currently Supported Versions**:

- v1.0.x (LTS): Supported until 2027-12-27
- v0.9.x (Stable): Supported until 2026-06-27

______________________________________________________________________

## 2. Platform Compatibility

### 2.1 Hardware Platforms

| Platform | Architecture | Support Status | Min Version | Recommended | Notes |
| ---------- | -------------- | ---------------- | ------------- | ------------- | ------- |
| **Ascend A2** | 910B1 | ✅ Full support | v1.0.0 | v1.0.0 | 24 cores, 512 KB L1/core |
| **Ascend A3** | 910B2 | ✅ Full support | v1.0.0 | v1.0.0 | 24 cores, 512 KB L1/core |
| **Ascend A5** | 910_9599 | ✅ Full support | v1.0.0 | v1.0.0 | 32 cores, 1 MB L1/core |
| **CPU (x86_64)** | x86-64 | ✅ Simulation | v1.0.0 | v1.0.0 | For development/debug |
| **CPU (AArch64)** | ARM64 | ✅ Simulation | v1.0.0 | v1.0.0 | For development/debug |

**Platform Feature Comparison**:

| Feature | A2 | A3 | A5 |
| --------- | ---- | ---- | ----- |
| Core Count | 24 | 24 | 32 |
| L1 Capacity/Core | 512 KB | 512 KB | 1 MB |
| Max Tile Size | 16×512 | 16×512 | 16×1024 |
| FP16 Support | ✅ | ✅ | ✅ |
| BF16 Support | ❌ | ✅ | ✅ |
| INT8 Support | ✅ | ✅ | ✅ |

### 2.2 Operating Systems

| OS | Version | Architecture | Support Status |
| ---- | --------- | -------------- | ---------------- |
| **Ubuntu** | 20.04, 22.04 | x86_64, aarch64 | ✅ Full support |
| **CentOS** | 7.6+, 8.x | x86_64, aarch64 | ✅ Full support |
| **EulerOS** | 2.8, 2.10 | x86_64, aarch64 | ✅ Full support |
| **Kylin** | V10 | x86_64, aarch64 | ✅ Full support |

### 2.3 Compiler Support

| Compiler | Min Version | Recommended | Notes |
| ---------- | ------------- | ------------- | ------- |
| **GCC** | 13.0 | 13.2+ | C++20 support required |
| **Clang** | 15.0 | 16.0+ | C++20 support required |
| **MSVC** | 2022 (17.0) | 2022 (17.5+) | Windows only |

### 2.4 Framework Integration

| Framework | Min Version | Recommended | Notes |
| ----------- | ------------- | ------------- | ------- |
| **PyTorch** | 2.0.0 | 2.1.0+ | Via torch_npu |
| **TensorFlow** | 2.6.0 | 2.12.0+ | Via tf_adapter |
| **ONNX Runtime** | 1.12.0 | 1.15.0+ | Custom EP |
| **MindSpore** | 2.0.0 | 2.2.0+ | Native support |

______________________________________________________________________

## 3. API Compatibility

### 3.1 Stable APIs

**Guaranteed Stable** (no breaking changes in v1.x):

```cpp
// Core Tile operations
TLOAD, TSTORE, TADD, TMUL, TMATMUL, TEXP, TMAX, ...

// Event synchronization
Event<Op1, Op2>, WAIT()

// Tile types
Tile<TileType, DataType, Rows, Cols>
TileLeft<DataType, Rows, Cols>
TileRight<DataType, Rows, Cols>
TileAcc<DataType, Rows, Cols>

// Global tensor
GlobalTensor<DataType>
```

### 3.2 Experimental APIs

**May Change** (marked with `EXPERIMENTAL_`):

```cpp
// Experimental features (may change in minor versions)
EXPERIMENTAL_TQUANTIZE()
EXPERIMENTAL_TDEQUANTIZE()
EXPERIMENTAL_TMXFP4()
```

### 3.3 Deprecated APIs

**To Be Removed** (marked with `DEPRECATED_`):

```cpp
// Deprecated in v1.0, will be removed in v2.0
DEPRECATED_TLOAD_OLD()  // Use TLOAD() instead
DEPRECATED_TSTORE_OLD() // Use TSTORE() instead
```

______________________________________________________________________

## 4. Version History

### v1.0.0 (2025-12-27) - LTS Release

**New Features**:

- ✅ Complete Tile abstraction API
- ✅ Event-based synchronization
- ✅ Multi-core SPMD programming model
- ✅ A2/A3/A5 platform support
- ✅ CPU simulation backend

**Performance Improvements**:

- 🚀 GEMM performance optimized (up to 95% peak)
- 🚀 Softmax performance improved by 40%
- 🚀 Reduced compilation time by 30%

**Bug Fixes**:

- 🐛 Fixed L1 memory overflow detection
- 🐛 Fixed event synchronization edge cases
- 🐛 Fixed alignment check for dynamic shapes

### v0.9.0 (2025-11-15) - Beta Release

**New Features**:

- ✅ Initial Tile API
- ✅ Basic operator support
- ✅ A2/A3 platform support

**Known Issues**:

- ⚠️ A5 support incomplete
- ⚠️ Event API unstable
- ⚠️ Documentation incomplete

______________________________________________________________________

## 5. Migration Guide

### 5.1 Migrating from v0.9 to v1.0

**API Changes**:

```cpp
// v0.9 (Old)
TLOAD_OLD(tile, ptr, size);
TSTORE_OLD(ptr, tile, size);

// v1.0 (New)
TLOAD(tile, GlobalTensor(ptr));
TSTORE(GlobalTensor(ptr), tile);
```

**Event API Changes**:

```cpp
// v0.9 (Old)
Event e;
TLOAD(tile, ..., &e);
WAIT_EVENT(e);

// v1.0 (New)
Event<Op::TLOAD, Op::TADD> e;
e = TLOAD(tile, ...);
WAIT(e);
```

**Tile Type Changes**:

```cpp
// v0.9 (Old)
Tile<float, 16, 256> tile;

// v1.0 (New)
Tile<TileType::Vec, float, 16, 256> tile;
```

### 5.2 Platform-Specific Migration

**A2 → A3 Migration**:

- ✅ No code changes required
- ✅ Recompile with `-DSOC_VERSION=Ascend910B2`
- ⚠️ BF16 now available (optional optimization)

**A3 → A5 Migration**:

- ✅ No code changes required
- ✅ Recompile with `-DSOC_VERSION=Ascend910_9599`
- 🚀 Can increase Tile sizes (larger L1)
- 🚀 Can use more cores (32 vs 24)

**Optimization Opportunities**:

```cpp
// A2/A3 configuration
constexpr int TILE_SIZE = 256;
constexpr int NUM_CORES = 24;

// A5 configuration (optimized)
#ifdef ASCEND_A5
constexpr int TILE_SIZE = 512;  // Larger L1
constexpr int NUM_CORES = 32;   // More cores
#else
constexpr int TILE_SIZE = 256;
constexpr int NUM_CORES = 24;
#endif
```

### 5.3 Compatibility Testing

**Test Checklist**:

```bash
# 1. Compile for all platforms
cmake -B build-a2 -DSOC_VERSION=Ascend910B1
cmake -B build-a3 -DSOC_VERSION=Ascend910B2
cmake -B build-a5 -DSOC_VERSION=Ascend910_9599

# 2. Run unit tests
ctest --test-dir build-a2
ctest --test-dir build-a3
ctest --test-dir build-a5

# 3. Verify numerical correctness
python3 tests/verify_accuracy.py --all-platforms

# 4. Performance regression test
python3 tests/benchmark.py --compare-baseline
```
