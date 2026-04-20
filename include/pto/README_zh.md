# include/pto/

本目录是 PTO Tile Library 的**主要公共头文件入口**，包含：

- Tile 类型系统与共享工具
- PTO 指令 API 声明（Auto / Manual 两种形式）
- CPU 仿真 / stub 支持
- NPU 指令实现（按 SoC 代际划分）

## 推荐的 include 方式

| 场景 | 推荐头文件 |
|------|-----------|
| 上层代码（通用） | `include/pto/pto-inst.hpp` — 统一入口 |
| CPU 仿真时自动注入 stub | `__CPU_SIM` 定义时自动引入 `pto/common/cpu_stub.hpp` |

## 目录结构

```
include/pto/
├── pto-inst.hpp              # 统一入口头文件（推荐）
├── pto.hpp                    # 核心头文件（含 pto-inst.hpp）
│
├── common/                    # 平台无关的基础设施
│   ├── pto_tile.hpp          # 核心 Tile 类型与布局
│   ├── pto_instr.hpp         # PTO 指令声明
│   ├── pto_instr_impl.hpp    # PTO 指令共享实现
│   ├── memory.hpp             # 内存操作相关
│   ├── constants.hpp          # 常量定义
│   ├── utils.hpp              # 通用工具
│   ├── type.hpp               # 类型定义
│   └── cpu_stub.hpp           # CPU 仿真 stub
│
├── cpu/                       # CPU 侧仿真实现（如启用）
│
└── npu/                      # NPU 侧实现（按 SoC 版本拆分）
    ├── a2a3/                 # Ascend A2/A3 系列
    │   ├── TAdd.hpp          # TADD 实现
    │   ├── TMatmul.hpp       # TMATMUL 实现
    │   ├── TLoad.hpp         # TLOAD 实现
    │   └── ...               # 其他指令实现
    └── a5/                   # Ascend A5 系列
        ├── TAdd.hpp
        ├── TMatmul.hpp
        ├── TLoad.hpp
        └── ...
```

## 常用 TileType 与硬件 Buffer 对应关系

| TileType | 硬件 Buffer | 容量 | 典型用途 |
|----------|------------|------|----------|
| `Vec` | Unified Buffer（UB） | 256 KB | 通用逐元素运算 |
| `Mat` | L1 | 512 KB | 矩阵乘法操作数 |
| `Left` | L0A | 64 KB | 矩阵乘法 A 操作数 |
| `Right` | L0B | 64 KB | 矩阵乘法 B 操作数 |
| `Acc` | L0C | 256 KB | 矩阵乘法累加器 |

## 典型使用方式

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

// 定义 GlobalTensor
using DynShape = Shape<1, 1, 1, kGRows_, kGCols_>;
using DynStride = Stride<1, 1, 1, kGCols_, 1>;
using GlobalData = GlobalTensor<T, DynShape, DynStride>;
GlobalData srcGlobal(src);

// 定义 Tile
using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
TileData srcTile(kTRows_, kTCols_);

// PTO 指令调用
TLOAD(srcTile, srcGlobal);
TADD(dstTile, src0Tile, src1Tile);
TSTORE(dstGlobal, dstTile);
```

## 相关文档

- [ISA 指令参考](../docs/isa/) — PTO ISA 完整指令语义
- [Tile 编程模型](../../docs/coding/Tile_zh.md) — Tile 类型系统详解
- [include/README_zh.md](./README_zh.md) — 中文版入口
