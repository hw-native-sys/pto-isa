# C++接口与IR接口区别介绍

PTO 虚拟指令集提供两个接口面：「PTO虚拟指令集（C++）」描述宿主侧的 C++ 内建
接口；「PTO虚拟指令集（IR）」描述同一批指令在 IR 层的操作形式。本文说明两者
的定位差异与相互映射规则，供读者在两章之间互查。

---

## 目录

- [1. 定位差异](#1-定位差异)
- [2. 同一条指令的两种写法](#2-同一条指令的两种写法)
- [3. 命名映射规则](#3-命名映射规则)
- [4. 类型系统对照](#4-类型系统对照)
- [5. 参数、属性与可选操作数](#5-参数属性与可选操作数)
- [6. 结果传递与同步](#6-结果传递与同步)

---

## 1. 定位差异

| | PTO虚拟指令集（C++） | PTO虚拟指令集（IR） |
| --- | --- | --- |
| 接口形态 | C++ 模板函数，统一入口头文件 `<pto/pto-inst.hpp>` | `pto.*` 操作的文本装配形式 |
| 面向读者 | 编写 kernel / 算子的开发者 | 编写或阅读编译器前端、查看指令汇编形态的工程师 |
| 结果传递 | 出参引用 `dst`，返回 `RecordEvent` | 目标传递风格（DPS），结果写入 `outs()` |
| 操作数类型 | C++ 类型（`Tile<>`、`GlobalTensor<>`、标量值） | IR 类型（`!pto.tile_buf<...>`、`!pto.partition_tensor_view<...>`、`!pto.ptr<T>`） |
| 变体表达 | 重载、模板参数、运行时枚举参数 | 点号后缀操作名、属性 `{...}`、可选操作数 |
| 文档粒度 | 一条指令一页 | 一组操作一页，页内按 `pto.xxx` 分段 |

两章描述同一套架构语义：对有效区域内每个 `(i, j)` 的行为定义完全一致，
差异只在接口形态与承载方式。指令的 IR（汇编）形态统一以 IR 章为准。

---

## 2. 同一条指令的两种写法

以逐元素乘法为例。

C++ 接口（Auto 模式）：

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
    using TileT = Tile<TileType::Vec, float, 16, 16>;
    TileT src0, src1, dst;
    TMUL(dst, src0, src1);
}
```

IR 操作：

```mlir
pto.tmul ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                  v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                  fractal=512, pad=0>,
                  !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                  v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                  fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                  v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                  fractal=512, pad=0>)
```

两侧语义相同：对有效区域内每个 `(i, j)`，`dst[i,j] = src0[i,j] * src1[i,j]`。

---

## 3. 命名映射规则

### 3.1 基本规则

C++ 为 PascalCase，IR 为小写并带 `pto.` 前缀：

```
TMUL  <->  pto.tmul
TLOAD <->  pto.tload
```

在另一章查找同一条指令时，直接变换大小写并加/去前缀即可。

### 3.2 变体后缀

C++ 用下划线后缀，IR 用点号后缀：

| C++ | IR |
| --- | --- |
| `TMATMUL` | `pto.tmatmul` |
| `TMATMUL_ACC` | `pto.tmatmul.acc` |
| `TMATMUL_BIAS` | `pto.tmatmul.bias` |
| `TMATMUL_MX` | `pto.tmatmul.mx` |

### 3.3 一对多：方向特化

C++ 用一个多态接口配合方向模板参数，IR 将方向烘焙进操作名：

| C++ | IR |
| --- | --- |
| `TPUSH` | `pto.tpush_to_aiv` / `pto.tpush_to_aic` |
| `TPOP` | `pto.tpop_from_aiv` / `pto.tpop_from_aic` |
| `TALLOC` | `pto.talloc_to_aiv` / `pto.talloc_to_aic` |
| `TFREE` | `pto.tfree_from_aiv` / `pto.tfree_from_aic` |

---

## 4. 类型系统对照

### 4.1 Tile 类型

| C++ `Tile<>` 模板参数 | IR `!pto.tile_buf<>` 参数 | 含义 |
| --- | --- | --- |
| `TileType::Vec` / `Acc` / `Mat` … | `loc=vec` / `acc` / `mat` / `left` / `right` / `scaling` / `ctrl` | 位置空间 |
| `float` / `half` / `bfloat16_t` / `int32_t` … | `f32` / `f16` / `bf16` / `i32` … | 元素类型 |
| `Rows` / `Cols` | `rows` / `cols` | 物理形状 |
| `ValidRow` / `ValidCol` | `v_row` / `v_col` | 有效区域 |
| `BLayout::RowMajor` / `ColMajor` | `blayout=row_major` / `col_major` | 块布局 |
| `SLayout::NoneBox` / `RowMajor` … | `slayout=none_box` / `row_major` … | 存储布局 |
| `SFractalSize` | `fractal=512` / `1024` | 分形粒度 |
| `PadVal` | `pad` | 填充值 |

IR 侧元素类型另有 `f8E4M3`、`f8E5M2`、`!pto.hif8`、`!pto.f4E1M2x2`、`!pto.f4E2M1x2` 等低精度形式。

### 4.2 全局内存与指针

| C++ | IR |
| --- | --- |
| `GlobalTensor<>` | `!pto.partition_tensor_view<MxNxdtype>` |
| 由 C++ 对象封装的地址能力 | `!pto.ptr<T, space>`、memref |

### 4.3 动态有效形状

- C++：模板参数传 `-1`，运行期通过构造参数指定。
- IR：类型中写 `v_row=?` / `v_col=?`，由 `pto.alloc_tile` 的 `valid_row` / `valid_col`
  操作数在创建处提供：

```mlir
%t1 = pto.alloc_tile valid_row = %c16 valid_col = %c256
    : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=256,
                    v_row=?, v_col=?, blayout=col_major, ...>
```

---

## 5. 参数、属性与可选操作数

C++ 的编译期模板参数与运行时枚举参数，在 IR 侧统一为操作属性：

| C++ | IR |
| --- | --- |
| `TMATMUL_MX<AccPhase::Final>(...)` | `pto.tmatmul.mx ... {accPhase = #pto<acc_phase final>}` |
| `TCVT(dst, src, RoundMode, SaturationMode)` | `pto.tcvt ... {rmode = #pto<round_mode RINT>, satmode = ...}` |

C++ 的重载族在 IR 侧常用可选操作数表达：

| C++ 重载 | IR |
| --- | --- |
| `TCVT(dst, src, tmp, mode, ...)` | `pto.tcvt ins(<src>[, <tmp>] ...)`，`tmp` 可选 |

---

## 6. 结果传递与同步

- C++ 接口统一返回 `RecordEvent`，结果写入出参 `dst`；通过 `WaitEvents&&...`
  可变尾参建立与先前操作的依赖关系。
- IR 计算与搬运类操作均为 DPS，无 SSA 返回值；产生句柄或做查询的操作返回
  SSA 值（如 `pto.alloc_tile`、`pto.load`、`pto.tpop_from_*`、`pto.get_block_idx`）。
- IR 侧缓冲区生命周期相关的同步令牌由 `pto.get_buf` / `pto.rls_buf` 管理
  （见「指针与视图操作」）。
