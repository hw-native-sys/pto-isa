# Tile 编程模型

PTO Tile Lib 程序围绕 **Tile** 进行编写：Tile 是固定容量的二维缓冲区，是大多数 PTO 指令的计算单元，也是数据搬运的基本单位。

概念上，Tile 位于**片上 Tile 存储**（类似寄存器文件或 SRAM 的存储区），并通过 `TLOAD`/`TSTORE` 在全局内存（GM）与片上之间搬运。在 CPU 仿真后端中，Tile 存放在主机内存，但会保持相同的形状/布局/有效区域规则，便于验证代码合法性与语义一致性。

本文档描述 `include/pto/common/pto_tile.hpp` 中的 C++ Tile 类型以及其布局/有效区域约束。

## Tile 表示什么

一个 Tile 可由以下五类属性刻画：

- **位置（Location）**：Tile 属于哪一类逻辑 Tile 存储（例如向量 vs 矩阵/立方寄存器类）。
- **元素类型（Element type）**：标量元素类型（`float`、`half`、`int8_t` 等）。
- **容量形状（Capacity shape）**：编译期 `Rows × Cols` 固定容量。
- **布局（Layout）**：基础布局（`BLayout`）以及可选的盒化/分形布局（`SLayout`、`SFractalSize`）。
- **有效区域（Valid region）**：本次操作中有意义的行/列数（静态或动态）。

## `pto::Tile` 类型

Tile 以 C++ 模板形式声明：

```cpp
pto::Tile<
  pto::TileType Loc_,
  Element_,
  Rows_,
  Cols_,
  pto::BLayout BLayout_      = pto::BLayout::RowMajor,
  RowValid_                  = Rows_,
  ColValid_                  = Cols_,
  pto::SLayout SLayout_      = pto::SLayout::NoneBox,
  SFractalSize_              = pto::TileConfig::fractalABSize,
  pto::PadValue PadValue_    = pto::PadValue::Null
>;
```

### 位置（`TileType`）

`TileType` 编码 Tile 的逻辑/物理存储类，并参与重载选择与编译期检查：

- `TileType::Vec`：向量 Tile 存储（UB / vector pipeline）。
- `TileType::Mat`：通用矩阵 Tile 存储（Matrix L1）。
- `TileType::Left`、`TileType::Right`：矩阵乘操作数 Tile（Matrix L0A/L0B）。
- `TileType::Acc`：矩阵乘累加器 Tile。
- `TileType::Bias`、`TileType::Scaling`：部分 matmul/move 路径的辅助 Tile。

`docs/isa/` 中的指令页会声明每条指令允许哪些位置类型。

### 容量形状（`Rows_`、`Cols_`）

`Rows_` 与 `Cols_` 定义 Tile 对象的**静态容量**。多数指令要求静态形状以便编译期特化与优化。

### 有效区域（`RowValid_`、`ColValid_`）

Tile 具有 **有效区域** `(valid_row, valid_col)`，定义哪些元素在本次操作中有意义：

- 若 `RowValid_ == Rows_` 且 `ColValid_ == Cols_`，有效区域完全静态。
- 若二者之一为 `pto::DYNAMIC`（`-1`），有效值存储在 Tile 对象中，并通过 `GetValidRow()` / `GetValidCol()` 查询。

对一个 Tile `t`，有效区域总是一个**连续前缀**：

- 有效索引满足 `0 <= i < t.GetValidRow()` 且 `0 <= j < t.GetValidCol()`。
- 有效区域外的元素除非指令显式定义 padding/行为，否则均为**未指定（unspecified）**。

通常指令语义以“对有效区域内每个元素”来解释。注意域（domain）可能因指令而异（例如某些指令以源 Tile 的有效区域定义语义域），请以 `docs/isa/*` 与 `docs/isa/conventions_zh.md` 为准。

### 布局（`BLayout`、`SLayout`、`SFractalSize`）

PTO 用两层布局描述 Tile：

- **基础布局** `BLayout`（`RowMajor`/`ColMajor`）：外层（未盒化）矩阵解释。
- **盒化/分形布局** `SLayout`（`NoneBox`、`RowMajor`、`ColMajor`）：是否将 Tile 在内部划分为固定大小的“基块”（base tile，亦常称 *fractals*）。
- **基块大小** `SFractalSize`：单个基块的字节大小。当前 PTO Tile Lib 常用：
  - `TileConfig::fractalABSize = 512` bytes（常用于 A/B 操作数 Tile）
  - `TileConfig::fractalCSize = 1024` bytes（常用于累加器 Tile）

#### 为什么需要盒化/分形布局

部分矩阵引擎偏好固定大小基块的访问/计算模式。显式表达盒化/分形布局可以：

- 让编译器尽早选择合法布局与形状。
- 避免运行时走慢速的“修正（fixup）”路径。
- 使同一份源码更容易映射到不同硬件代际（其微约束可能不同）。

#### 示例：512-byte 基块（示意）

当 `SFractalSize == 512` 且内层盒化布局为 row-major 时，常见基块形状示例：

- `fp32`：`16 × 8`  （16 * 8 * 4 bytes = 512 bytes）
- `fp16`：`16 × 16`（16 * 16 * 2 bytes = 512 bytes）
- `int8/fp8`：`16 × 32`（16 * 32 * 1 byte = 512 bytes）

若内层盒化布局为 col-major，同一基块可视为转置（例如 `8 × 16`、`16 × 16`、`32 × 16`）。

精确细节依赖后端；以 `pto::Tile` 的编译期检查与各指令页面的约束为准。

### Padding（`PadValue`）

`PadValue` 是编译期策略，用于部分实现处理有效区域外元素（例如 select/copy/pad）。其效果依赖指令与后端。

## 概念性约束（程序员模型）

除编译期布局检查外，PTO 程序通常依赖以下概念约束：

- Tile 是**二维**对象（矩阵形状）。
- Tile 是最小的调度/数据搬运单位：操作以 Tile 为单位消费/产生（而非子 Tile）。
- Tile 容量形状（`Rows`、`Cols`）意图保持**静态**；有效区域可静态或动态。
- 真实硬件常对 Tile 大小施加设备相关范围约束（常见量级为**数百字节到数十 KB**）。CPU 仿真更宽松，但可移植内核应遵守目标限制。

## 编译期约束（对齐与盒化）

`pto::Tile` 通过 `static_assert` 强制一组布局约束：

- 对**未盒化 row-major** Tile：`Cols * sizeof(Element)` 必须是 `TileConfig::alignedSize`（32 bytes）的整数倍。
- 对**未盒化 col-major** Tile：`Rows * sizeof(Element)` 必须是 32 bytes 的整数倍。
- 对**盒化** Tile：形状必须与 `(SLayout, SFractalSize)` 隐含的基块维度兼容（对部分 `Vec` Tile 有小范围例外）。

这些约束是有意为之：它们阻止生成在真实硬件上非法或低效的程序。

## 常用别名（Aliases）

`include/pto/common/pto_tile.hpp` 提供了矩阵乘相关的便捷别名：

- `pto::TileLeft<Element, Rows, Cols>`
- `pto::TileRight<Element, Rows, Cols>`
- `pto::TileAcc<Element, Rows, Cols>`

这些别名会为目标选择合适的盒化布局与分形大小。例如在 CPU 仿真后端：

- `TileLeft`：外层 col-major + 内层 row-major（常称 “Nz”）
- `TileRight`：外层 row-major + 内层 col-major（常称 “Zn”）
- `TileAcc`：使用 `TileConfig::fractalCSize` 的累加器布局

## 地址绑定（`TASSIGN`）

在手动放置（manual placement）流程中，`TASSIGN(tile, addr)` 会将 Tile 对象绑定到实现定义的地址。在 Auto 流程中，`TASSIGN(tile, addr)` 可能因构建配置而成为 no-op。

详情参见：`docs/isa/TASSIGN_zh.md`。

## 示例

### 基本向量 Tile

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT a, b, c;
  TADD(c, a, b);
}
```

### 静态有效区域（mask）

```cpp
using TileT = pto::Tile<pto::TileType::Vec, float, 128, 256,
                        pto::BLayout::RowMajor,
                        127 /*row_valid*/, 127 /*col_valid*/,
                        pto::SLayout::NoneBox, pto::TileConfig::fractalABSize,
                        pto::PadValue::Zero>;
```

### 动态有效区域（mask）

```cpp
using TileT = pto::Tile<pto::TileType::Vec, float, 128, 256,
                        pto::BLayout::RowMajor,
                        pto::DYNAMIC /*row_valid*/, 127 /*col_valid*/>;

TileT t(/*row_valid_runtime=*/m);
```
