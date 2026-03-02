# TMATMUL

## 指令示意图

![TMATMUL tile operation](../figures/isa/TMATMUL.svg)

## 简介

矩阵乘法 (GEMM)，生成累加器/输出 Tile。

## 数学语义

Let:

- `M = aMatrix.GetValidRow()`
- `K = aMatrix.GetValidCol()`
- `N = bMatrix.GetValidCol()`

For `0 <= i < M` and `0 <= j < N` (output elements in the effective matmul domain):

$$ \mathrm{C}_{i,j} = \sum_{k=0}^{K-1} \mathrm{A}_{i,k} \cdot \mathrm{B}_{k,j} $$

Exact accumulator behavior and datatype promotion are target/implementation-defined.

## 汇编语法

PTO-AS 形式：参见 [docs/grammar/PTO-AS_zh.md](../grammar/PTO-AS_zh.md)。

同步形式：

```text
%acc = tmatmul %a, %b : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### IR Level 1（SSA）

```text
%c = pto.tmatmul %a, %b : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### IR Level 2（DPS）

```text
pto.tmatmul ins(%a, %b : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, WaitEvents&... events);
```

## 约束

- **实现检查 (A2A3)**:
  - Supported `(CType, AType, BType)` triples:
    - `(int32_t, int8_t, int8_t)`
    - `(float, half, half)`
    - `(float, float, float)`
    - `(float, bfloat16_t, bfloat16_t)`
  - Static shape constraints: `TileLeft::Rows == TileRes::Rows`, `TileLeft::Cols == TileRight::Rows`, `TileRight::Cols == TileRes::Cols`.
  - Tile locations: `TileLeft::Loc == Left`, `TileRight::Loc == Right`, `TileRes::Loc == Acc`.
  - Runtime: `m/k/n` (taken from `aMatrix.GetValidRow()`, `aMatrix.GetValidCol()`, `bMatrix.GetValidCol()`) must be in `[1, 4095]`.
- **实现检查 (A5)**:
  - Accumulator type must be `int32_t` or `float`.
    - If `int32_t`: `AType == int8_t` and `BType == int8_t`.
    - If `float`: supports `half/bfloat16_t/float` and selected fp8 pairs (target-defined).
  - Static shape constraints: `TileLeft::Rows == TileRes::Rows`, `TileLeft::Cols == TileRight::Rows`, `TileRight::Cols == TileRes::Cols`.
  - Fractal/layout constraints are enforced:
    - Left: `Loc == Left`, `!isRowMajor`, `SFractal == RowMajor`
    - Right: `Loc == Right`, `isRowMajor`, `SFractal == ColMajor`
    - Acc: `Loc == Acc`, `!isRowMajor`, `SFractal == RowMajor`
    - Runtime: `m/k/n` (taken from `aMatrix.GetValidRow()`, `aMatrix.GetValidCol()`, `bMatrix.GetValidCol()`) must be in `[1, 4095]`.

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using A = TileLeft<half, 16, 16>;
  using B = TileRight<half, 16, 16>;
  using C = TileAcc<float, 16, 16>;
  A a;
  B b;
  C c;
  TMATMUL(c, a, b);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using A = TileLeft<half, 16, 16>;
  using B = TileRight<half, 16, 16>;
  using C = TileAcc<float, 16, 16>;
  A a;
  B b;
  C c;
  TASSIGN(a, 0x1000);
  TASSIGN(b, 0x2000);
  TASSIGN(c, 0x3000);
  TMATMUL(c, a, b);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%c = pto.tmatmul %a, %b : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%c = pto.tmatmul %a, %b : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO 汇编形式

```text
%acc = tmatmul %a, %b : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
# IR Level 2 (DPS)
pto.tmatmul ins(%a, %b : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c : !pto.tile_buf<...>)
```

