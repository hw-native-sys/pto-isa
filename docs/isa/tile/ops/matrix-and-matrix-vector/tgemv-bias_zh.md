# TGEMV_BIAS

## 指令示意图

![TGEMV_BIAS tile operation](../../../../figures/isa/TGEMV_BIAS.svg)

## 简介

`TGEMV_BIAS` 表示“先做 GEMV，再并入 bias”。它仍然属于 cube 路径，只是运行时固定 `m = 1`。

这条指令的 bias 不是任意 shape 的普通 tile，而是单行 Bias tile。也正因此，它表达的是输出列上的偏置，而不是一般意义上的逐元素加法。

## 数学语义

设：

- `K = bMatrix.GetValidRow()`
- `N = bMatrix.GetValidCol()`

对 `0 <= j < N`：

$$ \mathrm{C}_{0,j} = \mathrm{Bias}_{0,j} + \sum_{k=0}^{K-1} \mathrm{A}_{0,k} \cdot \mathrm{B}_{k,j} $$

## 机制

`TGEMV_BIAS` 沿用 `TGEMV` 的 `Left` / `Right` / `Acc` 角色合同，再额外引入一块单行 `Bias` tile。由于输出本身只有一行，这里不需要像 `TMATMUL_BIAS` 那样再解释“按列向多行广播”；bias 的每一项直接对应结果中的一列。

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../../../../assembly/PTO-AS_zh.md)。

同步形式：

```text
%acc = tgemv.bias %a, %b, %bias : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%c = pto.tgemv.bias %a, %b, %bias : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tgemv.bias ins(%a, %b, %bias : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileRes, typename TileLeft, typename TileRight, typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_BIAS(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData,
                                WaitEvents &... events);

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename TileBias,
          typename... WaitEvents>
PTO_INST RecordEvent TGEMV_BIAS(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData,
                                WaitEvents &... events);
```

## 输入与输出

- `aMatrix`：左操作数 tile，必须是 `Left`。
- `bMatrix`：右操作数 tile，必须是 `Right`。
- `biasData`：单行 Bias tile。
- `cMatrix`：结果累加器 tile，必须是 `Acc`。

## 约束

### 通用约束

- `TGEMV` 的角色、shape、dtype 和 target-profile 约束在这里全部成立；
- bias 的数据类型必须与结果累加器 `TileRes::DType` 一致；
- bias 必须是单行 Bias tile。

### A2A3 约束

`A2A3` 指 Ascend 910B 与 Ascend 910C。当前实现要求：

- `TileBias::Loc == TileType::Bias`
- `TileBias::Rows == 1`

### A5 约束

`A5` 指 Ascend 950 PR 与 Ascend 950 DT。当前实现要求：

- `TileBias::Loc == TileType::Bias`
- `TileBias::Rows == 1`
- `TileBias::isRowMajor == true`

## 不允许的情形

- bias 不是单行；
- bias 的角色或 dtype 不合法；
- 违反 `TGEMV` 的任一合法性约束。

## 性能与吞吐

当前仓内 A2A3 costmodel 对 `TGEMV_BIAS` 仍复用 `mad/mmad` 的 GEMV 公式：

```text
cycles = 14 + ceil(N/16) * ceil(K / baskK) * repeat_cost
```

其中：

- `baskK = 32 / sizeof(left_element_type)`；
- int8、fp16 bucket 的 `repeat_cost = 1`；
- fp32 bucket 的 `repeat_cost = 2`。

当前仓库没有公开单列的 A5 latency / throughput 表。

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using A = TileLeft<half, 1, 16>;
  using B = TileRight<half, 16, 16>;
  using Bias = Tile<TileType::Bias, float, 1, 16>;
  using C = TileAcc<float, 1, 16>;
  A a;
  B b;
  Bias bias;
  C c;
  TGEMV_BIAS(c, a, b, bias);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using A = TileLeft<half, 1, 16>;
  using B = TileRight<half, 16, 16>;
  using Bias = Tile<TileType::Bias, float, 1, 16>;
  using C = TileAcc<float, 1, 16>;
  A a;
  B b;
  Bias bias;
  C c;
  TASSIGN(a, 0x1000);
  TASSIGN(b, 0x2000);
  TASSIGN(bias, 0x3000);
  TASSIGN(c, 0x4000);
  TGEMV_BIAS(c, a, b, bias);
}
```

## 相关页面

- [TGEMV](./tgemv_zh.md)
- [TGEMV_ACC](./tgemv-acc_zh.md)
- [TGEMV_MX](./tgemv-mx_zh.md)
- [矩阵与矩阵-向量指令集](../../matrix-and-matrix-vector_zh.md)
