# TFMODS

## 指令示意图

![TFMODS tile operation](../figures/isa/TFMODS.svg)

## 简介

与标量的逐元素余数：`fmod(src, scalar)`。

## 数学语义

对每个元素 `(i, j)` 在有效区域内：

$$\mathrm{dst}_{i,j} = \mathrm{fmod}(\mathrm{src}_{i,j}, \mathrm{scalar})$$

## 汇编语法

PTO-AS 形式：参见 [docs/grammar/PTO-AS_zh.md](../grammar/PTO-AS_zh.md)。

同步形式：

```text
%dst = tfmods %src, %scalar : !pto.tile<...>, f32
```

### IR Level 1（SSA）

```text
%dst = pto.tfmods %src, %scalar : !pto.tile<...>, f32
```

### IR Level 2（DPS）

```text
pto.tfmods ins(%src, %scalar : !pto.tile_buf<...>, f32) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TFMODS(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar, 
                            WaitEvents&... events);
```

## 约束

- Division-by-zero behavior is target-defined; the CPU simulator asserts in debug builds.
- The op iterates over `dst.GetValidRow()` / `dst.GetValidCol()`.

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  TFMODS(out, x, 3.0f);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tfmods %src, %scalar : !pto.tile<...>, f32
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tfmods %src, %scalar : !pto.tile<...>, f32
```

### PTO 汇编形式

```text
%dst = tfmods %src, %scalar : !pto.tile<...>, f32
# IR Level 2 (DPS)
pto.tfmods ins(%src, %scalar : !pto.tile_buf<...>, f32) outs(%dst : !pto.tile_buf<...>)
```

