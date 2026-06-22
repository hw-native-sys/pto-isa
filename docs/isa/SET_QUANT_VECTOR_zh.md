# SET_QUANT_VECTOR

## 简介

设置向量量化参数，用于后续 `TPUSH` 操作。通过将 Scaling 类型 Tile 的地址转换为量化参数地址格式并写入硬件 FPC 寄存器，完成量化配置。

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename FpTileData, typename... WaitEvents>
PTO_INST RecordEvent SET_QUANT_VECTOR(FpTileData &fpTile, WaitEvents &...events);
```

## 约束

- `FpTileData::Loc` 必须为 `TileType::Scaling`。仅支持 Scaling 类型的 Tile 作为输入。
- 该指令必须在消费此配置的 `TPUSH` 指令之前调用。
- Tile 地址编码到硬件 FPC 寄存器的具体方式属于实现定义行为。

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
AICORE void example_set_quant_vector()
{
    using ScalingTile = Tile<TileType::Scaling, T, 1, 128, BLayout::RowMajor, 1, 128>;
    ScalingTile fpTile;
    TASSIGN(fpTile, 0x0);

    SET_QUANT_VECTOR(fpTile);
}
```

## 汇编示例（ASM）

当前公开的汇编参考尚未为 `SET_QUANT_VECTOR` 定义稳定的 PTO-AS 写法。设置量化配置时请使用 C++ intrinsic 形式。
