# TXORS

## 产品支持情况<a name="section73648168211"></a>

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：支持
<!-- end id3 -->


## 指令示意图

![TXORS tile operation](../../figures/isa/TXORS.svg)

## 简介

Tile与标量的逐元素按位异或。

## 数学语义

对每个元素 `(i, j)` 在有效区域内：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \oplus \mathrm{scalar} $$

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TXORS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar, TileDataTmp &tmp, WaitEvents &... events);
```

## 约束

- **实现检查 （Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品）**:
    - 支持的元素类型为 `uint8_t`、`int8_t`、`uint16_t`、`int16_t`、`uint32_t`、`int32_t`。
    - `dst`、`src` 和 `tmp` 必须使用相同的元素类型。
    - 在手动模式下，源、目标和临时存储的内存区域不得重叠。
- **实现检查 (Ascend 950PR/Ascend 950DT)**:
    - 支持的元素类型为 `uint8_t`、`int8_t`、`uint16_t`、`int16_t`、`uint32_t`、`int32_t`、`int64_t`、`uint64_t`。
    - `dst` 和 `src` 的元素类型必须一致。
    - `src.GetValidRow()/GetValidCol()` 必须与 `dst` 一致。
- **有效区域**:
    - 该操作使用 `dst.GetValidRow()` / `dst.GetValidCol()` 作为迭代域。

## 临时空间

### Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品

`tmp` **被使用**作为标量XOR分解的中间暂存存储。`tmp` 必须与 `dst` 具有相同的元素类型和有效形状。

### Ascend 950PR/Ascend 950DT

`tmp` 被接口接受但Ascend 950PR/Ascend 950DT实现**不使用**。Ascend 950PR/Ascend 950DT后端使用 `vxor` 向量指令配合广播标量寄存器，不需要暂存Tile存储。`tmp` 仅为了与Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品的API兼容性而保留在C++内建接口签名中。

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileDst = Tile<TileType::Vec, uint32_t, 16, 16>;
  using TileSrc = Tile<TileType::Vec, uint32_t, 16, 16>;
  using TileTmp = Tile<TileType::Vec, uint32_t, 16, 16>;
  TileDst dst;
  TileSrc src;
  TileTmp tmp;
  TXORS(dst, src, 0x1u, tmp);
}
```
