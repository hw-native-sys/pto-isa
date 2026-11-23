# TGATHERB

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

![TGATHERB tile operation](../../figures/isa/TGATHERB.svg)

## 简介

按字节偏移从源 Tile 收集数据。`offset` 的每个元素是 `uint32_t`，表示相对 `src` 起始地址的字节偏移；每个偏移从该地址拷贝 32 字节到 `dst`。`dst` 的有效区域按 32 字节分块，每块对应一个偏移。元素大小为 1、2 或 4 字节，硬件通过 `vgatherb` 完成收集。

## 数学语义

对有效区域内的每个元素：

$$ \mathrm{dst}_{i,j} = *\left(\mathrm{srcBase} + \mathrm{offset}_{i,j}\right) $$

精确的边界行为由实现定义。

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <typename TileDataDst, typename TileDataSrc, typename TileDataOffset, typename... WaitEvents>
PTO_INST RecordEvent TGATHERB(TileDataDst &dst, TileDataSrc &src, TileDataOffset &offset, WaitEvents &... events);
```

## 约束

- **实现检查 （Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品）**:
    - 目标布局必须是行主序（`TileDataDst::isRowMajor`）。
    - 目标元素大小必须是 `1`、`2` 或 `4` 字节（通过辅助函数中的 `static_assert` 强制执行）。
    - `TileDataSrc::DType`/`TileDataDst::DType` 必须是 `int8_t` 或 `uint8_t` 或 `int16_t` 或 `uint16_t` 或 `int32_t` 或 `uint32_t` 或 `half` 或 `bfloat16_t` 或 `float`。
- **实现检查 (Ascend 950PR/Ascend 950DT)**:
    - 目标元素大小必须是 `1`、`2` 或 `4` 字节。
    - `TileDataSrc::DType`/`TileDataDst::DType` 必须是 `int8_t` 或 `uint8_t` 或 `int16_t` 或 `uint16_t` 或 `int32_t` 或 `uint32_t` 或 `half` 或 `bfloat16_t` 或 `float`。
- **偏移量解释**:
    - 偏移量被实现解释为 `uint32_t` 值（字节偏移量）。
    - 偏移量边界不通过显式运行时断言进行验证；超出范围的偏移量由目标定义。

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, uint8_t, 1, 256>;
  using OffT = Tile<TileType::Vec, uint32_t, 1, 256>;
  using DstT = Tile<TileType::Vec, uint8_t, 1, 256>;
  SrcT src;
  OffT off;
  DstT dst;
  TGATHERB(dst, src, off);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, uint8_t, 1, 256>;
  using OffT = Tile<TileType::Vec, uint32_t, 1, 256>;
  using DstT = Tile<TileType::Vec, uint8_t, 1, 256>;
  SrcT src;
  OffT off;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(off, 0x2000);
  TASSIGN(dst, 0x3000);
  TGATHERB(dst, src, off);
}
```
