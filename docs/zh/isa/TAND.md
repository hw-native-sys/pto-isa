# TAND

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

![TAND tile operation](../../figures/isa/TAND.svg)

## 简介

两个Tile的逐元素按位与。

## 数学语义

对每个元素 `(i, j)` 在有效区域内：

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \;\&\; \mathrm{src1}_{i,j} $$

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TAND(TileData &dst, TileData &src0, TileData &src1, WaitEvents &... events);
```

## 约束

- **实现检查 （Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品）**:
    - 支持的元素类型为1字节、2字节或4字节整数类型。
    - `dst`、`src0` 和 `src1` 必须使用相同的元素类型。
    - `dst`、`src0` 和 `src1` 必须是行主序。
    - 运行时：`src0.GetValidRow()/GetValidCol()` 和 `src1.GetValidRow()/GetValidCol()` 必须与 `dst` 一致。
- **实现检查 (Ascend 950PR/Ascend 950DT)**:
    - 支持的元素类型为1字节、2字节、4字节或8字节整数类型。
    - `dst`、`src0` 和 `src1` 必须使用相同的元素类型。
    - `dst`、`src0` 和 `src1` 必须是行主序。
    - 运行时：`src0.GetValidRow()/GetValidCol()` 和 `src1.GetValidRow()/GetValidCol()` 必须与 `dst` 一致。
- **有效区域**:
    - 该操作使用 `dst.GetValidRow()` / `dst.GetValidCol()` 作为迭代域。

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, int32_t, 16, 16>;
  TileT a, b, out;
  TAND(out, a, b);
}
```
