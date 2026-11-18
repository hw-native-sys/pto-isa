# TRESHAPE

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

![TRESHAPE tile operation](../figures/isa/TRESHAPE.svg)

## 简介

将Tile重新解释为另一种Tile类型/形状，同时保留底层字节。

## 数学语义

除非另有说明，语义定义在有效区域上，目标相关行为标记为实现定义。

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TRESHAPE(TileDataOut &dst, TileDataIn &src, WaitEvents &... events);
```

## 约束

由 `TRESHAPE_IMPL` 强制执行：

- **Tile类型必须匹配**：`TileDataIn::Loc == TileDataOut::Loc`。
- **总字节大小必须匹配**。CPU 模拟和 NPU 手动模式均比较 `TileDataIn::StorageBytes` 和 `TileDataOut::StorageBytes`，计入元素打包和面板填充。
- **不允许boxed/non-boxed转换**：
    - 不能在 `SLayout::NoneBox` 与boxed布局之间进行reshape。

## 备注

- **CPU模拟**：`dst` 与 `src` 共享底层存储。通过任一视图写入的数据都可从另一视图读到。
- **Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品**：实现为别名（`TASSIGN_IMPL(dst, src.data())`），因此 `dst` 和 `src` 引用同一底层存储。

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using Src = Tile<TileType::Vec, float, 16, 16>;
  using Dst = Tile<TileType::Vec, float, 8, 32>;
  static_assert(Src::Numel == Dst::Numel);

  Src src;
  Dst dst;
  TRESHAPE(dst, src);
}
```
