# pto.taxpy

`pto.taxpy` 属于[矩阵与矩阵向量指令](./tile/matrix-and-matrix-vector_zh.md)集。

## 概述

AXPY 风格融合更新：将 Tile 乘以标量并累加到目标 Tile。TAXPY 计算 `dst = dst + alpha * src`，其中 alpha 为标量，src 为源 Tile。

## 机制

### 数学语义

对有效区域中的每个元素 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{dst}_{i,j} + \alpha \cdot \mathrm{src}_{i,j} $$

其中 $\alpha$ 为标量系数。行为按目标 valid region 定义。

## 语法

### PTO-AS

参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

### AS Level 1（SSA）

```mlir
%dst = pto.taxpy %src, %alpha : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.taxpy ins(%src, %alpha : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`。

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `src` | 输入 | 源 Tile |
| `alpha` | 标量 | 缩放系数 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `dst` | Tile（累加器） | 原地更新：$dst = dst + \alpha \cdot src$ |

## 副作用

`dst` 原地被修改。

## 约束

- 数据类型、layout、location 和 shape 的进一步限制以对应 backend 的合法性检查为准
- `dst` 和 `src` 的 valid region 必须兼容
- 标量 `alpha` 的类型必须与 tile 元素类型匹配或可隐式转换

## 异常与非法情形

- 若 `dst` 和 `src` 的 valid region 不匹配，行为未定义
- 若 `alpha` 类型不兼容，编译失败

## 示例

### C++

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT dst, src;
  float alpha = 2.5f;
  TAXPY(dst, src, alpha);
}
```

### PTO-AS

```mlir
# 自动模式
%dst = pto.taxpy %src, %alpha : (!pto.tile<f32, 16, 16>, f32) -> !pto.tile<f32, 16, 16>
```
