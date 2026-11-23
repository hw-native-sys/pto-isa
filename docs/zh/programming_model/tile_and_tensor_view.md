# Tile 与 Tensor View

## 概述

PTO 把“全局张量对象”和“片上 tile 对象”明确分成两套模型。前者用于描述全局数据的逻辑视图，后者用于描述本地计算与搬运的真实工作对象。很多编译检查和自动分析都建立在这条分界线上。

## 四层数据对象

从上层编程模型看，用户在写一个 PTO kernel 时，通常会依次接触下面四层对象：

1. 全局地址入口
2. 全局张量视图
3. 张量局部分区
4. 本地 tile 对象

这四层对象对应的是同一份数据在不同阶段的表达方式，而不是四种完全独立的数据。

### 全局地址入口

在上层模型中，它表示 kernel 拿到的一段全局地址入口。

- C++ 表示：`__gm__` 指针（内核入口参数，如 `__gm__ float* in`）
- IR 表示：`!pto.ptr<T>`

它只表达“某类元素的全局地址”，不携带：

- shape
- stride
- 切片
- tile 布局

因此它更像 kernel 的原始输入或输出句柄。单独一个 `ptr` 还不足以表达“这是一块怎样的张量”。

### 全局张量视图

在上层模型中，它表示“把一段全局地址按张量方式来理解”。

- C++ 表示：`pto::GlobalTensor<Element, Shape, Stride, Layout>`
- IR 表示：`!pto.tensor_view<...>`

`tensor_view` 用于描述：

- 全局张量的逻辑 shape
- 与 shape 对应的 stride / layout 解释方式

从用户角度看，这一步是在说：

- 这块全局内存要按什么张量形状来理解
- 每一维应该怎样寻址

它仍然是全局对象，还不是本地计算对象。

### 张量局部分区

在上层模型中，它表示“从整张量中取出当前这次要处理的一块”。

- C++ 表示：同样是 `GlobalTensor`，通过指针偏移和子形状构造（C++ 侧没有独立的分区类型，第②③层共用同一类型）
- IR 表示：`!pto.partition_tensor_view<...>`

`partition_tensor_view` 进一步把一个全局张量视图收缩到逻辑分区。它表达的是：

- 某个局部子区域
- 这个子区域的 shape
- 它与上游全局张量之间的逻辑关系

从编程模型上看，这一步通常对应：

- 当前 block 只处理整张量中的一块
- 当前循环轮次只处理某个子块
- 当前一次 `TLOAD` / `TSTORE` 只对应其中一个局部区域

也就是说，它回答的是“这次要处理哪一块”。

### 本地 tile 对象

在上层模型中，它表示“已经进入片上、可直接参与计算的局部数据块”。

- C++ 表示：`pto::Tile<TileType::Loc, Element, Rows, Cols, ...>`
- IR 表示：`!pto.tile_buf<loc=..., dtype=..., rows=..., cols=..., ...>`

`tile_buf` 是本地 tile 对象，显式带有：

- 地址空间
- 元素类型
- 物理尺寸
- 有效尺寸
- 布局 / pad / fractal 等配置

从用户角度看，这已经不是“全局张量的一种看法”，而是：

- 已经进入片上
- 已经放到某个具体本地位置
- 可以直接参与搬运、逐元素、矩阵或规约计算

因此它是 PTO kernel 中最直接的工作对象。

## 典型使用路径

最常见的路径是：

| 步骤 | C++ | IR |
| --- | --- | --- |
| ① 全局地址 | `__gm__` 指针（内核参数） | `ptr` |
| ② 构造全局视图 | 构造 `GlobalTensor<...>` | `make_tensor_view` |
| ③ 切出分区 | 构造带偏移的 `GlobalTensor<...>` | `partition_view` |
| ④ 搬到片上 | `TLOAD` | `tload` / `tprefetch` |
| ⑤ 本地计算 | 在 `Tile<...>` 上 | 在 `tile_buf` 上 |
| ⑥ 写回 | `TSTORE` | `tstore` |

C++ 写法：

```cpp
using GT = pto::GlobalTensor<float,
                             pto::Shape<1,1,1,32,32>,
                             pto::Stride<1,1,1,32,1>>;
using TileT = pto::Tile<pto::TileType::Vec, float, 32, 32>;

GT gin(in);        // ①→② 从 __gm__ 指针构造全局视图
TileT t;           // ④ 本地 tile 对象
TLOAD(t, gin);     // ②→④ 从全局搬到片上
// ... 在 t 上做本地计算 ...
TSTORE(gout, t);   // ④→② 写回
```

IR 写法：

```mlir
%tv = pto.make_tensor_view %arg0,
  shape = [%c32, %c32],
  strides = [%c32, %c1]
  : !pto.tensor_view<?x?xf32>

%pv = pto.partition_view %tv,
  offsets = [%c0, %c0],
  sizes = [%c32, %c32]
  : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
```

这条路径可以直观理解为：

- 先说明“整块全局内存是什么张量”
- 再说明“这次取其中哪一块”
- 再把这一块搬到片上作为 tile 使用

## 为什么要区分 view 和 tile

如果不区分 view 和 tile，那么在上层模型上就会变得不清楚：

- 数据现在还在 GM，还是已经进入片上
- 当前操作是在“选一块数据”，还是在“真正搬数据”
- 当前对象描述的是逻辑张量区域，还是本地计算缓冲
- 后续操作应该接全局视图，还是接本地 tile

PTO 正是通过这两套对象模型，把这些阶段区别稳定保留下来，可以把它们粗略理解为：

- view 模型：关注“全局数据如何被理解和切分”
- tile 模型：关注“片上数据如何被放置和使用”

对于写 kernel 的用户，这个区分通常比类型细节更重要。

## 编程上的含义

### 先想清楚数据现在处于哪一层

写 PTO ISA 时，最先要判断的是：

- 现在手里的是全局地址
- 全局张量视图
- 张量分区
- 还是本地 tile

不同层级对应的合法操作并不一样。

### 全局切块和本地计算要分开表达

全局切块通常通过构造全局张量视图（可带偏移）完成；本地计算通常发生在本地 tile 对象上。把这两步分开写，程序会更清楚。

### 让数据流路径保持直观

对大多数 kernel，最稳定的写法就是：

1. 从 `ptr` 构造 `tensor_view`
2. 从 `tensor_view` 切出 `partition_tensor_view`
3. 用 `TLOAD` 把数据搬到本地 tile
4. 在 `tile_buf` 上完成局部计算
5. 用 `TSTORE` 写回

越是保持这条路径清楚，后续阅读、调试和维护就越直接。
