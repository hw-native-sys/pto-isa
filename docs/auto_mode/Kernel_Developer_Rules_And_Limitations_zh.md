
这个文档列出了一些kernel开发者使用auto模式的一些规则和限制。

不遵守这些规则可能导致以下任意后果：

1. 无法编译（可能是源码层面编译报错，或者编译器挂掉）
2. 结果错误（例如，精度问题）
3. 差劲的性能

# 1 - 控制流规则

复杂的控制流（尤其是循环）会给编译器的自动同步，尤其是涉及到double-buffering和不同pipe之间的精准同步带来挑战。
因为PTO AUTO编译器首先需要确保结果正确，在遇到复杂控制流的情况下会趋向保守，无法插入最优的同步指令导致性能受损。

## 1.1 - 隔离第一个和最后一个循环的迭代

任何单独隔离开循环的第一个和最后一个迭代的条件控制都需要使用固定形式的表达，来使编译器能够静态分析。这样能让AUTO模式编译器剥离开第一次和最后一次迭代，能显著简化自动同步的分析难度。

例如：

```cpp
for (int tile_id = 0; tile_id < total_tiles; tile_id++) {
    if (tile_id == 0) {
        TLOAD(srcTile, globalSrc);
    }
    ...
    if (tile_id == total_tiles-1) {
        TSTORE(globalDst, dstTile);
    }
}
```

## 1.2 - 循环中的循环不变条件需要被外提

对于一个在内层循环中的if语句，如果其是循环不变条件（其判断不依赖循环的induction variable），则其应该被外提到这个循环之外。

例如对下面的在内层循环中的if语句：

```cpp
for (int tile_id = 0; tile_id < total_tiles; tile_id++) {
    int next_tile = tile_id < total_tiles-1 ? tile_id + 1 : -1;
    ...
    for (int subtile_id = 0; subtile_id < total_subtiles; subtile_id++) {
        if (next_tile != -1) {
            ... // computation here
        }
    }
}
```

应该写成：

```cpp
for (int tile_id = 0; tile_id < total_tiles; tile_id++) {
    int next_tile = tile_id < total_tiles-1 ? tile_id + 1 : -1;
    ...
    if (next_tile != -1) {
        for (int subtile_id = 0; subtile_id < total_subtiles; subtile_id++) {
            ... // computation here
        }
    }
}
```

## 1.3 - 复杂的条件判断表达式

针对一个较复杂的条件判断逻辑，如果其用来判断PTO指令的执行与否，强烈建议将其统一用一个bool变量表达之后，再用此bool变量用作`if/else if`判断。

例如：

```cpp
if ((srcTile.GetValidRow() > 16 || srcTile.GetValidCol() > 16) && srcTile.GetKAligned()) {
    TLOAD(srcTile, globalSrc1);
}
else {
    TLOAD(srcTile, globalSrc0);
}
```

最好写成如下形式：

```cpp
bool cond = (srcTile.GetValidRow() > 16 || srcTile.GetValidCol() > 16) && srcTile.GetKAligned();

if (cond) {
    TLOAD(srcTile, globalSrc1);
}
else {
    TLOAD(srcTile, globalSrc0);
}
```

# 2 - 内存分配相关规则

AUTO模式下由于不能使用`TASSIGN`，编译器无法自动得知两个tile之间的alias关系，因为无法得知程序员的意图，所以需要程序员显式告诉编译器两个tile的alias关系。

## 2.1 使用`TRESHAPE`来告诉编译器两个tile拥有相同的首地址

如果你想表达两个tile必须具有相同的首地址，你可以使用`TRESHAPE`指令。这个指令manual和auto模式通用。例如：

```cpp
TileSrcTypeA tileA;
TileSrcTypeB tileB;

// Invalid in auto mode
TASSIGN(tileA, 0x0);
TASSIGN(tileB, 0x0);

// Correct in auto mode
TRESHAPE(tileB, tileA);
```

## 2.2 使用`TSUBVIEW`来告诉编译器tile B是tile A的一个subview

和`TRESHAPE`目的相同，但是用来表达tileB是tileA的一个subview。语义是，tileB的地址是在tileA的首地址基础上，加上一些rowOffset和colOffset而来。auto模式需要这个接口，是因为需要专门的接口来告诉编译器两个tile之间的alias关系。详见`docs/isa/TSUBVIEW_zh.md`。

示例:

```cpp
uint16_t rowOffset, colOffset; // can be runtime variable

// addr(tileB) = addr(tileA) + offsets
TileData tileA(...);
TileData tileB(...);

// Invalid in auto mode
TASSIGN(tileA, 0x0);
TASSIGN(tileB, 0x0 + rowOffset * TileData::Col + colOffset * 1 + sizeof(T));

// Correct for auto mode
TSUBVIEW(tileB, tileA, rowOffset, colOffset);
```

## 2.3 - 记住auto模式的重要编程思维：Tile一旦被定义了，其地址不能在运行时被改变

PTO AUTO编译器会为每一个定义的Tile变量自动分配内存。这个分配是一次性的，意思是一旦被编译器自动分配之后，就永远不会改变。
也就是说，当你在auto模式下编程时，如果你的代码逻辑依赖一个tile需要在运行时改变地址，那这样的代码就不能在auto模式下正确运行。例如，在manual模式下你可以这样：

```cpp
TileData tile;
for (int i = 0; i < N; i++) {
    TASSIGN(tile, 0x100 * i);
    foo(tile);
}
```

manual模式下程序员拥有完全的自由，可以在运行时的任何时间地点改变一个tile的地址，然而这在auto模式是不允许的，因为这样的动态性会给编译器的内存分配带来几乎不可能做到的巨大挑战，因此编译器的内存分配是一次性的、静态的。

因此，对于auto模式来说，一个至关重要的思维模式是：
**把每个tile想象成一个C++的引用，其在被定义的时候它们的内存就已经被绑定了，且永远不能再变。**

## 2.4 - 正确理解`TRESHAPE`和`TSUBVIEW`在auto模式下的语义

在manual模式下，这两个都是实际上的PTO指令：它们在内部都是直接调用`TASSIGN`。这意味着，就像上一条讲的，理论上程序员可以使用它们在任何时间地点来改变一个Tile的地址。
然而，在auto模式下，它们不是可执行的PTO指令，而只是单纯的对于编译器的提示：用来表达两个tile之间的alias关系用的。因此，它们不能用来改变tile的地址，所以如果你用`TRESHAPE`或者`TSUBVIEW`在同一个tile上重复用作输出，那是未定义行为，比如：

```cpp
TRESHAPE(tile0, tile1);
foo(tile0);
... 
TSUBVIEW(tile0, tile2, 0, 0);
bar(tile0);
```

同时，因为它们在auto模式下只是单纯的给编译器的hint，因此它们在源码中的位置并不太重要。然而为了避免困惑，还是建议将它们的调用放在紧接着输入和输出Tile的定义之后。

# 3 - 通用规则

## 3.1 - 不要在destination tile上调用`TLOAD`

如果一个Tile只会被用作输出（没有需要从GM copied-in的数据），那就不要调用`TLOAD`。
首先，这是没有必要的；
其次，在auto模式下可能造成数据踩踏。

比如：

```cpp
TLOAD(dstTile, dstGlobal); // redundant
TLOAD(srcTile, srcGlobal);

TEXP(dstTile, srcTile);

TSTORE(dstGlobal, dstTile);
```

在manual模式下，程序员可以给`srcTile`和`dstTile`手动分配不同的地址，这样的话没有任何问题。
但是，在auto模式下，这可能会有数据踩踏的问题：这里`srcTile`和`dstTile`的生命周期不重叠，因此编译器会给`dstTile`复用`srcTile`的地址。这种情况下，由于两个`TLOAD`在pipe中同时执行，就会造成数据踩踏。

## 3.2 - 不要在kernel中直接调用CCE intrinsic

kernel开发者应该只调用PTO指令，避免CCE intrinsics。两点原因：

1. CCE intrinsics的入参都是裸指针类型，这在auto模式下是无法编译的：`Tile` struct里的`TileDType`类型，在manual模式下定义是指针类型；然而在auto模式下，其定义是vector类型，无法暴露出指针。
2. PTO编译器的分析和优化都是在tile这个抽象层级进行的，无论是manual模式下的优化（比如tile fusion）和auto模式（自动同步和内存分配）都是如此；PTO编译器无法也不会识别CCE intrinsics，因此无法正确自动插入同步。

基于这个原因，kernel开发者应该避免调用`Tile::data()`成员函数；理论上讲，这个接口不是给kernel开发者用的，而只是给库开发者在tile function上使用的。

## 3.3 - 尽量使用`PtoSetWaitFlag`或者`TSYNC`而不是`set_flag`和`wait_flag`

在`PtoSetWaitFlag`和`TSYNC`的内部实现中，存在manual和auto模式的隔离：manual模式下是正常调用`set_flag`和`wait_flag`，而auto模式下是no-op，因此其不会对auto模式下编译器自动插入的同步产生冲突。如果kernel开发者直接调用`set_flag`和`wait_flag`，则他们需要手动使用`__PTO_AUTO__`宏隔离开auto模式，比较麻烦。
