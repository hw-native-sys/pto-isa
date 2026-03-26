# PTO AUTO模式

## auto模式是什么

PTO AUTO是一种新的编程模式，主要提供以下两点优势：

* 降低kernel的开发难度的同时能使开发者实现必要的优化
* 确保跨代兼容不同的昇腾硬件架构

更具体来说，在AUTO模式下，kernel开发者不用手动为tile分配内存，也不用亲自手写不同pipe间的同步。作为替代，PTO AUTO编译器会帮助kernel开发者在不同buffer上分配内存。而且，编译器也会在PTO指令之间自动插入同步，最大化pipe之间的流水线并行。最后，kernel开发者也不用关心不同昇腾硬件架构之间的区别（尤其是关于CUBE和VECTOR交流和同步的机制）。

## 简单示例

一个简单的示例：逐元素的乘法。这展示了最关键的auto模式与manual模式的区别：

### TMUL manual模式

```cpp
template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
__global__ AICORE void runTMul(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(kGRows_, kGCols_);
    TileData src1Tile(kGRows_, kGCols_);
    TileData dstTile(kGRows_, kGCols_);
    
    TASSIGN(src0Tile, 0x0 + 0x400 * block_idx);
    TASSIGN(src1Tile, 0x4000 + 0x400 * block_idx);
    TASSIGN(dstTile, 0x8000 + 0x400 * block_idx);

    int offset = (block_idx / 4) * (64 * 16) + (block_idx % 4) * 16;
    GlobalData src0Global(src0 + offset);
    GlobalData src1Global(src1 + offset);
    GlobalData dstGlobal(out + offset);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMUL(dstTile, src0Tile, src1Tile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    
    out = dstGlobal.data();
}
```

### TMUL AUTO模式

```cpp
template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
__global__ AICORE void runTMul(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    
    TileData src0Tile(kGRows_, kGCols_);
    TileData src1Tile(kGRows_, kGCols_);
    TileData dstTile(kGRows_, kGCols_);
    
    int offset = (block_idx / 4) * (64 * 16) + (block_idx % 4) * 16;
    GlobalData src0Global(src0 + offset);
    GlobalData src1Global(src1 + offset);
    GlobalData dstGlobal(out + offset);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    TMUL(dstTile, src0Tile, src1Tile);
    TSTORE(dstGlobal, dstTile);
    
    out = dstGlobal.data();
}
```

## PTO AUTO编译器特性  

### 不同架构之间的兼容性

PTO AUTO编译器确保同一个PTO源码程序能在不同昇腾架构上编译和运行，同时确保性能。

### 自动同步

在manual模式下，程序员需要使用PTO的Event机制，精准地在必要的源码位置上手动调用同步指令来确保正确的结果和性能。这非常繁琐且容易出错。

Auto模式编译器让程序员避免了这个麻烦。编译器会自动在需要的位置插入同步，确保正确的结果和较好的性能。

### Tile内存分配

在manual模式下，当用户声定义一个Tile变量后，需要显式调用`TASSIGN`来为这个Tile分配在指定内存空间里的内存地址。
在auto模式下，用户不需要手动分配内存，只需要定义Tile变量即可；编译器会自动为所有Tile在正确的buffer上分配内存地址。

# PTO AUTO文档

更多PTO AUTO的详细文档如下：

* [PTO_AUTO_kernel_developer_rules_and_limitations](kernel_developer_rules_and_limitations.md)
* [PTO_AUTO_library_developer_rules_and_limitations](library_developer_rules_and_limitations.md)
* [Compiling with AUTO Mode](PTO_AUTO_compilation_guide.md)
* [PTO AUTO Code Examples](Examples.md)
