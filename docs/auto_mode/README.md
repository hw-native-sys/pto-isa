# PTO AUTO Mode

## What is PTO AUTO

PTO AUTO is a programming mode for PTO that provides two major benefits:

* Simpify developing efficient PTO code while providing kernel developers with the mechanisms that are necessary to implement their optimizations.
* Compatibility across different generations of the Ascend architecture.

More specifically, in PTO AUTO, the kernel developer does not need to explicitly specify tile memory addresses or synchronization between different pipes. Instead the PTO AUTO compiler automatically allocates optimal memory addressess for the tiles in different chip buffers. Moreover, the compiler automatically synchronizes the PTO tile operations in order to maximize parallelism among different pipes. Finally, the kernel developer does not need to be concerned with the minor differences between various generations of the Ascend architecture (particulary in terms of the way Cube and Vector computations are coordinated).

Note: auto mode currently only supports the compiler `-O2` option.

## Simple Example

A simple example, elementwise multiplication demonstrates the key differences between the PTO AUTO and manual modes:

### TMUL Manual Mode

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

### TMUL AUTO Mode

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

## PTO AUTO Compiler Features

### Cross-Architecture Compatibility

PTO AUTO Compiler ensures a single source PTO program can be compiled for different Ascend architecture generations without requiring any source-level modifications while maintaining performance.

### Automatic Synchronization

In manual mode, user would normally have to keep track of the asynchronous nature of the hardware by using PTO's [`event model`](../coding/Event.md) at precise code locations in order to ensure both functional correctness and high performance in execution. This might be tedious and error prone.

Auto mode compilation will allow users to avoid having to use the event model to synchronize their code. The compiler will automatically determine the locations to insert synchronization under the hood - ensuring functional correctness and competitive performance.

### Tile Memory Allocation

In the default mode of PTO compilation, after instantiating `Tile` variables, we would need to complement them with a `TASSIGN` instruction to manually assign a dedicated buffer address that it operates on. However in auto mode, this is not required anymore. By simply instantiating the `Tile` variable the compiler will automatically allocate the buffer addresses under the hood for the user.

## PTO AUTO Documents

More detailed documentations of the PTO AUTO programming and compilations are organized into the following documents.

* [PTO_AUTO_kernel_developer_rules_and_limitations](Kernel_Developer_Rules_And_Limitations_zh.md)
* [PTO_AUTO_library_developer_rules_and_limitations](Library_Developer_Rules_And_Limitations.md)
* [PTO AUTO Code Examples](Examples.md)
