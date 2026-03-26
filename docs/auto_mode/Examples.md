# Auto Mode Examples

## Scope

This document shows examples of kernels written in auto mode versus in manual mode.


## TADD

```cpp

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

AICORE void runTAdd(__gm__ float __out__ *out, __gm__ float __in__ *src0, __gm__ float __in__ *src1) {
    using DynShapeDim5 = Shape<1, 1, 1, 64, 64>;
    using DynStridDim5 = Stride<1, 1, 1, 64, 1>;
    using GlobalData = GlobalTensor<float, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor, 64, 64>;
    TileData src0Tile(64, 64);
    TileData src1Tile(64, 64);
    TileData dstTile(64, 64);

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);

    TADD(dstTile, src0Tile, src1Tile);

    TSTORE(dstGlobal, dstTile);
}

```

Whereas in manual mode, it would look like this:

```cpp

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

AICORE void runTAdd(__gm__ float __out__ *out, __gm__ float __in__ *src0, __gm__ float __in__ *src1) {
    using DynShapeDim5 = Shape<1, 1, 1, 64, 64>;
    using DynStridDim5 = Stride<1, 1, 1, 64, 1>;
    using GlobalData = GlobalTensor<float, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor, 64, 64>;
    TileData src0Tile(64, 64);
    TileData src1Tile(64, 64);
    TileData dstTile(64, 64);

    /* TAssign only in manual mode */
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x10000);
    TASSIGN(dstTile, 0x20000);

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);

    /* event model only in manual mode */
    Event<Op::TLOAD, Op::TADD> event0;
    Event<Op::TADD, Op::TSTORE_VEC> event1;

    TLOAD(src0Tile, src0Global);
    event0 = TLOAD(src1Tile, src1Global);
    event1 = TADD(dstTile, src0Tile, src1Tile, event0);
    TSTORE(dstGlobal, dstTile, event1);
}

```

## TMATMUL

```cpp

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

template <typename cType, typename aType, typename bType, typename fbType, typename l0cType, int M, int K, int N,
          int ValidM, int ValidK, int ValidN>
__global__ AICORE void runTMatMul(__gm__ cType *out, __gm__ aType *src0, __gm__ bType *src1, __gm__ fbType *src2)
{
    using GlobalDataSrc0 = GlobalTensor<aType, pto::Shape<1, 1, 1, ValidM, ValidK>,
                                        pto::Stride<ValidM * ValidK, ValidM * ValidK, ValidM * ValidK, ValidK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<bType, pto::Shape<1, 1, 1, ValidK, ValidN>,
                                        pto::Stride<ValidK * ValidN, ValidK * ValidN, ValidK * ValidN, ValidN, 1>>;
    using GlobalDataSrc2 =
        GlobalTensor<fbType, pto::Shape<1, 1, 1, 1, ValidN>, pto::Stride<ValidN, ValidN, ValidN, ValidN, 1>>;
    using GlobalDataOut = GlobalTensor<cType, pto::Shape<1, 1, 1, ValidM, ValidN>,
                                       pto::Stride<ValidM * ValidN, ValidM * ValidN, ValidM * ValidN, ValidN, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, aType, M, K, BLayout::ColMajor, ValidM, ValidK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, bType, K, N, BLayout::ColMajor, ValidK, ValidN, SLayout::RowMajor, 512>;
    using TileMatFbData = Tile<TileType::Mat, fbType, 1, N, BLayout::RowMajor, 1, ValidN, SLayout::NoneBox>;

    using LeftTile = TileLeft<aType, M, K, ValidM, ValidK>;
    using RightTile = TileRight<bType, K, N, ValidK, ValidN>;
    using AccTile = TileAcc<l0cType, M, N, ValidM, ValidN>;

    using FbTile = Tile<TileType::Scaling, fbType, 1, N, BLayout::RowMajor, 1, ValidN, SLayout::NoneBox>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileMatFbData fbMatTile;

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    FbTile fbTile;

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    TLOAD(fbMatTile, src2Global);

    /**************************TMOV & TMATMUL**************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
    TMATMUL(cTile, aTile, bTile);
    TMOV(fbTile, fbMatTile);

    /********************************TSTORE****************************/
    TSTORE_FP<AccTile, GlobalDataOut, FbTile>(dstGlobal, cTile, fbTile);
}
```

Versus in manual mode:

```cpp

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

template <typename cType, typename aType, typename bType, typename fbType, typename l0cType, int M, int K, int N,
          int ValidM, int ValidK, int ValidN>
__global__ AICORE void runTMatMul(__gm__ cType *out, __gm__ aType *src0, __gm__ bType *src1, __gm__ fbType *src2)
{
    using GlobalDataSrc0 = GlobalTensor<aType, pto::Shape<1, 1, 1, ValidM, ValidK>,
                                        pto::Stride<ValidM * ValidK, ValidM * ValidK, ValidM * ValidK, ValidK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<bType, pto::Shape<1, 1, 1, ValidK, ValidN>,
                                        pto::Stride<ValidK * ValidN, ValidK * ValidN, ValidK * ValidN, ValidN, 1>>;
    using GlobalDataSrc2 =
        GlobalTensor<fbType, pto::Shape<1, 1, 1, 1, ValidN>, pto::Stride<ValidN, ValidN, ValidN, ValidN, 1>>;
    using GlobalDataOut = GlobalTensor<cType, pto::Shape<1, 1, 1, ValidM, ValidN>,
                                       pto::Stride<ValidM * ValidN, ValidM * ValidN, ValidM * ValidN, ValidN, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, aType, M, K, BLayout::ColMajor, ValidM, ValidK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, bType, K, N, BLayout::ColMajor, ValidK, ValidN, SLayout::RowMajor, 512>;
    using TileMatFbData = Tile<TileType::Mat, fbType, 1, N, BLayout::RowMajor, 1, ValidN, SLayout::NoneBox>;

    using LeftTile = TileLeft<aType, M, K, ValidM, ValidK>;
    using RightTile = TileRight<bType, K, N, ValidK, ValidN>;
    using AccTile = TileAcc<l0cType, M, N, ValidM, ValidN>;

    using FbTile = Tile<TileType::Scaling, fbType, 1, N, BLayout::RowMajor, 1, ValidN, SLayout::NoneBox>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileMatFbData fbMatTile;

    /* TAssign only in manual mode */
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);
    TASSIGN(fbMatTile, 0x20000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    FbTile fbTile;

    /* TAssign only in manual mode */
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(fbTile, 0x0);

    /* event model only in manual mode */
    Event<Op::TLOAD, Op::TMOV_M2L> evtLoad_Mov;
    Event<Op::TMOV_M2B, Op::TMATMUL> evtMov_Matmul;
    Event<Op::TMATMUL, Op::TMOV_M2S> evtMatmul_MovM2s;

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    evtLoad_Mov = TLOAD(fbMatTile, src2Global);

    /**************************TMOV & TMATMUL**************************/
    TMOV(aTile, aMatTile, evtLoad_Mov);
    evtMov_Matmul = TMOV(bTile, bMatTile);
    evtMatmul_MovM2s = TMATMUL(cTile, aTile, bTile, evtMov_Matmul);
    TMOV(fbTile, fbMatTile, evtMatmul_MovM2s);

    /********************************TSTORE****************************/
    TSTORE_FP<AccTile, GlobalDataOut, FbTile>(dstGlobal, cTile, fbTile);
}
```