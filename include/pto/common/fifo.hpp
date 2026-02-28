/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_FIFO_HPP
#define PTO_FIFO_HPP
#include <type_traits>

using namespace std;

namespace pto {

// FIFO definitons
enum class FIFOType : uint8_t
{
    GM_FIFO = 0,  // FIFO implemented in Global Memory
    VEC_FIFO = 1, // FIFO implemented in Vector core's local memory
    MAT_FIFO = 2, // FIFO implemented in Cube core's local memory (e.g., L1)
};

enum class VecCubeRatio : uint8_t
{
    V1C1_VEC0 = 0, // 1 Vector core : 1 Cube core, Vec core only used Vector 0
    V1C1_VEC1 = 1, // 1 Vector core : 1 Cube core, Vec core only used Vector 1
    V2C1_VECS = 2, // 2 Vector cores : 1 Cube core
};

template <typename DataType, FIFOType FifoType, int Depth, int Period, typename Enable = void>
struct DataFIFO;

// 1. Specialization for GM FIFO
template <FIFOType T>
struct IsGMFiFo {
    static constexpr bool value = (T == FIFOType::GM_FIFO);
};

template <typename DataType, FIFOType FifoType, int Depth, int Period>
struct DataFIFO<DataType, FifoType, Depth, Period, typename std::enable_if<IsGMFiFo<FifoType>::value>::type> {
    static constexpr int fifoDepth = Depth;
    static constexpr int fifoPeriod = Period;
    static constexpr FIFOType fifoType = FifoType;
    using DType = DataType;

    __gm__ DataType *fifoBase;

    PTO_INTERNAL DataFIFO(__gm__ DataType *ptr) : fifoBase(ptr)
    {}

    __gm__ DataType *getBasePtr()
    {
        return fifoBase;
    }
};

// 2. Specialization for VEC_FIFO and MAT_FIFO (both use local memory)
template <FIFOType T>
struct IsTileFiFo {
    static constexpr bool value = (T == FIFOType::VEC_FIFO) || (T == FIFOType::MAT_FIFO);
};

template <typename DataType, FIFOType FifoType, int Depth, int Period>
struct DataFIFO<DataType, FifoType, Depth, Period, typename std::enable_if<IsTileFiFo<FifoType>::value>::type> {
    static constexpr int fifoDepth = Depth;
    static constexpr int fifoPeriod = Period;
    static constexpr FIFOType fifoType = FifoType;

    DataType *tilePtr;

    // Constructor for Pointer
    PTO_INTERNAL DataFIFO(DataType *ptr) : tilePtr(ptr)
    {}

    // Constructor for Reference/Tile
    PTO_INTERNAL DataFIFO(DataType &tile)
    {
        tilePtr = reinterpret_cast<DataType *>(tile.data());
    }
};

} // namespace pto
#endif