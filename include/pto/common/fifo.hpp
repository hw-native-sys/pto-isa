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

enum TileSplitAxis : uint8_t
{
    TILE_NO_SPLIT = 0,   // 1:1 mode, no split, using AIV0
    TILE_UP_DOWN = 1,    // Split/combine along rows: AIV0=upper half, AIV1=lower half
    TILE_LEFT_RIGHT = 2, // Split/combine along cols: AIV0=left half, AIV1=right half
};

enum Direction : uint8_t
{
    UNDEFINED = 0,
    DIR_C2V = 1,                  // Cube → Vector: Cube is producer, Vector is consumer
    DIR_V2C = 2,                  // Vector → Cube: Vector is producer, Cube is consumer
    DIR_BOTH = DIR_C2V | DIR_V2C, // Support both directions
    DIR_V2C_CTRL = 4,
#ifdef PTO_NPU_ARCH_A5
    DIR_C2V_GM = 5,
    DIR_V2C_GM = 6,
    DIR_BOTH_GM = DIR_C2V_GM | DIR_V2C_GM,
#endif
};

enum class PipeChunk : uint8_t
{
    Single = 0, // Full slot in one call: entry sync + exit sync + advance slot
    First = 1,  // First partial chunk: entry sync, keep current slot open
    Middle = 2, // Middle partial chunk: no sync, keep current slot open
    Last = 3,   // Final partial chunk: exit sync + advance slot
};

template <int SlotSize, int SlotNum, int LocalSlotNum>
struct RingFIFO {
    __gm__ void *GM_SLOT_BUFFER = nullptr; // Global memory
    uint32_t C2V_CONSUMER_BUF = 0x0;       // UB buffer
    uint32_t V2C_CONSUMER_BUF = 0x0;       // L1 buffer
    uint64_t V2C_CONTROL_BUF = 0x0;        // scalar buffer for control signals
    static constexpr uint32_t SLOT_SIZE = SlotSize;
    static constexpr uint32_t SLOT_NUM = SlotNum;
    static constexpr uint32_t SLOT_SYNCT = SlotNum;
    static constexpr uint32_t LOCAL_SLOT_NUM = LocalSlotNum;

    PTO_INTERNAL RingFIFO(__gm__ void *gmSlotBuf, uint32_t c2vConsBuf, uint32_t v2cConBuf)
        : GM_SLOT_BUFFER(gmSlotBuf), C2V_CONSUMER_BUF(c2vConsBuf), V2C_CONSUMER_BUF(v2cConBuf)
    {}
};

//---------------------------------------------------
enum class FIFOType : uint8_t
{
    GM_FIFO = 0,   // FIFO implemented in Global Memory
    VEC_FIFO = 1,  // FIFO implemented in Vector core's local memory
    MAT_FIFO = 2,  // FIFO implemented in Cube core's local memory (e.g., L1)
    CTRL_FIFO = 3, // FIFO for control signals, implemented in scalar buffer
};

enum class VecCubeRatio : uint8_t
{
    V1C1_VEC0 = 0, // 1 Vector core : 1 Cube core, Vec core only used Vector 0
    V1C1_VEC1 = 1, // 1 Vector core : 1 Cube core, Vec core only used Vector 1
    V2C1_VECS = 2, // 2 Vector cores : 1 Cube core
};

template <typename DataType, FIFOType FifoType, int Depth, int Period, int LocalDepth = 0, typename Enable = void>
struct DataFIFO;

// 1. GM FIFO
template <FIFOType T>
struct IsGMFiFo {
    static constexpr bool value = (T == FIFOType::GM_FIFO);
};

template <typename DataType, FIFOType FifoType, int Depth, int Period, int LocalDepth>
struct DataFIFO<DataType, FifoType, Depth, Period, LocalDepth,
                typename std::enable_if<IsGMFiFo<FifoType>::value>::type> {
    static constexpr int fifoDepth = Depth;
    static constexpr int fifoPeriod = Period;
    static constexpr FIFOType fifoType = FifoType;
    using DType = DataType;
    // base address of the GM FIFO buffer in global memory
    __gm__ DataType *fifoBase = nullptr;
    // base address of local FIFO buffer in local memory
    uint32_t localFiFoBase = 0;
    static constexpr int localFiFoDepth = LocalDepth;
    static constexpr bool useLocalFiFo = (localFiFoDepth > 0);

    PTO_INTERNAL DataFIFO(__gm__ DataType *ptr, uint32_t localBase) : fifoBase(ptr), localFiFoBase(localBase)
    {}

    __gm__ DataType *getBasePtr()
    {
        return fifoBase;
    }
};

// 2. VEC_FIFO and MAT_FIFO (both use local memory)
template <FIFOType T>
struct IsTileFiFo {
    static constexpr bool value = (T == FIFOType::VEC_FIFO) || (T == FIFOType::MAT_FIFO);
};

template <typename DataType, FIFOType FifoType, int Depth, int Period, int LocalDepth_unused>
struct DataFIFO<DataType, FifoType, Depth, Period, LocalDepth_unused,
                typename std::enable_if<IsTileFiFo<FifoType>::value>::type> {
    static constexpr int fifoDepth = Depth;
    static constexpr int fifoPeriod = Period;
    static constexpr FIFOType fifoType = FifoType;

    uint32_t fifoBase = 0; // fifo base address in local memory

    PTO_INTERNAL DataFIFO(uint32_t base) : fifoBase(base)
    {}
};

template <FIFOType T>
struct IsCtrlFiFo {
    static constexpr bool value = (T == FIFOType::CTRL_FIFO);
};

template <typename DataType, FIFOType FifoType, int Depth, int Period, int LocalDepth_unused>
struct DataFIFO<DataType, FifoType, Depth, Period, LocalDepth_unused,
                typename std::enable_if<IsCtrlFiFo<FifoType>::value>::type> {
    static constexpr int fifoDepth = Depth;
    static constexpr int fifoPeriod = Period;
    static constexpr FIFOType fifoType = FifoType;
    bool ctrlSignal = false; // control signal to be sent through the FIFO
    uint32_t fifoBase = 0x0; // fifo base address in global memory or scalar buffer

    PTO_INTERNAL DataFIFO(uint32_t base) : fifoBase(base)
    {}
};

} // namespace pto
#endif
